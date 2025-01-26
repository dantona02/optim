import re
import numpy as np
import multiprocessing
from bmc.simulate import simulate
import gc

def _run_variation(seq_path_on, seq_path_off, config_path, adc_time, z_pos, webhook):
    try:
        t_ex = 0

        match = re.search(r'\d+', seq_path_on)
        if match:
            t_ex = int(match.group())

        sim_on = simulate(
            config_file=config_path,
            seq_file=seq_path_on,
            adc_time=adc_time,
            z_positions=z_pos,
            return_zmag=False,
            iso_select=[0],
            show_plot=False,
            n_backlog=2,
            webhook=webhook,
            plt_range=None
        )
        _, _, _, _, m_complex_on = sim_on.get_mag(return_cest_pool=False)
        m_on = np.abs(m_complex_on[-600:])
        
        del m_complex_on, sim_on
        gc.collect()

        sim_off = simulate(
            config_file=config_path,
            seq_file=seq_path_off,
            adc_time=adc_time,
            z_positions=z_pos,
            return_zmag=False,
            iso_select=[0],
            show_plot=False,
            n_backlog=2,
            webhook=webhook,
            plt_range=None
        )
        _, _, _, _, m_complex_off = sim_off.get_mag(return_cest_pool=False)
        m_off = np.abs(m_complex_off[-600:])

        del m_complex_off, sim_off
        gc.collect()

        if len(m_on) != 600 or len(m_off) != 600:
            raise ValueError(f"Arrays sind zu klein: m_on={len(m_on)}, m_off={len(m_off)}")

        signal_corrected = m_on - m_off
        signal = np.max(signal_corrected)

        del m_on, m_off, signal_corrected
        gc.collect()

        return t_ex, signal

    except Exception as e:
        print(f"Fehler bei der Verarbeitung: {e}")
        return None, None

def run_variation_helper(args):
    return _run_variation(*args)

def run_variation(seq_path_on, seq_path_off, config_path, adc_time, z_pos, webhook, num_points=2, batch_size=10, max_processes=4, save_path="results.npy"):
    assert len(seq_path_on) == len(seq_path_off), "Eingabelisten müssen die gleiche Länge haben"
    
    num_processes = min(multiprocessing.cpu_count(), max_processes)  # Begrenze Prozesse
    results = []

    # Lade vorherige Ergebnisse, falls vorhanden
    try:
        results = np.load(save_path, allow_pickle=True).tolist()
        print(f"Geladene Ergebnisse: {len(results)}")
    except FileNotFoundError:
        print("Keine vorherigen Ergebnisse gefunden, starte neu.")
    
    # Extrahiere bereits verarbeitete t_ex-Werte
    processed_indices = set()
    for res in results:
        if res is not None:
            if isinstance(res[0], str):  # Prüfen, ob res[0] ein String (Dateiname) ist
                match = re.search(r'\d+', res[0])  # Nur wenn es ein String ist
                if match:
                    processed_indices.add(int(match.group()))
            elif isinstance(res[0], (int, float)):  # Falls es bereits ein Index ist
                processed_indices.add(int(res[0]))

    print(f"Bereits verarbeitete Indizes: {len(processed_indices)}")

    # Batchweise Verarbeitung
    for batch_start in range(0, num_points, batch_size):
        batch_end = min(batch_start + batch_size, num_points)

        # Überprüfe, ob Dateien basierend auf t_ex noch verarbeitet werden müssen
        args_list = []
        for i in range(batch_start, batch_end):
            match = re.search(r'\d+', seq_path_on[i])
            if match:
                t_ex = int(match.group())
                if t_ex not in processed_indices:
                    args_list.append((
                        seq_path_on[i],
                        seq_path_off[i],
                        config_path,
                        adc_time,
                        z_pos,
                        webhook
                    ))

        if not args_list:
            continue 

        with multiprocessing.Pool(processes=num_processes) as pool:
            for br in pool.imap(run_variation_helper, args_list, chunksize=1):
                if br and br[0] is not None:
                    results.append(br)
                    processed_indices.add(br[0])
                    np.save(save_path, results)
                    print(f"Ergebnis gespeichert: {br} | {len(results)} Einträge")

    if not results:
        raise ValueError("Keine gültigen Ergebnisse")

    t_ex, signal = zip(*results)
    return np.array(t_ex), np.array(signal)