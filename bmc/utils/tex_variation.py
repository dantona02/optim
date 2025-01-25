import re
import psutil
import numpy as np
import multiprocessing
import time
from bmc.simulate import simulate
import gc


def _run_variation(seq_path_on, seq_path_off, config_path, adc_time, z_pos, webhook, result_queue):
    """
    Führt die Simulation aus und speichert das Ergebnis in der result_queue.
    """
    try:
        t_ex = 0
        match = re.search(r'\d+', seq_path_on)
        if match:
            t_ex = int(match.group())

        # --- Simulation "On"
        sim_on = simulate(
            config_file=config_path,
            seq_file=seq_path_on,
            adc_time=adc_time,
            z_positions=z_pos,
            return_zmag=False,
            iso_select=[0],
            show_plot=False,
            write_all_mag=True,
            webhook=webhook,
            plt_range=None
        )
        _, _, _, _, m_complex_on = sim_on.get_mag(return_cest_pool=False)
        m_on = np.abs(m_complex_on[-600:])
        del m_complex_on, sim_on
        gc.collect()

        # --- Simulation "Off"
        sim_off = simulate(
            config_file=config_path,
            seq_file=seq_path_off,
            adc_time=adc_time,
            z_positions=z_pos,
            return_zmag=False,
            iso_select=[0],
            show_plot=False,
            write_all_mag=True,
            webhook=webhook,
            plt_range=None
        )
        _, _, _, _, m_complex_off = sim_off.get_mag(return_cest_pool=False)
        m_off = np.abs(m_complex_off[-600:])
        del m_complex_off, sim_off
        gc.collect()

        # Signal berechnen
        if len(m_on) != 600 or len(m_off) != 600:
            raise ValueError(f"Arrays sind zu klein: m_on={len(m_on)}, m_off={len(m_off)}")

        signal_corrected = m_on - m_off
        signal = np.max(signal_corrected)
        del m_on, m_off, signal_corrected
        gc.collect()

        result_queue.put((t_ex, signal))
    except Exception as e:
        print(f"Fehler bei der Verarbeitung: {e}")
        result_queue.put(None)


def run_variation(
    seq_path_on,
    seq_path_off,
    config_path,
    adc_time,
    z_pos,
    webhook,
    num_points=2,
    batch_size=10,
    max_processes=4,
    save_path="results.npy"
):
    """
    Führt die '_run_variation'-Funktion aus, wobei die RAM-Auslastung
    pro Prozess dynamisch kontrolliert wird.
    """
    assert len(seq_path_on) == len(seq_path_off), "Listen müssen gleiche Länge haben"

    results = []

    # Vorherige Ergebnisse laden
    try:
        results = np.load(save_path, allow_pickle=True).tolist()
        print(f"Geladene Ergebnisse: {len(results)}")
    except FileNotFoundError:
        print("Keine vorherigen Ergebnisse gefunden, starte neu.")

    # Bereits verarbeitete Indizes
    processed_indices = set(t_ex for t_ex, _ in results if t_ex is not None)
    print(f"Bereits verarbeitete Indizes: {len(processed_indices)}")

    # Ergebnis-Queue für Prozesse
    result_queue = multiprocessing.Queue()

    # Prozesse manuell steuern
    for i in range(num_points):
        if i in processed_indices:
            continue

        # Warte, bis RAM-Auslastung unter 90% ist
        while psutil.virtual_memory().percent > 90:
            print(f"RAM {psutil.virtual_memory().percent}% > 90%: warte...")
            time.sleep(5)

        # Prozess starten
        process = multiprocessing.Process(
            target=_run_variation,
            args=(seq_path_on[i], seq_path_off[i], config_path, adc_time, z_pos, webhook, result_queue)
        )
        process.start()

        # Aktive Prozesse begrenzen
        while len(multiprocessing.active_children()) >= max_processes:
            time.sleep(1)

        # Ergebnis sammeln
        while not result_queue.empty():
            result = result_queue.get()
            if result:
                results.append(result)
                np.save(save_path, results)
                print(f"Ergebnis gespeichert: {result} | {len(results)} Einträge")

    # Alle Prozesse abschließen
    for process in multiprocessing.active_children():
        process.join()

    if not results:
        raise ValueError("Keine gültigen Ergebnisse")

    t_ex, signal = zip(*results)
    return np.array(t_ex), np.array(signal)