import re
import numpy as np
import multiprocessing
from bmc.simulate import simulate

def _run_variation(seq_path_on, seq_path_off, config_path, adc_time, z_pos, webhook):
    try:
        signal = 0
        t_ex = 0
        match = re.search(r'\d+', seq_path_on)
        if match:
            t_ex = int(match.group())

        simOn = simulate(config_file=config_path, 
                         seq_file=seq_path_on, 
                         adc_time=adc_time,
                         z_positions=z_pos,
                         return_zmag=False,
                         iso_select=[0],
                         show_plot=False,
                         write_all_mag=True,
                         webhook=webhook,
                         plt_range=None)

        simOff = simulate(config_file=config_path, 
                          seq_file=seq_path_off, 
                          adc_time=adc_time,
                          z_positions=z_pos,
                          return_zmag=False,
                          iso_select=[0],
                          show_plot=False,
                          write_all_mag=True,
                          webhook=webhook,
                          plt_range=None)

        _, _, _, _, m_complex_on = simOn.get_mag(return_cest_pool=False)
        _, _, _, _, m_complex_off = simOff.get_mag(return_cest_pool=False)

        assert len(m_complex_on) >= 600 and len(m_complex_off) >= 600, \
            f"Arrays sind zu klein: m_complex_on={len(m_complex_on)}, m_complex_off={len(m_complex_off)}"

        m_on = np.abs(m_complex_on)[-600:]
        m_off = np.abs(m_complex_off)[-600:]
        signal_corrected = m_on - m_off
        signal = np.max(signal_corrected)
        return t_ex, signal
    except Exception as e:
        print(f"Fehler bei der Verarbeitung: {e}")
        return None, None

def run_variation(seq_path_on, seq_path_off, config_path, adc_time, z_pos, webhook, num_points=2, batch_size=10, max_processes=4):
    assert len(seq_path_on) == len(seq_path_off), "Eingabelisten müssen die gleiche Länge haben"
    
    num_processes = min(multiprocessing.cpu_count(), max_processes)  # Begrenze Prozesse
    results = []

    # Batchweise Verarbeitung
    for batch_start in range(0, num_points, batch_size):
        batch_end = min(batch_start + batch_size, num_points)
        args_list = [
            (seq_path_on[i], seq_path_off[i], config_path, adc_time, z_pos, webhook)
            for i in range(batch_start, batch_end)
        ]

        with multiprocessing.Pool(num_processes) as pool:
            batch_results = pool.starmap(_run_variation, args_list)
        results.extend(batch_results)

    # Filtere ungültige Ergebnisse
    results = [res for res in results if res is not None]
    if not results:
        raise ValueError("Keine gültigen Ergebnisse")

    t_ex, signal = zip(*results)
    return np.array(t_ex), np.array(signal)