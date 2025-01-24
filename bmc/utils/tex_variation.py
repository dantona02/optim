import re
import numpy as np
import multiprocessing
from bmc.simulate import simulate

def _run_variation(seq_path_on: str,
                   seq_path_off: str,
                   config_path: str,
                   adc_time: float,
                   z_pos: np.ndarray,
                   webhook: bool):
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

def run_variation(seq_path_on, seq_path_off, config_path, adc_time, z_pos, webhook, num_points=2):
    assert len(seq_path_on) == len(seq_path_off), "Eingabelisten müssen die gleiche Länge haben"
    args_list = [(seq_path_on[i], seq_path_off[i], config_path, adc_time, z_pos, webhook) for i in range(num_points)]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.starmap(_run_variation, args_list)
    results = [res for res in results if res is not None]
    if not results:
        raise ValueError("Keine gültigen Ergebnisse")
    t_ex, signal = zip(*results)
    return np.array(t_ex), np.array(signal)