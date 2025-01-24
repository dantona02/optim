import re
import torch
import torch.multiprocessing as mp
from bmc.simulate import simulate

def _run_variation(seq_path_on: str,
                   seq_path_off: str,
                   config_path: str,
                   adc_time: float,
                   z_pos: torch.Tensor,
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

        m_on = torch.abs(m_complex_on[-600:])
        m_off = torch.abs(m_complex_off[-600:])
        
        signal_corrected = m_on - m_off
        signal = torch.max(signal_corrected).item()
        return t_ex, signal
    except Exception as e:
        print(f"Fehler bei der Verarbeitung: {e}")
        return None, None

def run_variation(seq_path_on, seq_path_off, config_path, adc_time, z_pos, webhook, num_points=2):
    assert len(seq_path_on) == len(seq_path_off), "Eingabelisten müssen die gleiche Länge haben"

    args_list = [(seq_path_on[i], seq_path_off[i], config_path, adc_time, z_pos, webhook) for i in range(num_points)]

    # Use PyTorch multiprocessing
    mp.set_start_method('spawn', force=True)
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(_run_variation, args_list)

    results = [res for res in results if res is not None]
    if not results:
        raise ValueError("Keine gültigen Ergebnisse")

    t_ex, signal = zip(*results)
    return torch.tensor(t_ex, dtype=torch.float32), torch.tensor(signal, dtype=torch.float32)

# import re
# import torch
# import multiprocessing
# import os
# from bmc.simulate import simulate

# def _run_variation(seq_path_on: str,
#                    seq_path_off: str,
#                    config_path: str,
#                    adc_time: float,
#                    z_pos: torch.Tensor,
#                    webhook: bool):
#     try:
#         signal = 0
#         t_ex = 0

#         seq_path_on = os.path.abspath(seq_path_on)
#         seq_path_off = os.path.abspath(seq_path_off)
#         config_path = os.path.abspath(config_path)

#         match = re.search(r'\d+', seq_path_on)
#         if match:
#             t_ex = int(match.group())

#         simOn = simulate(config_file=config_path, 
#                          seq_file=seq_path_on, 
#                          adc_time=adc_time,
#                          z_positions=z_pos,
#                          return_zmag=False,
#                          iso_select=[0],
#                          show_plot=False,
#                          write_all_mag=True,
#                          webhook=webhook,
#                          plt_range=None)

#         simOff = simulate(config_file=config_path, 
#                           seq_file=seq_path_off, 
#                           adc_time=adc_time,
#                           z_positions=z_pos,
#                           return_zmag=False,
#                           iso_select=[0],
#                           show_plot=False,
#                           write_all_mag=True,
#                           webhook=webhook,
#                           plt_range=None)

#         _, _, _, _, m_complex_on = simOn.get_mag(return_cest_pool=False)
#         _, _, _, _, m_complex_off = simOff.get_mag(return_cest_pool=False)

#         assert len(m_complex_on) >= 600 and len(m_complex_off) >= 600, \
#             f"Arrays sind zu klein: m_complex_on={len(m_complex_on)}, m_complex_off={len(m_complex_off)}"

#         m_on = torch.abs(m_complex_on[-600:])
#         m_off = torch.abs(m_complex_off[-600:])
        
#         signal_corrected = m_on - m_off
#         signal = torch.max(signal_corrected).item()
#         return t_ex, signal
#     except Exception as e:
#         print(f"Fehler bei der Verarbeitung: {e}")
#         return None, None

# def run_variation(seq_path_on, seq_path_off, config_path, adc_time, z_pos, webhook, num_points=2):
#     assert len(seq_path_on) == len(seq_path_off), "Eingabelisten müssen die gleiche Länge haben"


#     # Argumentliste für die Prozesse erstellen
#     args_list = [(seq_path_on[i], seq_path_off[i], config_path, adc_time, z_pos, webhook) for i in range(num_points)]

#     # Multiprocessing mit der 'spawn'-Methode
#     multiprocessing.set_start_method('spawn', force=True)
#     with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
#         results = pool.starmap(_run_variation, args_list)

#     # Ergebnisse filtern
#     results = [res for res in results if res is not None]
#     if not results:
#         raise ValueError("Keine gültigen Ergebnisse")

#     t_ex, signal = zip(*results)
#     return torch.tensor(t_ex, dtype=torch.float32), torch.tensor(signal, dtype=torch.float32)

# import re
# import torch
# import torch.multiprocessing as mp
# from bmc.simulate import simulate

# def _run_variation(args):
#     """
#     Führt eine einzelne Variation aus. Erwartet die Argumente als Tupel.
#     """
#     try:
#         seq_path_on, seq_path_off, config_path, adc_time, z_pos_cpu, webhook = args

#         # Z_pos auf die GPU verschieben (pro Subprozess)
#         z_pos = z_pos_cpu.to('cuda')

#         signal = 0
#         t_ex = 0
#         match = re.search(r'\d+', seq_path_on)
#         if match:
#             t_ex = int(match.group())

#         simOn = simulate(config_file=config_path, 
#                          seq_file=seq_path_on, 
#                          adc_time=adc_time,
#                          z_positions=z_pos,
#                          return_zmag=False,
#                          iso_select=[0],
#                          show_plot=False,
#                          write_all_mag=True,
#                          webhook=webhook,
#                          plt_range=None)

#         simOff = simulate(config_file=config_path, 
#                           seq_file=seq_path_off, 
#                           adc_time=adc_time,
#                           z_positions=z_pos,
#                           return_zmag=False,
#                           iso_select=[0],
#                           show_plot=False,
#                           write_all_mag=True,
#                           webhook=webhook,
#                           plt_range=None)

#         _, _, _, _, m_complex_on = simOn.get_mag(return_cest_pool=False)
#         _, _, _, _, m_complex_off = simOff.get_mag(return_cest_pool=False)

#         assert len(m_complex_on) >= 600 and len(m_complex_off) >= 600, \
#             f"Arrays sind zu klein: m_complex_on={len(m_complex_on)}, m_complex_off={len(m_complex_off)}"

#         m_on = torch.abs(torch.tensor(m_complex_on[-600:], dtype=torch.cfloat, device='cuda'))
#         m_off = torch.abs(torch.tensor(m_complex_off[-600:], dtype=torch.cfloat, device='cuda'))
#         signal_corrected = m_on - m_off
#         signal = torch.max(signal_corrected).item()

#         return t_ex, signal
#     except Exception as e:
#         print(f"Fehler bei der Verarbeitung: {e}")
#         return None, None

# def run_variation(seq_path_on, seq_path_off, config_path, adc_time, z_pos, webhook, num_points=2):
#     """
#     Führt Variationen parallel aus.
#     """
#     assert len(seq_path_on) == len(seq_path_off), "Eingabelisten müssen die gleiche Länge haben"

#     # Argumentliste vorbereiten
#     z_pos_cpu = z_pos.to('cpu')  # z_pos vorab auf CPU verschieben
#     args_list = [(seq_path_on[i], seq_path_off[i], config_path, adc_time, z_pos_cpu, webhook) for i in range(num_points)]

#     # Multiprocessing mit 'spawn'-Methode
#     mp.set_start_method('spawn', force=True)

#     # Ergebnisse mit Pool parallel verarbeiten
#     results = []
#     with mp.Pool(processes=mp.cpu_count()) as pool:
#         for result in pool.imap_unordered(_run_variation, args_list):
#             if result is not None:
#                 results.append(result)

#     if not results:
#         raise ValueError("Keine gültigen Ergebnisse")

#     t_ex, signal = zip(*results)
#     return torch.tensor(t_ex, dtype=torch.float32).to('cuda'), torch.tensor(signal, dtype=torch.float32).to('cuda')