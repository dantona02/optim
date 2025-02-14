import re
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import multiprocessing
import gc

# Annahme: Diese Funktion muss für PyTorch angepasst werden
from bmc.simulate import simulate

def _run_variation(seq_path_on: str, seq_path_off: str, config_path: str, adc_time: float, 
                   z_pos: torch.Tensor, webhook: str, show_plot: bool) -> Tuple[Optional[int], Optional[float]]:
    try:
        t_ex = 0
        match = re.search(r'\d+', seq_path_on)
        if match:
            t_ex = int(match.group())

        # Annahme: simulate() gibt jetzt PyTorch Tensoren zurück
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
        t, _, _, _, m_complex_on = sim_on.get_mag(return_cest_pool=False)
        m_on = torch.abs(m_complex_on[-600:])
        
        del m_complex_on, sim_on
        torch.cuda.empty_cache()

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
        m_off = torch.abs(m_complex_off[-600:])

        del m_complex_off, sim_off
        torch.cuda.empty_cache()

        if m_on.shape[0] != 600 or m_off.shape[0] != 600:
            raise ValueError(f"Arrays too small: m_on={m_on.shape[0]}, m_off={m_off.shape[0]}")

        signal_corrected = m_on - m_off
        signal = torch.max(signal_corrected).item()

        if show_plot:
            plt.plot(t[-600:].cpu().numpy(), signal_corrected.cpu().numpy(), 'o', c='blue', markersize=1)
            plt.axhline(0, c='black')
            plt.show()

        del m_on, m_off, signal_corrected, t
        torch.cuda.empty_cache()

        return t_ex, signal

    except Exception as e:
        print(f"Error during processing: {e}")
        return None, None

def run_variation_helper(args):
    return _run_variation(*args)

def run_variation(seq_path_on: List[str], seq_path_off: List[str], config_path: str, adc_time: float, 
                  z_pos: torch.Tensor, webhook: str, num_points: int = 2, batch_size: int = 10, 
                  max_processes: int = 4, save_path: str = "results.pt", save_to_file: bool = True, 
                  show_plot: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    
    assert len(seq_path_on) == len(seq_path_off), "Input lists must have the same length"
    
    num_processes = min(mp.cpu_count(), max_processes)
    results = []

    if save_to_file:
        try:
            results = torch.load(save_path)
            print(f"Loaded previous results: {len(results)} entries")
        except FileNotFoundError:
            print("No previous results found, starting fresh.")
    else:
        print("Results will not be saved to a file.")

    processed_indices = set()
    for res in results:
        if res is not None:
            if isinstance(res[0], str):
                match = re.search(r'\d+', res[0])
                if match:
                    processed_indices.add(int(match.group()))
            elif isinstance(res[0], (int, float)):
                processed_indices.add(int(res[0]))

    print(f"Already processed indices: {len(processed_indices)}")

    for batch_start in range(0, num_points, batch_size):
        batch_end = min(batch_start + batch_size, num_points)

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
                        webhook,
                        show_plot
                    ))

        if not args_list:
            continue 

        with mp.Pool(processes=num_processes) as pool:
            for br in pool.imap(run_variation_helper, args_list, chunksize=1):
                if br and br[0] is not None:
                    results.append(br)
                    processed_indices.add(br[0])
                    if save_to_file:
                        torch.save(results, save_path)
                    print(f"Result processed: {br} | Total entries: {len(results)}")

    if not results:
        raise ValueError("No valid results obtained")

    t_ex, signal = zip(*results)
    return torch.tensor(t_ex), torch.tensor(signal)

def run_sim(seq_path, config_path, adc_time, z_pos, webhook, plt_range, get_t):
    """
    Führt eine Simulation durch und gibt je nach get_t entweder
    (t, m, m_plot) oder nur m zurück.
    """
    sim = simulate(
        config_file=config_path,
        seq_file=seq_path,
        adc_time=adc_time,
        z_positions=z_pos,
        return_zmag=False,
        iso_select=[0],
        show_plot=False,
        n_backlog=2,
        webhook=webhook,
        plt_range=plt_range
    )
    # get_mag liefert üblicherweise (t, _, _, _, m_complex)
    res = sim.get_mag(return_cest_pool=False)
    if get_t:
        _, _, _, _, m_complex = res
        t, m_plot = sim.get_exact()
    else:
        _, _, _, _, m_complex = res

    # Konvertiere das komplexe Ergebnis in einen Torch-Tensor und berechne den Betrag
    m = torch.abs(m_complex)
    return (t, m, m_plot) if get_t else m

def run_variation_parallel(seq_path_on, seq_path_off, config_path, adc_time, z_pos, webhook, show_plot, save_plot):
    # Extrahiere t_ex aus dem Dateinamen der on-Sequenz
    t_ex = 0
    match = re.search(r'\d+', seq_path_on)
    if match:
        t_ex = int(match.group())

    # Starte beide Simulationen parallel mittels Multiprocessing
    num_processes = 2  # ein Prozess für "on", einer für "off"
    pool = multiprocessing.Pool(processes=num_processes)
    
    async_on = pool.apply_async(
        run_sim, args=(seq_path_on, config_path, adc_time, z_pos, webhook, None, True)
    )
    async_off = pool.apply_async(
        run_sim, args=(seq_path_off, config_path, adc_time, z_pos, webhook, None, True)
    )
    pool.close()
    pool.join()

    on_result = async_on.get()
    off_result = async_off.get()

    t, m_on, m_plot_on = on_result
    _, m_off, m_plot_off = off_result

    if m_on.size(0) != m_off.size(0):
        raise ValueError(f"Arrays m_on and m_off do not have the same length (m_on={m_on.size(0)}, m_off={m_off.size(0)})!")

    signal_corrected = m_on - m_off

    signal_plot = [torch.abs(mp_on) - torch.abs(mp_off) for mp_on, mp_off in zip(m_plot_on, m_plot_off)]
    signal = torch.max(signal_corrected)

    if show_plot:
        fig, ax = plt.subplots(dpi=150)
        # Konvertiere t in ein NumPy-Array zum Plotten
        # Iteriere über die einzelnen Zeitpunkte und plotte das jeweilige Slice
        for index, t_val in enumerate(t):
            ax.plot(t_val.cpu().numpy(), signal_plot[index].cpu().numpy(), 'o', markersize=1)
        ax.axhline(0, c='black')
        ax.set_xlim(0.00, 0.005)
        ax.set_ylim(-0.0008, 0.0003)
        plt.show()
        if save_plot:
            fig.savefig(f"/Users/danielmiksch/Downloads/racete_0ppm.png", dpi=300, bbox_inches='tight')

    del m_on, m_off, signal_corrected, t
    gc.collect()

    return t_ex, signal, fig