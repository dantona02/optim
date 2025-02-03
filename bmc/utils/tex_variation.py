import re
import numpy as np
import multiprocessing
from bmc.simulate import simulate
import gc

def _run_variation(seq_path_on, seq_path_off, config_path, adc_time, z_pos, webhook):
    try:
        t_ex = 0

        # Extract a numerical value from the sequence file name
        match = re.search(r'\d+', seq_path_on)
        if match:
            t_ex = int(match.group())

        # Run the simulation for the "on" sequence
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

        # Run the simulation for the "off" sequence
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

        # Check that the arrays have the expected size
        if len(m_on) != 600 or len(m_off) != 600:
            raise ValueError(f"Arrays too small: m_on={len(m_on)}, m_off={len(m_off)}")

        # Calculate the corrected signal and extract the maximum value
        signal_corrected = m_on - m_off
        signal = np.max(signal_corrected)

        del m_on, m_off, signal_corrected
        gc.collect()

        return t_ex, signal

    except Exception as e:
        print(f"Error during processing: {e}")
        return None, None

def run_variation_helper(args):
    return _run_variation(*args)

def run_variation(seq_path_on, seq_path_off, config_path, adc_time, z_pos, webhook,
                  num_points=2, batch_size=10, max_processes=4, save_path="results.npy",
                  save_to_file=True):
    """
    Executes simulation variations.

    Parameters:
      - seq_path_on: List of file paths for the "on" sequences.
      - seq_path_off: List of file paths for the "off" sequences.
      - config_path: Path to the configuration file.
      - adc_time: ADC time parameter for the simulation.
      - z_pos: z positions used in the simulation.
      - webhook: Webhook URL or parameter.
      - num_points: Total number of points to process.
      - batch_size: Number of points to process in one batch.
      - max_processes: Maximum number of parallel processes.
      - save_path: Path to the file where results are saved.
      - save_to_file: If True, results will be saved to the file; if False, they will be kept only in memory.
    """
    assert len(seq_path_on) == len(seq_path_off), "Input lists must have the same length"
    
    num_processes = min(multiprocessing.cpu_count(), max_processes)  # Limit the number of processes
    results = []

    # Load previous results if saving to file is enabled
    if save_to_file:
        try:
            results = np.load(save_path, allow_pickle=True).tolist()
            print(f"Loaded previous results: {len(results)} entries")
        except FileNotFoundError:
            print("No previous results found, starting fresh.")
    else:
        print("Results will not be saved to a file.")

    # Extract already processed t_ex values
    processed_indices = set()
    for res in results:
        if res is not None:
            if isinstance(res[0], str):  # Check if res[0] is a string (e.g., a filename)
                match = re.search(r'\d+', res[0])
                if match:
                    processed_indices.add(int(match.group()))
            elif isinstance(res[0], (int, float)):  # If it is already a numeric index
                processed_indices.add(int(res[0]))

    print(f"Already processed indices: {len(processed_indices)}")

    # Process in batches
    for batch_start in range(0, num_points, batch_size):
        batch_end = min(batch_start + batch_size, num_points)

        # Check if files based on t_ex still need to be processed
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
                    # Save the results only if save_to_file is True
                    if save_to_file:
                        np.save(save_path, results)
                    print(f"Result processed: {br} | Total entries: {len(results)}")

    if not results:
        raise ValueError("No valid results obtained")

    t_ex, signal = zip(*results)
    return np.array(t_ex), np.array(signal)