"""
simulate.py
    Script to run the BMCTool simulation based on a seq-file and a *.yaml config file.
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import torch
import pypulseq as pp
from tqdm import tqdm

from bmc.bmc_tool import BMCTool, prep_rf_simulation
from bmc.fid.engine import BMCSim
from bmc.set_params import load_params
from bmc.solver import BlochMcConnellSolver
from bmc.utils.eval import plot_z, plot_sim
from bmc.utils.global_device import GLOBAL_DEVICE


def simulate_zspec(
    config_file: Union[str, Path],
    seq_file: Union[str, Path],
    show_plot: bool = False,
    norm_threshold: float = 295,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Z-spectrum from a saturation+ADC sequence.

    Each repetition in the seq file must follow the pattern:
        [RF saturation block] → [pseudo-ADC]

    Magnetization is reset to thermal equilibrium after each ADC when
    reset_init_mag is True in the config (default). If the first offset
    is a far off-resonance M0 reference (|offset| > norm_threshold [ppm]),
    the spectrum is normalized by that M0 value automatically.

    Parameters
    ----------
    config_file : Union[str, Path]
        Path to the YAML config file.
    seq_file : Union[str, Path]
        Path to the .seq file.
    show_plot : bool, optional
        Plot the Z-spectrum after simulation, by default False
    norm_threshold : float, optional
        Offsets beyond this [ppm] are treated as M0 reference, by default 295

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (offsets_ppm, m_z) — offset array and normalized Mz values.
    """
    config_file = Path(config_file).resolve()
    seq_file = Path(seq_file).resolve()

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not seq_file.exists():
        raise FileNotFoundError(f"Seq file not found: {seq_file}")

    params = load_params(config_file)

    seq = pp.Sequence()
    seq.read(seq_file)
    try:
        defs = seq.definitions
        block_events = seq.block_events
    except AttributeError:
        defs = seq.dict_definitions
        block_events = seq.dict_block_events

    offsets_ppm = np.array(defs["offsets_ppm"])
    n_offsets = offsets_ppm.size

    solver = BlochMcConnellSolver(
        params=params,
        n_offsets=1,
        z_positions=torch.tensor([0.0], dtype=torch.float64, device=GLOBAL_DEVICE),
    )

    m_init = params.m_vec.copy()
    m_out = np.zeros([m_init.shape[0], n_offsets])

    def _reset_mag() -> torch.Tensor:
        return torch.tensor(
            m_init[np.newaxis, np.newaxis, :, np.newaxis],
            dtype=torch.float64, device=GLOBAL_DEVICE,
        )

    mag = _reset_mag()
    current_adc = 0

    for block_event in tqdm(block_events, desc="Z-spectrum"):
        block = seq.get_block(block_event)

        if block.adc is not None:
            m_out[:, current_adc] = mag.squeeze().cpu().numpy()
            current_adc += 1
            if current_adc < n_offsets and params.options["reset_init_mag"]:
                mag = _reset_mag()

        elif block.rf is not None:
            amp_, ph_, dtp_, delay_, rf_freq_ = prep_rf_simulation(
                block, params.options["max_pulse_samples"]
            )
            for i in range(amp_.numel()):
                solver.update_matrix(rf_amp=amp_[i], rf_phase=ph_[i], rf_freq=rf_freq_)
                mag = solver.solve_equation(mag=mag, dtp=dtp_)
            if delay_ > 0:
                solver.update_matrix(0, 0, 0)
                mag = solver.solve_equation(mag=mag, dtp=delay_)

        elif block.gz is not None:
            solver.update_matrix(0, 0, 0)
            mag = solver.solve_equation(mag=mag, dtp=block.block_duration)
            for j in range((len(params.cest_pools) + 1) * 2):
                mag[0, 0, j, 0] = 0.0  # complete spoiling

        elif hasattr(block, "block_duration") and block.block_duration != "0":
            solver.update_matrix(0, 0, 0)
            mag = solver.solve_equation(mag=mag, dtp=block.block_duration)

    mz_loc = params.mz_loc
    if n_offsets > 1 and abs(offsets_ppm[0]) >= norm_threshold:
        m0 = m_out[mz_loc, 0]
        offsets_out = offsets_ppm[1:]
        m_z = np.abs(m_out[mz_loc, 1:]) / m0 if m0 != 0 else np.abs(m_out[mz_loc, 1:])
    else:
        offsets_out = offsets_ppm
        m_z = np.abs(m_out[mz_loc, :])

    if show_plot:
        plot_z(m_z=m_z, offsets=offsets_out, **kwargs)

    return offsets_out, m_z


def simulate_(config_file: Union[str, Path], seq_file: Union[str, Path], show_plot: bool = False, **kwargs) -> BMCTool:
    """
    simulate Run BMCTool simulation based on given seq-file and config file.

    Parameters
    ----------
    config_file : Union[str, Path]
        Path to the config file.
    seq_file : Union[str, Path]
        Path to the seq-file.
    show_plot : bool, optional
        Flag to activate plotting of simulated data, by default False

    Returns
    -------
    BMCTool
        BMCTool object containing the simulation results.

    Raises
    ------
    FileNotFoundError
        If the config_file or seq_file is not found.
    """
    if not Path(config_file).exists():
        raise FileNotFoundError(f"File {config_file} not found.")

    if not Path(seq_file).exists():
        raise FileNotFoundError(f"File {seq_file} not found.")

    # load config file(s)
    sim_params = load_params(config_file)

    # create BMCTool object and run simulation
    sim = BMCTool(sim_params, seq_file, **kwargs)
    sim.run()
    

    if show_plot:
        if "offsets" in kwargs:
            offsets = kwargs.pop("offsets")
            _, m_z = sim.get_zspec()
        else:
            offsets, m_z = sim.get_zspec()

        plot_z(m_z=m_z, offsets=offsets, **kwargs)

    return sim


def simulate(config_file: Union[str, Path],
                 seq_file: Union[str, Path],
                 z_positions: np.ndarray,
                 adc_time: np.float64,
                 iso_select: Union[list, tuple],
                 return_zmag: bool = False,
                 write_all_mag: bool = False,
                 show_plot: bool = False,
                 webhook: bool = False,
                 plt_range: list = [0, 5],
                 **kwargs) -> BMCSim:
    """
    simulate Run BMCTool simulation based on given seq-file and config file.

    Parameters
    ----------
    config_file : Union[str, Path]
        Path to the config file.
    seq_file : Union[str, Path]
        Path to the seq-file.
    show_plot : bool, optional
        Flag to activate plotting of simulated data, by default False

    Returns
    -------
    BMCTool
        BMCTool object containing the simulation results.

    Raises
    ------
    FileNotFoundError
        If the config_file or seq_file is not found.
    """

    seq_file = Path(seq_file).resolve()
    config_file = Path(config_file).resolve()

    if not Path(config_file).exists():
        raise FileNotFoundError(f"File {config_file} not found.")

    if not Path(seq_file).exists():
        raise FileNotFoundError(f"File {seq_file} not found.")

    # load config file(s)
    sim_params = load_params(config_file)

    # create BMCTool object and run simulation
    sim = BMCSim(adc_time, sim_params, seq_file, write_all_mag=write_all_mag, z_positions=z_positions, webhook=webhook, **kwargs)
    sim.run_fid()
    

    if show_plot:

        time, m_z, m_z_total, m_c, m_c_total = sim.get_mag()

        if return_zmag:
            plot_sim(m_out=[m_z.cpu(), m_z_total.cpu()], time=time.cpu(), plt_range=plt_range, iso_select=iso_select, **kwargs)
        else:
            plot_sim(m_out=[m_c.cpu(), m_c_total.cpu()], time=time.cpu(), plt_range=plt_range, iso_select=iso_select, **kwargs)

    return sim


# def sim_example() -> None:
#     """
#     Function to run an example WASABI simulation.
#     """
#     seq_file = Path(__file__).parent / "library" / "seq-library" / "WASABI.seq"
#     config_file = Path(__file__).parent / "library" / "sim-library" / "config_wasabi.yaml"

#     simulate(
#         config_file=config_file, seq_file=seq_file, show_plot=True, title="WASABI example spectrum", normalize=True
#     )



# if __name__ == "__main__":
#     sim_example()
