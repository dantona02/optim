"""
simulate.py
    Script to run the BMCTool simulation based on a seq-file and a *.yaml config file.
"""

from pathlib import Path
from typing import Union
import numpy as np

from bmc.bmc_tool import BMCTool
from bmc.fid.sim_engine import BMCSim
from bmc.set_params import load_params
from bmc.utils.eval import plot_z, plot_sim



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
                 n_backlog: Union[str, int],
                 return_zmag: bool = False,
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
    sim = BMCSim(adc_time, sim_params, seq_file, n_backlog=n_backlog, z_positions=z_positions, webhook=webhook, **kwargs)
    sim.run_fid()
    

    if show_plot:

        time, m_z, m_z_total, m_c, m_c_total = sim.get_mag()

        if return_zmag:
            plot_sim(m_out=[m_z, m_z_total], time=time, plt_range=plt_range, iso_select=iso_select, **kwargs)
        else:
            plot_sim(m_out=[m_c, m_c_total], time=time, plt_range=plt_range, iso_select=iso_select, **kwargs)

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
