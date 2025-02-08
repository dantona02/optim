"""
eval.py
    Functions for evaluation and visualization.
"""
from typing import Tuple, Union
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib import colors as mcolors
from manim import color_gradient, PURE_BLUE, PURE_GREEN, PURE_RED
from bmc.optimize.optimizer import BMCSim


def calc_mtr_asym(m_z: np.ndarray, offsets: np.ndarray, n_interp: int = 1000) -> np.ndarray:
    """
    calc_mtr_asym Calculate MTRasym from the magnetization vector.

    Parameters
    ----------
    z : np.ndarray
        Magnetization values.
    offsets : np.ndarray
        Offset values.
    n_interp : int, optional
        Number of interpolation steps, by default 1000

    Returns
    -------
    np.ndarray
        Array containing the MTRasym values.
    """
    x_interp = np.linspace(np.min(offsets), np.max(np.absolute(offsets)), n_interp)
    y_interp = np.interp(x_interp, offsets, m_z)
    asym = y_interp[::-1] - y_interp
    return np.interp(offsets, x_interp, asym)


def normalize_data(
    m_z: np.ndarray, offsets: np.ndarray, threshold: Union[int, float, list, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    normalize_data Normalize magnetization values by the mean of values corresponding to offsets exceeding the given threshold.

    Parameters
    ----------
    m_z : np.ndarray
        Magnetization values.
    offsets : np.ndarray
        Offset values.
    threshold : Union[int, float, list, np.ndarray]
        Threshold for data splitting. If int or float, the absolute value is used. If list or np.ndarray, the min and max value are used.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the normalized magnetization vector and the corresponding offsets.
    """
    offsets, data, norm = split_data(m_z, offsets, threshold)

    if norm is not None:
        m_z = np.divide(data, np.mean(norm), out=np.zeros_like(data), where=np.mean(norm) != 0)

    return m_z, offsets


def split_data(
    m_z: np.ndarray, offsets: np.ndarray, threshold: Union[int, float, list, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    """
    split_data Split magnetization vector into data and normalization data.

    Parameters
    ----------
    m_z : np.ndarray
        Magnetization values.
    offsets : np.ndarray
        Offset values.
    threshold : Union[int, float, list, np.ndarray]
        Threshold for data splitting. If int or float, the absolute value is used. If list or np.ndarray, the min and max value are used.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]
        Tuple containing offsets, data and normalization data.

    Raises
    ------
    TypeError
        If threshold is not of type int, float, list or np.ndarray.
    """
    if isinstance(threshold, (int, float)):
        th_high = np.abs(threshold)
        th_low = -th_high
    elif isinstance(threshold, list) and len(threshold) == 2:
        th_high = max(threshold)
        th_low = min(threshold)
    elif isinstance(threshold, np.ndarray) and threshold.size == 2:
        th_high = max(threshold)
        th_low = min(threshold)
    else:
        raise TypeError(f"Threshold of type '{type(threshold)}' not supported.")

    idx_data = np.where(np.logical_and(offsets > th_low, offsets < th_high))
    idx_norm = np.where(np.logical_or(offsets <= th_low, offsets >= th_high))

    if idx_norm[0].size == 0:
        return offsets, m_z, None

    offsets = offsets[idx_data]
    data = m_z[idx_data]
    norm = m_z[idx_norm]

    return offsets, data, norm


def plot_z(
    m_z: np.ndarray,
    offsets: Union[np.ndarray, None] = None,
    normalize: bool = False,
    norm_threshold: Union[int, float, list, np.ndarray] = 295,
    invert_ax: bool = True,
    plot_mtr_asym: bool = False,
    title: str = "spectrum",
    x_label: str = "offsets [ppm]",
    y_label: str = "signal",
) -> Figure:
    """
    plot_z Plot Z-spectrum according to the given parameters.

    Parameters
    ----------
    m_z : np.array
        Magnetization values.
    offsets : np.array, optional
        Offset values, by default None
    normalize : bool, optional
        Flag to activate normalization of magnetization values, by default False
    norm_threshold : Union[int, float, list, np.ndarray], optional
        Threshold used for normalization, by default 295
    invert_ax : bool, optional
        Flag to activate the inversion of the x-axis, by default True
    plot_mtr_asym : bool, optional
        Flag to activate the plotting of MTRasym values, by default False
    title : str, optional
        Figure title, by default "spectrum"
    x_label : str, optional
        Label of x-axis, by default "offsets [ppm]"
    y_label : str, optional
        Label of y-axis, by default "signal"

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    if offsets is None:
        offsets = range(len(m_z))

    if normalize:
        m_z, offsets = normalize_data(m_z, offsets, norm_threshold)

    fig, ax1 = plt.subplots()
    ax1.set_ylim([round(min(m_z) - 0.05, 2), round(max(m_z) + 0.05, 2)])
    ax1.set_ylabel(y_label, color="b")
    ax1.set_xlabel(x_label)
    plt.plot(offsets, m_z, ".--", label="$Z$", color="b")
    if invert_ax:
        plt.gca().invert_xaxis()
    ax1.tick_params(axis="y", labelcolor="b")

    if plot_mtr_asym:
        mtr_asym = calc_mtr_asym(m_z=m_z, offsets=offsets)

        ax2 = ax1.twinx()
        ax2.set_ylim([round(min(mtr_asym) - 0.01, 2), round(max(mtr_asym) + 0.01, 2)])
        ax2.set_ylabel("$MTR_{asym}$", color="y")
        ax2.plot(offsets, mtr_asym, label="$MTR_{asym}$", color="y")
        ax2.tick_params(axis="y", labelcolor="y")
        fig.tight_layout()

    plt.title(title)
    plt.show()

    return fig

from matplotlib.colors import TwoSlopeNorm

def plot_sim( 
    m_out: Union[np.ndarray, list, tuple], 
    time: Union[np.ndarray, None] = None,
    plt_range: list = None,
    iso_select: Union[list, tuple] = None,  # Neues Argument für Isochromaten-Auswahl
    title_signal: str = "Signal", 
    title_phase: str = "Phase",
    x_label: str = "time [s]", 
    y_label_signal: str = "Signal Amplitude",
    y_label_phase: str = "Phase [radians]",
    colorbar_label: str = "Isochromat Index"
) -> None:
    """
    Plots Signal and Phase with a centered color gradient and colorbar showing Isochromat indices.
    The second vector in m_out is always plotted in black.
    """
    if time is None:
        time = np.arange(m_out.shape[-1])
    
    if not isinstance(m_out, (list, tuple)):
        m_out = [m_out]

    n_isochromats = m_out[0].shape[0]
    center_idx = n_isochromats // 2

    # Dynamische Auswahl der Isochromaten
    if iso_select is not None:
        if isinstance(iso_select, tuple):  # Bereichsangabe (-2, 2)
            selected_indices = [
                i for i in range(n_isochromats) 
                if iso_select[0] <= (i - center_idx) <= iso_select[1]
            ]
        elif isinstance(iso_select, list):  # Einzelne Indizes
            selected_indices = [
                i + center_idx for i in iso_select if -center_idx <= i <= center_idx
            ]
        else:
            raise ValueError("iso_select must be a list of indices or a tuple (range).")
    else:
        selected_indices = range(n_isochromats)  # Standard: Alle Isochromaten
    
    # Farbverlauf exakt wie in Animation Class
    manim_colors = color_gradient([PURE_BLUE, PURE_GREEN, PURE_RED], n_isochromats)
    hex_colors = [mcolors.to_hex(c.to_rgb()) for c in manim_colors]

    # Plot vorbereiten
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Zwei Subplots nebeneinander

    # Signal Plot (Magnitude)
    for idx in selected_indices:
        ax1.plot(
            time, 
            np.abs(m_out[0][idx]), 
            ".--",                    # Punkte mit gestrichelter Linie
            color=hex_colors[idx], 
            label=f"Iso {idx - center_idx}",
            markersize=4,
            linewidth=.6
        )
    # Der zweite Vektor wird immer schwarz geplottet
    ax1.plot(
        time, 
        np.abs(m_out[1]), 
        ".--", 
        color="black", 
        label="Second Vector",
        markersize=4,
        linewidth=.6
    )
    ax1.set_title(title_signal)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label_signal)
    ax1.set_xlim(plt_range)
    ax1.grid(True)

    # Phase Plot
    for idx in selected_indices:
        ax2.plot(
            time, 
            np.angle(m_out[0][idx]), 
            ".--",                    # Punkte mit gestrichelter Linie
            color=hex_colors[idx], 
            label=f"Iso {idx - center_idx}",
            markersize=4,
            linewidth=.6
        )
    # Der zweite Vektor wird immer schwarz geplottet
    ax2.plot(
        time, 
        np.angle(m_out[1]), 
        ".--", 
        color="black", 
        label="Second Vector",
        markersize=4,
        linewidth=.6
    )
    ax2.set_title(title_phase)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label_phase)
    ax2.set_xlim(plt_range)
    ax2.grid(True)

    # Farbleiste zentriert bei 0
    center_idx = n_isochromats // 2
    vmin, vcenter, vmax = -center_idx, 0, center_idx
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=mcolors.ListedColormap(hex_colors), norm=norm)
    sm.set_array([])

    # Farbverlauf hinzufügen
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation="horizontal", label=colorbar_label)
    ticks = [-center_idx, 0, center_idx]
    cbar.set_ticks(ticks)
    cbar.ax.set_position([0.35, -5.3, 0.32, 5.2])

    plt.tight_layout()
    plt.show()

    return fig



def calculate_flip_angle(rf_amplitudes_hz: torch.Tensor, dt: float, sim_instance: BMCSim) -> float:
    """
    Calculate the flip angle of a given RF pulse tensor

    Args:
        rf_amplitudes_hz (torch.Tensor): RF amplitudes in Hz
        dt (float): Time step of puls fragment in seconds

    Returns:
        float: Flip angle in degrees
    """

    flip_angle_rad = torch.sum(rf_amplitudes_hz) * dt *  (2 * torch.pi)# dt in Sekunden
    flip_angle_deg = torch.rad2deg(flip_angle_rad).item()

    flip_angle_deg = flip_angle_deg % 360
    if flip_angle_deg > 180:
        flip_angle_deg = 360 - flip_angle_deg

    return flip_angle_deg