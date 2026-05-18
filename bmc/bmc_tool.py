"""
bmc_tool.py
    Tool to solve the Bloch-McConnell (BMC) equations using a (parallelized) eigenwert ansatz.
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
import pypulseq as pp
from pypulseq.Sequence.read_seq import __strip_line as strip_line
from tqdm import tqdm

from bmc.bmc_solver import BlochMcConnellSolver
from bmc.params import Params
import torch
import torch.nn.functional as F

from bmc.utils.global_device import GLOBAL_DEVICE


def prep_rf_simulation(block: SimpleNamespace, max_pulse_samples: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Alternative zu prep_rf_simulation, die auch 0-Werte im RF-Signal beibehält.
    Für die Berechnung von delay_after_pulse wird aber weiterhin nach Werten > 1e-6 gesucht.
    Der Rest (Blockpuls vs. shaped Pulse) bleibt unverändert.

    Parameters
    ----------
    block : SimpleNamespace
        PyPulseq block event.
    max_pulse_samples : int
        Maximum number of samples für den RF-Puls.

    Returns
    -------
    (amp_, ph_, dtp_, delay_after_pulse)
        amp_ : torch.Tensor [float64]
            Amplitudenwerte über alle Samples (ggf. resampled),
            inklusive Nullstellen.
        ph_ : torch.Tensor [float64]
            Phasenwerte (ggf. resampled).
        dtp_ : float
            Effektiver Zeitschritt.
        delay_after_pulse : float
            Verbleibende „stille“ Zeit nach dem letzten aktiven Sample.
    """
    import torch
    import torch.nn.functional as F
    from bmc.utils.global_device import GLOBAL_DEVICE

    t_tensor = torch.tensor(block.rf.t, dtype=torch.float64, device=GLOBAL_DEVICE)
    rf_signal_tensor = torch.tensor(block.rf.signal, dtype=torch.complex64, device=GLOBAL_DEVICE)

    # Neuer Code: Wandlung der Parameter in Tensoren und Nutzung eines komplexen Konstanten-Tensors
    phase = torch.tensor(block.rf.phase_offset, dtype=torch.float64, device=GLOBAL_DEVICE)
    freq  = torch.tensor(block.rf.freq_offset, dtype=torch.float64, device=GLOBAL_DEVICE)
    complex_const = torch.tensor(1j, dtype=torch.complex64, device=GLOBAL_DEVICE)
    
    w1_complex = (rf_signal_tensor) \
        * torch.exp(complex_const * phase) \
        * torch.exp(complex_const * 2 * torch.pi * freq * t_tensor)

    # amp_full = torch.real(w1_complex)
    # ph_full = torch.imag(w1_complex)

    amp_full = torch.abs(w1_complex)
    ph_full = torch.angle(w1_complex)

    idx_active = torch.nonzero(amp_full > 1e-6, as_tuple=False).squeeze()

    try:
        rf_length = amp_full.size(0)
        dtp = float(block.rf.t[1] - block.rf.t[0])
    except AttributeError:
        rf_length = amp_full.size(0)
        dtp = 1e-6

    # Berechnung des delay_after_pulse: 
    # Die Anzahl inaktiver Samples (≤ 1e-6) ganz am Ende des Signals
    if idx_active.numel() > 0:
        last_active_index = int(idx_active[-1])
        delay_after_pulse = (rf_length - last_active_index - 1) * dtp
    else:
        # Falls alles ≤ 1e-6 ist, liegt kein aktiver Puls vor: gesamte Länge = " still "
        delay_after_pulse = rf_length * dtp

    # Wir nutzen nun das komplette Signal (inkl. 0), 
    # statt "amp = amp[idx]" o.Ä. zu machen
    amp = amp_full
    ph = ph_full

    # Block pulse detection: use relative tolerance on amplitude to avoid float32
    # precision loss in exp(j*2π*f*t) for large frequency offsets causing false
    # "shaped pulse" classification of block pulses.
    amp_range = float(amp_full.max() - amp_full.min())
    amp_mean  = float(amp_full.mean())
    is_const_amp = amp_range < 1e-3 * max(amp_mean, 1e-9)
    n_unique_amp = 1 if is_const_amp else len(torch.unique(amp_full))
    n_unique = max(n_unique_amp, len(torch.unique(ph)))

    # block pulse for seq-files >= 1.4.0
    if n_unique_amp == 1 and amp.size(0) == 2:
        amp_ = amp[0].unsqueeze(0)
        ph_ = ph[0].unsqueeze(0)  # phase at t=0 = phase_offset (freq contribution = 0)
        dtp_ = dtp
        rf_freq_out = float(block.rf.freq_offset)
    # block pulse for seq-files < 1.4.0
    elif n_unique_amp == 1:
        amp_ = amp[0].unsqueeze(0)
        ph_ = ph[0].unsqueeze(0)
        dtp_ = dtp * amp.size(0)
        rf_freq_out = float(block.rf.freq_offset)
    # shaped pulse (Downsampling)
    elif n_unique > max_pulse_samples:
        sample_factor = int(torch.ceil(torch.tensor(amp.size(0) / max_pulse_samples, device=GLOBAL_DEVICE)))
        amp_ = amp[::sample_factor]
        ph_ = ph[::sample_factor]
        dtp_ = dtp * sample_factor
        rf_freq_out = 0.0  # freq already encoded in ph_ via w1_complex
    # shaped pulse (Interpolation auf max_pulse_samples)
    elif 1 < n_unique < max_pulse_samples:
        original_length = amp.size(0)

        # 1) Amplituden-Interpolation
        amp_reshaped = amp.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, original_length)
        amp_interp = F.interpolate(amp_reshaped, size=max_pulse_samples, mode='linear', align_corners=True)
        amp_ = amp_interp.squeeze(0).squeeze(0).to(GLOBAL_DEVICE)

        # 2) Phasen-Interpolation
        import numpy as np
        ph_np = ph.detach().cpu().numpy()  # original_length
        x_original = np.linspace(0, original_length - 1, original_length)
        x_resampled = np.linspace(0, original_length - 1, max_pulse_samples)
        # unwrap und Interpolation
        ph_unwrapped = np.unwrap(ph_np)
        ph_interp = np.interp(x_resampled, x_original, ph_unwrapped)
        ph_wrapped = (ph_interp + np.pi) % (2 * np.pi) - np.pi
        ph_ = torch.tensor(ph_wrapped, dtype=torch.float64, device=GLOBAL_DEVICE)

        dtp_ = dtp * (original_length - 1) / max_pulse_samples
        rf_freq_out = 0.0  # freq already encoded in ph_ via w1_complex
    else:
        raise Exception("Unexpected case encountered in prep_rf_simulation_including_zeros.")

    return amp_.to(dtype=torch.float64), ph_.to(dtype=torch.float64), dtp_, delay_after_pulse, rf_freq_out


# def prep_rf_simulation(block: SimpleNamespace, max_pulse_samples: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
#     """
#     prep_rf_simulation Resamples the amplitude and phase of given RF event.

#     Parameters
#     ----------
#     block : SimpleNamespace
#         PyPulseq block event
#     max_pulse_samples : int
#         Maximum number of samples for the RF pulse.

#     Returns
#     -------
#     Tuple[torch.Tensor, torch.Tensor, float, float]
#         Tuple of resampled amplitude, phase, time step, and delay after pulse.

#     Raises
#     ------
#     Exception
#         If number of unique samples is larger than 1 but smaller than max_pulse_samples (not implemented yet).

#     """
#     amp = torch.tensor(block.rf.signal, dtype=torch.complex64, device=GLOBAL_DEVICE).abs()
#     ph = torch.angle(torch.tensor(block.rf.signal, dtype=torch.complex64, device=GLOBAL_DEVICE))
#     idx = torch.nonzero(amp > 1e-6, as_tuple=False).squeeze()

#     try:
#         rf_length = amp.size(0)
#         dtp = float(block.rf.t[1] - block.rf.t[0])
#         delay_after_pulse = (rf_length - idx.size(0)) * dtp
#     except AttributeError:
#         rf_length = amp.size(0)
#         dtp = 1e-6
#         delay_after_pulse = (rf_length - idx.size(0)) * dtp

#     amp = amp[idx]
#     ph = ph[idx]
#     n_unique = max(len(torch.unique(amp)), len(torch.unique(ph)))

#     # block pulse for seq-files >= 1.4.0
#     if n_unique == 1 and amp.size(0) == 2:
#         amp_ = amp[0]
#         ph_ = ph[0]
#         dtp_ = dtp
#     # block pulse for seq-files < 1.4.0
#     elif n_unique == 1:
#         amp_ = amp[0]
#         ph_ = ph[0]
#         dtp_ = dtp * amp.size(0)
#     # shaped pulse
#     elif n_unique > max_pulse_samples:
#         sample_factor = int(torch.ceil(torch.tensor(amp.size(0) / max_pulse_samples, device=GLOBAL_DEVICE)))
#         amp_ = amp[::sample_factor]
#         ph_ = ph[::sample_factor]
#         dtp_ = dtp * sample_factor
#     elif 1 < n_unique < max_pulse_samples:
#         # Speichere die ursprüngliche Länge des RF-Signals
#         original_length = amp.size(0)
        
#         # Amplitudeninterpolation mittels F.interpolate:
#         amp_reshaped = amp.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, original_length)
#         amp_interp = F.interpolate(amp_reshaped, size=max_pulse_samples, mode='linear', align_corners=True)
#         amp_ = amp_interp.squeeze(0).squeeze(0)         # Shape: (max_pulse_samples,)
#         amp_ = amp_.to(GLOBAL_DEVICE)
        
#         # Phaseninterpolation:
#         # Konvertiere die Phase in ein NumPy-Array, um dort np.unwrap und np.interp anzuwenden.
#         ph_np = ph.detach().cpu().numpy()               # (original_length,)
#         # Erstelle die x-Achsen-Vektoren für die Original- und Zielindizes
#         x_original = np.linspace(0, original_length - 1, original_length)
#         x_resampled = np.linspace(0, original_length - 1, max_pulse_samples)
#         # Unwrappe die Phase, um Sprünge zu vermeiden
#         ph_unwrapped = np.unwrap(ph_np)
#         # Interpoliere die unwrapped Phase auf die gewünschte Anzahl von Samples
#         ph_interp = np.interp(x_resampled, x_original, ph_unwrapped)
#         # Bringe die interpolierte Phase wieder in den Bereich [-π, π]
#         ph_wrapped = (ph_interp + np.pi) % (2 * np.pi) - np.pi
#         ph_ = torch.tensor(ph_wrapped, dtype=torch.float64, device=GLOBAL_DEVICE)
        
#         # Passe den effektiven Zeitschritt an:
#         dtp_ = dtp * (original_length / max_pulse_samples)
        
#     else:
#         raise Exception("Unexpected case encountered in prep_grad_simulation.")

#     return amp_.to(dtype=torch.float64), ph_.to(dtype=torch.float64), dtp_, delay_after_pulse


#!/usr/bin/env python
# filepath: /Users/danielmiksch/JupyterLab/optim/bmc/bmc_tool.py
# 

def prep_grad_simulation(block: SimpleNamespace, max_pulse_samples: int, dtp_rf: float = None) -> Tuple[torch.Tensor, float, float]:
    """
    Alternative zu prep_grad_simulation. Diese Version behält alle Werte bei,
    inklusive solcher kleiner/gleich 1e-6. Lediglich für die Berechnung von
    delay_after_grad wird geschaut, wie viele "inaktive" Samples (<= 1e-6) am
    Ende liegen. Ansonsten bleibt das Verhalten für die Erkennung von block vs.
    shaped Gradienten identisch.

    Parameters
    ----------
    block : SimpleNamespace
        PyPulseq block event mit block.gz.waveform und block.gz.tt.
    max_pulse_samples : int
        Maximale Anzahl an Samples für das Gradientensignal.
    dtp_rf : float, optional
        Falls angegeben, wird das Gradientensignal so resampled, dass
        der effektive Zeitschritt dtp_ exakt diesem Wert entspricht.

    Returns
    -------
    Tuple[torch.Tensor, float, float]
        (amp_, dtp_, delay_after_grad)
          - amp_: reshaped/resampled Amplitude (mit Nullen)
          - dtp_: Effektiver Zeitschritt
          - delay_after_grad: Zeit nach dem letzten aktiven Sample
    """
    import torch
    import torch.nn.functional as F
    from bmc.utils.global_device import GLOBAL_DEVICE

    # Ursprüngliches Signal als float64 laden
    amp_full = torch.tensor(block.gz.waveform, dtype=torch.float64, device=GLOBAL_DEVICE)

    # Index der aktiven Samples für delay-Berechnung
    idx_active = torch.nonzero(amp_full > 1e-6, as_tuple=False).squeeze()

    # Versuch, dtp und delay aus den Zeitvektoren zu berechnen
    try:
        grad_length = amp_full.size(0)
        dtp = float(block.gz.tt[1] - block.gz.tt[0])
        # Falls idx_active leer ist (kein >1e-6-Sample), delay = gesamte Länge
        if idx_active.numel() > 0:
            last_active_index = int(idx_active[-1])
            delay_after_grad = (grad_length - last_active_index - 1) * dtp
        else:
            delay_after_grad = grad_length * dtp
    except AttributeError:
        # Fallback, wenn block.gz.tt nicht verfügbar ist
        grad_length = amp_full.size(0)
        dtp = 1e-6
        if idx_active.numel() > 0:
            last_active_index = int(idx_active[-1])
            delay_after_grad = (grad_length - last_active_index - 1) * dtp
        else:
            delay_after_grad = grad_length * dtp

    # amp_ ist jetzt das vollständige Signal, inkl. Nullen
    amp = amp_full

    # Anzahl unterschiedlicher Werte
    n_unique = torch.unique(amp).size(0)

    # Analoge Logik wie in der Originalfunktion, um zwischen block pulse und shaped pulse zu unterscheiden
    if n_unique == 1 and amp.size(0) == 2:
        # block pulse für seq-files >= 1.4.0
        amp_ = amp[0].unsqueeze(0)
        dtp_ = dtp
    elif n_unique == 1:
        # block pulse für seq-files < 1.4.0
        amp_ = amp[0].unsqueeze(0)
        dtp_ = dtp * amp.size(0)
    elif n_unique > max_pulse_samples:
        # Downsampling (shaped pulse)
        sample_factor = int(torch.ceil(torch.tensor(amp.size(0) / max_pulse_samples, device=GLOBAL_DEVICE)))
        amp_ = amp[::sample_factor]
        dtp_ = dtp * sample_factor
    elif 1 < n_unique < max_pulse_samples:
        # Interpolation auf max_pulse_samples (shaped pulse)
        original_length = amp.size(0)
        amp_unsq = amp.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, original_length)
        amp_interp = F.interpolate(amp_unsq, size=max_pulse_samples, mode='linear', align_corners=True)
        amp_ = amp_interp.squeeze(0).squeeze(0)
        amp_ = amp_.to(GLOBAL_DEVICE)
        dtp_ = dtp * (original_length - 1) / max_pulse_samples
    else:
        raise Exception("Unexpected case encountered in prep_grad_simulation_including_zeros.")

    # Falls zusätzlich ein gemeinsamer dtp_rf angegeben ist, resample das Signal weiter
    if dtp_rf is not None:
        total_duration = grad_length * dtp
        new_samples = int(round(total_duration / dtp_rf))
        if new_samples < 1:
            new_samples = 1
        # Falls amp_ ein Skalar ist, in einen Vektor umwandeln
        if amp_.ndim == 0:
            amp_ = amp_.repeat(new_samples)
        else:
            amp_unsq2 = amp_.unsqueeze(0).unsqueeze(0)
            amp_resampled2 = F.interpolate(amp_unsq2, size=new_samples, mode='linear', align_corners=True)
            amp_ = amp_resampled2.squeeze(0).squeeze(0)
        dtp_ = dtp_rf

    return amp_, dtp_, delay_after_grad

class BMCTool:
    """
    Definition of the BMCTool class.
    """

    def __init__(self, params: Params, seq_file: Union[str, Path], verbose: bool = True, **kwargs) -> None:
        """
        __init__ Initialize BMCTool object.

        Parameters
        ----------
        params : Params
            Params object containing all simulation parameters
        seq_file : Union[str, Path]
            Path to the seq-file
        verbose : bool, optional
            Flag to activate detailed outpus, by default True
        """
        self.params = params
        self.seq_file = seq_file
        self.verbose = verbose
        self.run_m0_scan = None
        self.bm_solver = None

        self.seq = pp.Sequence()
        self.seq.read(seq_file)

        try:
            self.defs = self.seq.definitions
        except AttributeError:
            self.defs = self.seq.dict_definitions

        self.offsets_ppm = np.array(self.defs["offsets_ppm"])
        self.n_offsets = self.offsets_ppm.size

        if "num_meas" in self.defs:
            self.n_measure = int(self.defs["num_meas"])
        else:
            self.n_measure = self.n_offsets
            if "run_m0_scan" in self.defs:
                if 1 in self.defs["run_m0_scan"] or "True" in self.defs["run_m0_scan"]:
                    self.n_measure += 1

        self.m_init = params.m_vec.copy()
        self.m_out = np.zeros([self.m_init.shape[0], self.n_measure])

        self.bm_solver = BlochMcConnellSolver(
            params=self.params,
            n_offsets=1,
            z_positions=torch.tensor([0.0], dtype=torch.float64, device=GLOBAL_DEVICE),
        )

    def update_params(self, params: Params) -> None:
        """
        Update Params and BlochMcConnellSolver.
        """
        self.params = params
        self.bm_solver.update_params(params)

    def run(self) -> None:
        """
        Start simulation process.
        """
        if self.n_offsets != self.n_measure:
            self.run_m0_scan = True

        current_adc = 0
        accum_phase = 0
        mag = torch.tensor(
            self.m_init[np.newaxis, np.newaxis, :, np.newaxis],
            dtype=torch.float64, device=GLOBAL_DEVICE,
        )  # shape [1, 1, size, 1]

        try:
            block_events = self.seq.block_events
        except AttributeError:
            block_events = self.seq.dict_block_events

        if self.verbose:
            loop_block_events = tqdm(range(1, len(block_events) + 1), desc="BMCTool simulation")
        else:
            loop_block_events = range(1, len(block_events) + 1)

        # code for pypulseq >= 1.4.0:
        try:
            for block_event in loop_block_events:
                block = self.seq.get_block(block_event)
                current_adc, accum_phase, mag = self.run_1_4_0(block, current_adc, accum_phase, mag)
        except AttributeError:
            for block_event in loop_block_events:
                block = self.seq.get_block(block_event)
                current_adc, accum_phase, mag = self.run_1_3_0(block, current_adc, accum_phase, mag)

    def run_1_4_0(self, block, current_adc, accum_phase, mag) -> Tuple[int, float, torch.Tensor]:
        # pseudo ADC event
        if block.adc is not None:
            self.m_out[:, current_adc] = mag.squeeze().cpu().numpy()
            accum_phase = 0
            current_adc += 1
            if current_adc <= self.n_offsets and self.params.options["reset_init_mag"]:
                mag = torch.tensor(
                    self.m_init[np.newaxis, np.newaxis, :, np.newaxis],
                    dtype=torch.float64, device=GLOBAL_DEVICE,
                )

        # RF pulse
        elif block.rf is not None:
            amp_, ph_, dtp_, delay_after_pulse, _ = prep_rf_simulation(block, self.params.options["max_pulse_samples"])
            for i in range(amp_.numel()):
                self.bm_solver.update_matrix(
                    rf_amp=amp_[i],
                    rf_phase=-ph_[i] + block.rf.phase_offset - accum_phase,
                    rf_freq=block.rf.freq_offset,
                )
                mag = self.bm_solver.solve_equation(mag=mag, dtp=dtp_)

            if delay_after_pulse > 0:
                self.bm_solver.update_matrix(0, 0, 0)
                mag = self.bm_solver.solve_equation(mag=mag, dtp=delay_after_pulse)

            phase_degree = dtp_ * amp_.numel() * 360 * block.rf.freq_offset
            phase_degree %= 360
            accum_phase += phase_degree / 180 * np.pi

        # spoiler gradient in z-direction
        elif block.gz is not None:
            dur_ = block.block_duration
            self.bm_solver.update_matrix(0, 0, 0)
            mag = self.bm_solver.solve_equation(mag=mag, dtp=dur_)
            for j in range((len(self.params.cest_pools) + 1) * 2):
                mag[0, 0, j, 0] = 0.0  # assume complete spoiling

        # delay or gradient(s) in x and/or y-direction
        elif hasattr(block, "block_duration") and block.block_duration != "0":
            delay = block.block_duration
            self.bm_solver.update_matrix(0, 0, 0)
            mag = self.bm_solver.solve_equation(mag=mag, dtp=delay)

        else:
            raise Exception("Unknown case")

        return current_adc, accum_phase, mag

    def run_1_3_0(self, block, current_adc, accum_phase, mag) -> Tuple[int, float, torch.Tensor]:
        # pseudo ADC event
        if hasattr(block, "adc"):
            self.m_out[:, current_adc] = mag.squeeze().cpu().numpy()
            accum_phase = 0
            current_adc += 1
            if current_adc <= self.n_offsets and self.params.options["reset_init_mag"]:
                mag = torch.tensor(
                    self.m_init[np.newaxis, np.newaxis, :, np.newaxis],
                    dtype=torch.float64, device=GLOBAL_DEVICE,
                )

        # RF pulse
        elif hasattr(block, "rf"):
            amp_, ph_, dtp_, delay_after_pulse, _ = prep_rf_simulation(block, self.params.options["max_pulse_samples"])
            for i in range(amp_.numel()):
                self.bm_solver.update_matrix(
                    rf_amp=amp_[i],
                    rf_phase=-ph_[i] + block.rf.phase_offset - accum_phase,
                    rf_freq=block.rf.freq_offset,
                )
                mag = self.bm_solver.solve_equation(mag=mag, dtp=dtp_)

            if delay_after_pulse > 0:
                self.bm_solver.update_matrix(0, 0, 0)
                mag = self.bm_solver.solve_equation(mag=mag, dtp=delay_after_pulse)

            phase_degree = dtp_ * amp_.numel() * 360 * block.rf.freq_offset
            phase_degree %= 360
            accum_phase += phase_degree / 180 * np.pi

        # spoiler gradient in z-direction
        elif hasattr(block, "gz"):
            dur_ = float(block.gz.rise_time + block.gz.flat_time + block.gz.fall_time)
            self.bm_solver.update_matrix(0, 0, 0)
            mag = self.bm_solver.solve_equation(mag=mag, dtp=dur_)
            for j in range((len(self.params.cest_pools) + 1) * 2):
                mag[0, 0, j, 0] = 0.0  # assume complete spoiling

        # gradient in x and/or y-direction (handled as delay)
        elif hasattr(block, "gx") or hasattr(block, "gy"):
            if hasattr(block, "gx"):
                delay = float(block.gx.rise_time + block.gx.flat_time + block.gx.fall_time)
            elif hasattr(block, "gy"):
                delay = float(block.gy.rise_time + block.gy.flat_time + block.gy.fall_time)
            self.bm_solver.update_matrix(0, 0, 0)
            mag = self.bm_solver.solve_equation(mag=mag, dtp=delay)

        # delay
        elif hasattr(block, "delay") and hasattr(block.delay, "delay"):
            delay = float(block.delay.delay)
            self.bm_solver.update_matrix(0, 0, 0)
            mag = self.bm_solver.solve_equation(mag=mag, dtp=delay)

        else:
            raise Exception("Unknown case")

        return current_adc, accum_phase, mag

    def get_zspec(self, return_abs: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        get_zspec Calculate/extract the Z-spectrum.

        Parameters
        ----------
        return_abs : bool, optional
            flag to activate the return of absolute values, by default True

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of offsets and Z-spectrum
        """
        if self.run_m0_scan:
            m_0 = self.m_out[self.params.mz_loc, 0]
            m_ = self.m_out[self.params.mz_loc, 1:]
            m_z = m_ / m_0
        else:
            m_z = self.m_out[self.params.mz_loc, :]

        if self.offsets_ppm.size != m_z.size:
            self.offsets_ppm = np.arange(0, m_z.size)

        if return_abs:
            m_z = np.abs(m_z)
        else:
            m_z = np.array(m_z)

        return self.offsets_ppm, m_z
