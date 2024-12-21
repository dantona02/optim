from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from bmc.utils.global_device import GLOBAL_DEVICE

from typing import Tuple, Union
from bmc.bmc_solver import BlochMcConnellSolver
from bmc.bmc_tool import BMCTool
from bmc.bmc_tool import prep_rf_simulation, prep_grad_simulation
from bmc.params import Params
from IPython import get_ipython
from manim import *
from manim.opengl import *


class BMCSim(BMCTool):
    def __init__(self, adc_time: float, params: Params, seq_file: str | Path, z_positions: torch.Tensor, verbose: bool = True, write_all_mag: bool = False, **kwargs) -> None:
        super().__init__(params, seq_file, verbose, **kwargs)
        self.z_positions = z_positions.to(GLOBAL_DEVICE)  # Torch-Tensor
        self.n_isochromats = len(self.z_positions)
        self.bm_solver = BlochMcConnellSolver(params=self.params, n_offsets=self.n_offsets, z_positions=self.z_positions)
        
        self.adc_time = adc_time
        self.write_all_mag = write_all_mag

        if self.write_all_mag:
            self.defs["num_meas"] = (self.params.options["max_pulse_samples"] + 1) * len(self.seq.block_events)
        else:
            self.defs["num_meas"] = (self.params.options["max_pulse_samples"] + 1)

        self.n_measure = int(self.defs["num_meas"]) if "num_meas" in self.defs else self.n_offsets

        self.m_out = torch.zeros(self.n_isochromats, self.m_init.shape[0], self.n_measure, dtype=torch.float32, device=GLOBAL_DEVICE)
        self.dt_adc = self.adc_time / self.params.options["max_pulse_samples"]

        self.t = torch.tensor([], dtype=torch.float32, device=GLOBAL_DEVICE)
        self.total_vec = None
        self.events = []


    def run_adc(self, block, current_adc, accum_phase, mag) -> Tuple[int, float, torch.Tensor]:
        """
        Handles the simulation of ADC, RF, and gradient blocks.
        Updates the time array, magnetization output, and accumulated phase.

        Parameters
        ----------
        block : Block
            The current block in the sequence.
        current_adc : int
            The current ADC index.
        accum_phase : float
            The accumulated RF phase.
        mag : torch.Tensor
            The current magnetization vector.

        Returns
        -------
        current_adc : int
            Updated ADC index.
        accum_phase : float
            Updated accumulated phase.
        mag : torch.Tensor
            Updated magnetization vector.
        """
        if block.adc is not None:
            start_time = self.t[-1].item() if self.t.numel() > 0 else 0
            self.events.append(f'adc at {start_time:.4f}s')
            time_array = start_time + torch.arange(self.params.options["max_pulse_samples"], dtype=torch.float32, device=GLOBAL_DEVICE) * self.dt_adc
            self.t = torch.cat((self.t, time_array))

            for step in range(self.params.options["max_pulse_samples"]):
                self.m_out[:, :, current_adc] = mag.squeeze()
                accum_phase = 0
                current_adc += 1

                self.bm_solver.update_matrix(0, 0, 0)  # No RF amplitude, phase, or frequency
                mag = self.bm_solver.solve_equation(mag=mag, dtp=self.dt_adc)
            
            # RF pulse
        elif block.rf is not None:
            amp_, ph_, dtp_, delay_after_pulse = prep_rf_simulation(block, self.params.options["max_pulse_samples"])

            if self.write_all_mag:
                start_time = self.t[-1].item() if self.t.numel() > 0 else 0
                self.events.append(f'rf at {start_time:.4f}s')
                time_array = start_time + torch.arange(amp_.numel(), dtype=torch.float32, device=GLOBAL_DEVICE) * dtp_
                self.t = torch.cat((self.t, time_array))

            for i in range(amp_.numel()):
                if self.write_all_mag:
                    self.m_out[:, :, current_adc] = mag.squeeze()
                    current_adc += 1

                self.bm_solver.update_matrix(
                    rf_amp=amp_[i].item(),
                    rf_phase=-ph_[i].item() + block.rf.phase_offset - accum_phase,
                    rf_freq=block.rf.freq_offset,
                )
                mag = self.bm_solver.solve_equation(mag=mag, dtp=dtp_)
                

            if delay_after_pulse > 0:
                self.bm_solver.update_matrix(0, 0, 0)
                if self.write_all_mag:
                    start_time = self.t[-1].item() if self.t.numel() > 0 else 0
                    time_array = start_time + torch.arange(1, dtype=torch.float32, device=GLOBAL_DEVICE) * delay_after_pulse
                    self.t = torch.cat((self.t, time_array))
                    self.m_out[:, :, current_adc] = mag.squeeze()
                    current_adc += 1
                mag = self.bm_solver.solve_equation(mag=mag, dtp=delay_after_pulse)

            phase_degree = dtp_ * amp_.numel() * 360 * block.rf.freq_offset
            phase_degree %= 360
            accum_phase += phase_degree / 180 * torch.pi
            
        elif block.gz is not None:
            amp_, dtp_, delay_after_grad = prep_grad_simulation(block, self.params.options["max_pulse_samples"])

            if self.write_all_mag:
                start_time = self.t[-1].item() if self.t.numel() > 0 else 0
                self.events.append(f'gz at {start_time:.4f}s')
                time_array = start_time + torch.arange(amp_.numel(), dtype=torch.float32, device=GLOBAL_DEVICE) * dtp_
                self.t = torch.cat((self.t, time_array))

            for i in range(amp_.numel()):
                if self.write_all_mag:
                    self.m_out[:, :, current_adc] = mag.squeeze()
                    current_adc += 1

                self.bm_solver.update_matrix(0, 0, 0, grad_amp=amp_[i].item())
                mag = self.bm_solver.solve_equation(mag=mag, dtp=dtp_)

            if delay_after_grad > 0:
                self.bm_solver.update_matrix(0, 0, 0)
                if self.write_all_mag:
                    start_time = self.t[-1].item() if self.t.numel() > 0 else 0
                    time_array = start_time + torch.arange(1, dtype=torch.float32, device=GLOBAL_DEVICE) * delay_after_grad
                    self.t = torch.cat((self.t, time_array))
                    self.m_out[:, :, current_adc] = mag.squeeze()
                    current_adc += 1
                mag = self.bm_solver.solve_equation(mag=mag, dtp=delay_after_grad)

        return current_adc, accum_phase, mag

    def run_fid(self) -> None:
        """
        Start simulation process.
        """
        if self.n_offsets != self.n_measure:
            self.run_m0_scan = True

        current_adc = 0
        accum_phase = 0

        mag = torch.tensor(
            self.m_init[np.newaxis, np.newaxis, :, np.newaxis], 
            dtype=torch.float32,
            device=GLOBAL_DEVICE
        )

        try:
            block_events = self.seq.block_events
        except AttributeError:
            block_events = self.seq.dict_block_events

        if self.verbose:
            loop_block_events = tqdm(range(1, len(block_events) + 1), desc="BMCTool simulation")
        else:
            loop_block_events = range(1, len(block_events) + 1)

        try:
            for block_event in loop_block_events:
                block = self.seq.get_block(block_event)
                current_adc, accum_phase, mag = self.run_adc(block, current_adc, accum_phase, mag)
        except AttributeError:
            for block_event in loop_block_events:
                block = self.seq.get_block(block_event)
                current_adc, accum_phase, mag = self.run_1_3_0(block, current_adc, accum_phase, mag)

        # Trim die Zeiteinträge und passe die Form von m_out an
        self.m_out = self.m_out[:, :, :self.t.numel()]
        print(self.events)
    

    def get_mag(self, return_cest_pool: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the complex transverse magnetization of water pool or ONE CEST pool. No implementation for MT pools.
        If return_cest_pool is True, only returns magnetization of the first CEST pool.
        ----------
        Parameters
        ----------
        return_zmag: bool, optional
            If True, returns z-magnetization
        return_cest_pool: bool, optional
            If True, returns magnetization of the first CEST pool
        """
        n_total_pools = len(self.params.cest_pools) + 1

        if return_cest_pool and self.params.cest_pools:

            m_x = self.m_out[:, n_total_pools + 1, :]
            m_y = self.m_out[:, 1, :]
            m_z = self.m_out[:, self.params.mz_loc + 1, :]

            m_x_total = torch.sum(m_x, dim=0)
            m_y_total = torch.sum(m_y, dim=0)
            m_z_total = torch.sum(m_z, dim=0)

            norm_factor = torch.max(torch.sqrt(m_x_total**2 + m_y_total**2 + m_z_total**2))

            m_z_total /= norm_factor if norm_factor != 0 else 1
            m_x_total /= norm_factor if norm_factor != 0 else 1
            m_y_total /= norm_factor if norm_factor != 0 else 1

            m_trans_c = m_x + 1j * m_y
            m_trans_c_total = m_x_total + 1j * m_y_total

            self.total_vec = torch.stack((m_x_total, m_y_total, m_z_total), dim=-1)

            return self.t, torch.abs(m_z), torch.abs(m_z_total), m_trans_c, m_trans_c_total

        else:

            m_x = self.m_out[:, 0, :]
            m_y = self.m_out[:, n_total_pools, :]
            m_z = self.m_out[:, self.params.mz_loc, :]

            m_x_total = torch.sum(m_x, dim=0)
            m_y_total = torch.sum(m_y, dim=0)
            m_z_total = torch.sum(m_z, dim=0)

            norm_factor = torch.max(torch.sqrt(m_x_total**2 + m_y_total**2 + m_z_total**2))

            m_z_total /= norm_factor if norm_factor != 0 else 1
            m_x_total /= norm_factor if norm_factor != 0 else 1
            m_y_total /= norm_factor if norm_factor != 0 else 1

            m_trans_c = m_x + 1j * m_y
            m_trans_c_total = m_x_total + 1j * m_y_total

            self.total_vec = torch.stack((m_x_total, m_y_total, m_z_total), dim=-1)

            return self.t, torch.abs(m_z), torch.abs(m_z_total), m_trans_c, m_trans_c_total
    


    def get_time(self) -> np.ndarray:
        """
        Returns the time array.
        """
        return self.t
    

    def animate(self, step: int = 1, run_time=0.1, track_path=False, ie=False, timing=False, total_mag: bool = False, **addParams) -> None:
        """
        Animates the magnetization vector for all isochromats in a 3D plot.
        ----------
        Parameters
        ----------
        step: int, optional
            Step size for animation, by default 1
        run_time: float, optional
            Duration of each animation step, by default 0.1
        track_path: bool, optional
            Flag to activate path tracking, by default False
        ie: bool, optional
            Flag to activate interactive embedding, by default False
        timing: bool, optional
            Adds the current simulation time to the corner of the animation.
        """

        time = self.t[::step].cpu()
        self.m_out = self.m_out.cpu()
        self.total_vec = self.total_vec.cpu()
        self.z_positions = self.z_positions.cpu()
        isochromats = self.n_isochromats

        # Magnetisierung für Wasserpools vorbereiten
        if self.params.cest_pools:
            n_total_pools = len(self.params.cest_pools) + 1
            m_vec_water = np.stack(
                (self.m_out[:, 0, :],
                 self.m_out[:, n_total_pools, :],
                 self.m_out[:, self.params.mz_loc, :]),
                 axis=2)
        else:
            m_vec_water = np.stack(
                (self.m_out[:, 0, :],
                 self.m_out[:, 1, :],
                 self.m_out[:, self.params.mz_loc, :]),
                 axis=2)
        m_vec_total = np.stack(
            (self.total_vec[:, 0],
            self.total_vec[:, 1],
            self.total_vec[:, 2]),
            axis=1
        ) if total_mag else None

        
        
        m_vec_water = m_vec_water[:, ::step]
        m_vec_total = m_vec_total[::step]  # Schrittweite anwenden
        middle_idx = np.where(self.z_positions == 0)[0][0]
        m_vec_middle = m_vec_water[middle_idx] if total_mag else None
        render_params = {
            'quality': '-ql',
            'write': '',
        }
        render_params.update(addParams)

        class Vector3DScene(ThreeDScene):
            def construct(self):
                # Achsen und Skalierungsfaktoren
                axes = ThreeDAxes(
                    x_range=(-1.4, 1.4, .2),
                    y_range=(-1.4, 1.4, .2),
                    z_range=(-1.4, 1.4, .2)
                )
                scale_factor_xy = axes.x_length / (axes.x_range[1] - axes.x_range[0])
                scale_factor_z = axes.z_length / (axes.z_range[1] - axes.z_range[0])
                scaling_array = np.array([scale_factor_xy, scale_factor_xy, scale_factor_z])
                labels = axes.get_axis_labels(Text("x").scale(.7), Text("y").scale(.7), Text("z").scale(.7))

                # Farbübergang für Isochromaten
                colors = color_gradient([PURE_BLUE, PURE_GREEN, PURE_RED], isochromats)

                # Tracker und Vektoren für alle Isochromaten initialisieren
                trackers = []
                vectors = []
                paths = []

                if total_mag:
                    for i, vec in enumerate([m_vec_total, m_vec_middle]):
                        colors_total_mag = [WHITE, PURE_GREEN]
                        x_tracker = ValueTracker(vec[0, 0] * scaling_array[0])
                        y_tracker = ValueTracker(vec[0, 1] * scaling_array[1])
                        z_tracker = ValueTracker(vec[0, 2] * scaling_array[2])
                        trackers.append((x_tracker, y_tracker, z_tracker))

                        # Vektor erstellen
                        vector = Vector(
                            [x_tracker.get_value(), y_tracker.get_value(), z_tracker.get_value()],
                            color=colors_total_mag[i]
                        )

                        def update_vector(v, x=x_tracker, y=y_tracker, z=z_tracker, col=colors_total_mag[i]):
                            v.become(Vector([x.get_value(), y.get_value(), z.get_value()], color=col))

                        vector.add_updater(update_vector)
                        vectors.append(vector)

                        # Pfad hinzufügen, falls aktiviert
                        if track_path:
                            path = TracedPath(vector.get_end, stroke_color=colors_total_mag[i], stroke_width=1)
                            paths.append(path)

                    self.add(axes, labels, *vectors)
                    if track_path:
                        self.add(*paths)

                    self.set_camera_orientation(phi=65 * DEGREES, theta=135 * DEGREES)

                    # Text für die Zeit
                    if timing:
                        decimal = Text("0", font_size=36)
                        time_tracker = ValueTracker(0)

                        def update_decimal(d):
                            current_index = int(time_tracker.get_value())
                            current_time = time[current_index] if current_index < len(time) else time[-1]
                            d.become(Text(f"t = {current_time:.4f} s", font_size=36))
                            d.fix_in_frame()
                            d.to_corner(UR).scale(0.7)

                        decimal.add_updater(update_decimal)
                        self.add(decimal)

                    # Animation über die Zeit
                    for t in range(1, len(time)):
                        self.play(
                            *[
                                trackers[i][0].animate.set_value(vec[t, 0] * scaling_array[0]) for i, vec in enumerate([m_vec_total, m_vec_middle])
                            ] + [
                                trackers[i][1].animate.set_value(vec[t, 1] * scaling_array[1]) for i, vec in enumerate([m_vec_total, m_vec_middle])
                            ] + [
                                trackers[i][2].animate.set_value(vec[t, 2] * scaling_array[2]) for i, vec in enumerate([m_vec_total, m_vec_middle])
                            ] + (
                                [time_tracker.animate.set_value(t)] if timing else []
                            ),
                            run_time=run_time, rate_func=linear
                        )
                else:

                    for i in range(isochromats):
                        x_tracker = ValueTracker(m_vec_water[i, 0, 0] * scaling_array[0])
                        y_tracker = ValueTracker(m_vec_water[i, 0, 1] * scaling_array[1])
                        z_tracker = ValueTracker(m_vec_water[i, 0, 2] * scaling_array[2])
                        trackers.append((x_tracker, y_tracker, z_tracker))

                        # Vektor erstellen
                        vector = Vector(
                            [x_tracker.get_value(), y_tracker.get_value(), z_tracker.get_value()],
                            color=colors[i]
                        )

                        def update_vector(v, x=x_tracker, y=y_tracker, z=z_tracker, col=colors[i]):
                            v.become(Vector([x.get_value(), y.get_value(), z.get_value()], color=col))

                        vector.add_updater(update_vector)
                        vectors.append(vector)

                        # Pfad hinzufügen, falls aktiviert
                        if track_path:
                            path = TracedPath(vector.get_end, stroke_color=colors[i], stroke_width=1)
                            paths.append(path)

                    self.add(axes, labels, *vectors)
                    if track_path:
                        self.add(*paths)

                    self.set_camera_orientation(phi=65 * DEGREES, theta=135 * DEGREES)

                    # Text für die Zeit
                    if timing:
                        decimal = Text("0", font_size=36)
                        time_tracker = ValueTracker(0)

                        def update_decimal(d):
                            current_index = int(time_tracker.get_value())
                            current_time = time[current_index] if current_index < len(time) else time[-1]
                            d.become(Text(f"t = {current_time:.4f} s", font_size=36))
                            d.fix_in_frame()
                            d.to_corner(UR).scale(0.7)

                        decimal.add_updater(update_decimal)
                        self.add(decimal)

                    # Animation über die Zeit
                    for t in range(1, len(time)):
                        self.play(
                            *[
                                trackers[i][0].animate.set_value(m_vec_water[i, t, 0] * scaling_array[0]) for i in range(isochromats)
                            ] + [
                                trackers[i][1].animate.set_value(m_vec_water[i, t, 1] * scaling_array[1]) for i in range(isochromats)
                            ] + [
                                trackers[i][2].animate.set_value(m_vec_water[i, t, 2] * scaling_array[2]) for i in range(isochromats)
                            ] + (
                                [time_tracker.animate.set_value(t)] if timing else []
                            ),
                            run_time=run_time, rate_func=linear
                        )

                if ie:
                    self.interactive_embed()

        ipython = get_ipython()
        if ipython:
            ipython.run_line_magic('manim', f'-v WARNING {render_params["quality"]} --disable_caching --renderer=opengl {render_params["write"]} Vector3DScene')
        else:
            print("Magic commands are not supported outside Jupyter notebooks.")   