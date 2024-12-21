from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import time

from datetime import timedelta
from typing import Tuple, Union
from bmc.bmc_solver import BlochMcConnellSolver
from bmc.bmc_tool import BMCTool
from bmc.bmc_tool import prep_rf_simulation, prep_grad_simulation
from bmc.params import Params
from IPython import get_ipython
from manim import *
from manim.opengl import *

from bmc.utils.webhook import DiscordNotifier


class BMCSim(BMCTool):
    def __init__(self, adc_time: np.float64, params: Params, seq_file: str | Path, z_positions: np.ndarray, verbose: bool = True, write_all_mag: bool = False, **kwargs) -> None:
        super().__init__(params, seq_file, verbose, **kwargs)
        """
        Parameters
        ----------
        adc_time: np.float64
            readout duration in seconds
        params : Params
            Params object containing all simulation parameters
        seq_file : Union[str, Path]
            Path to the seq-file
        verbose : bool, optional
            Flag to activate detailed outpus, by default True
        """
        self.z_positions = z_positions
        self.n_isochromats = len(self.z_positions)
        self.bm_solver = BlochMcConnellSolver(params=self.params, n_offsets=self.n_offsets, z_positions=self.z_positions)
        
        self.adc_time = adc_time
        self.write_all_mag = write_all_mag
        
        if self.write_all_mag:
            self.defs["num_meas"] = (self.params.options["max_pulse_samples"] + 1) * len(self.seq.block_events)
        else:
            self.defs["num_meas"] = (self.params.options["max_pulse_samples"] + 1)

        if "num_meas" in self.defs:
            self.n_measure = int(self.defs["num_meas"]) #redefining n_measure to max_pulse_samples
        else:
            self.n_measure = self.n_offsets

        self.m_out = np.zeros([self.n_isochromats, self.m_init.shape[0], self.n_measure]) #expanding m_out to the number of max_pulse_samples
        self.dt_adc = self.adc_time / self.params.options["max_pulse_samples"]

        self.t = np.array([])
        self.total_vec = None
        self.events = []

        


    def run_adc(self, block, current_adc, accum_phase, mag) -> Tuple[int, float, np.ndarray]:
        #adc with time dt and max_pulse_sampels
        if block.adc is not None:
            
            start_time = self.t[-1] if self.t.size > 0 else 0
            self.events.append(f'adc at {start_time:.4f}s')
            time_array = start_time + np.arange(self.params.options["max_pulse_samples"]) * self.dt_adc
            self.t = np.append(self.t, time_array)

            for step in range(self.params.options["max_pulse_samples"]):
                self.m_out[:, :, current_adc] = np.squeeze(mag)
                
                accum_phase = 0
                current_adc += 1

                self.bm_solver.update_matrix(0, 0, 0) #no rf_amp, no rf_phase, no rf_freq
                mag = self.bm_solver.solve_equation(mag=mag, dtp=self.dt_adc)
            
            # RF pulse
        elif block.rf is not None:
            amp_, ph_, dtp_, delay_after_pulse = prep_rf_simulation(block, self.params.options["max_pulse_samples"])

            if self.write_all_mag:
                start_time = self.t[-1] if self.t.size > 0 else 0
                self.events.append(f'rf at {start_time:.4f}s')
                time_array = start_time + np.arange(amp_.size) * dtp_
                self.t = np.append(self.t, time_array)

            for i in range(amp_.size):
                if self.write_all_mag: #might have a slight overhead, can be rewritten in to if else statement
                    self.m_out[:, :, current_adc] = np.squeeze(mag)
                    current_adc += 1

                self.bm_solver.update_matrix(
                    rf_amp=amp_[i],
                    rf_phase=-ph_[i] + block.rf.phase_offset - accum_phase,
                    rf_freq=block.rf.freq_offset,
                )

                mag = self.bm_solver.solve_equation(mag=mag, dtp=dtp_)
                

            if delay_after_pulse > 0: #might cause problems if delay_after_pulse is significantly larger than 0
                self.bm_solver.update_matrix(0, 0, 0)
                if self.write_all_mag:
                    start_time = self.t[-1] if self.t.size > 0 else 0
                    time_array = start_time + np.arange(1) * delay_after_pulse  # Ein Zeitschritt hinzufügen
                    self.t = np.append(self.t, time_array)
                    self.m_out[:, :, current_adc] = np.squeeze(mag)
                    current_adc += 1
                mag = self.bm_solver.solve_equation(mag=mag, dtp=delay_after_pulse)

            phase_degree = dtp_ * amp_.size * 360 * block.rf.freq_offset
            phase_degree %= 360
            accum_phase += phase_degree / 180 * np.pi
            
        elif block.gz is not None:
            amp_, dtp_, delay_after_grad = prep_grad_simulation(block, self.params.options["max_pulse_samples"])

            if self.write_all_mag:
                start_time = self.t[-1] if self.t.size > 0 else 0
                self.events.append(f'gz at {start_time:.4f}s')
                time_array = start_time + np.arange(amp_.size) * dtp_
                self.t = np.append(self.t, time_array)

            for i in range(amp_.size):
                
                if self.write_all_mag:
                    self.m_out[:, :, current_adc] = np.squeeze(mag)
                    current_adc += 1

                self.bm_solver.update_matrix(0, 0, 0, grad_amp=amp_[i])
                mag = self.bm_solver.solve_equation(mag=mag, dtp=dtp_)
                
            if delay_after_grad > 0:
                self.bm_solver.update_matrix(0, 0, 0)
                if self.write_all_mag:
                    start_time = self.t[-1] if self.t.size > 0 else 0
                    time_array = start_time + np.arange(1) * delay_after_grad  # Ein Zeitschritt hinzufügen
                    self.t = np.append(self.t, time_array)
                    self.m_out[:, :, current_adc] = np.squeeze(mag)
                    current_adc += 1
                mag = self.bm_solver.solve_equation(mag=mag, dtp=delay_after_grad)

        # elif block.gz is not None:
        #     dur_ = block.block_duration
        #     self.bm_solver.update_matrix(0, 0, 0)
        #     mag = self.bm_solver.solve_equation(mag=mag, dtp=dur_)
        #     for j in range((len(self.params.cest_pools) + 1) * 2):
        #         mag[0, j, 0] = 0.0  # assume complete spoiling

        return current_adc, accum_phase, mag

    def run_fid(self) -> None:
        """
        Start simulation process.
        """
        if self.n_offsets != self.n_measure:
            self.run_m0_scan = True

        current_adc = 0
        accum_phase = 0
        mag = self.m_init[np.newaxis, np.newaxis, :, np.newaxis] #extended to [n_isochromats, ...]

        try:
            block_events = self.seq.block_events
        except AttributeError:
            block_events = self.seq.dict_block_events

        if self.verbose:
            loop_block_events = tqdm(range(1, len(block_events) + 1), desc="BMCTool simulation")
        else:
            loop_block_events = range(1, len(block_events) + 1)

        webhook_url = "https://discord.com/api/webhooks/1319826840747245639/jeioX8DeZmynOv2Fatz5nB3F4-MMx3dCnMoM9Uit9c0yOtOuEvmlMpiE_MCDEXxlbkMg"
        notifier = DiscordNotifier(webhook_url)
        seq_filename = os.path.basename(self.seq_file) 
        n_cest_pools = len(self.params.cest_pools)
        total_blocks = len(block_events)

        initial_table = (
            "## __Simulation started__ 🚀\n"
            f"Sequence: _{seq_filename}_\n"
            f"N-CEST: {n_cest_pools}\n"
            f"N-Iso: {self.n_isochromats}\n"
            "```\n"
            "+----------------+--------------+------------+\n"
            "| Step           | Progress (%) | Status     |\n"
            "+----------------+--------------+------------+\n"
            f"| {0}/{total_blocks:<12} | {0:<12.2f} | {'Started':<10} |\n"
            "+----------------+--------------+------------+\n"
            "```"
        )
        notifier.send_initial_message(initial_table)
        
       
        # code for pypulseq >= 1.4.0:
        
        try:
            start_time = time.time()
            for i, block_event in enumerate(loop_block_events, start=1):
                block = self.seq.get_block(block_event)
                current_adc, accum_phase, mag = self.run_adc(block, current_adc, accum_phase, mag)

                # Calculate progress and update the status
                progress = (i / total_blocks) * 100
                status_message = (
                    
                    f"## __Simulation running__ 🔄\n"
                    f"Sequence: _{seq_filename}_\n"
                    f"N-CEST: {n_cest_pools}\n"
                    f"N-Iso: {self.n_isochromats}\n"
                    "```\n"
                    "+----------------+--------------+------------+\n"
                    "| Step           | Progress (%) | Status     |\n"
                    "+----------------+--------------+------------+\n"
                    f"| {i}/{total_blocks:<12} | {progress:<12.2f} | {'Running':<10} |\n"
                    "+----------------+--------------+------------+\n"
                    "```"
                )
                notifier.update_message(status_message)
        except AttributeError:
            for block_event in loop_block_events:
                block = self.seq.get_block(block_event)
                current_adc, accum_phase, mag = self.run_1_3_0(block, current_adc, accum_phase, mag)

        self.m_out = self.m_out[:, :, :self.t.size]
        print(self.events)

        end_time = time.time()
        elapsed_time = timedelta(seconds=end_time - start_time)

        notifier.update_message(
            f"## __Simulation completed__ ✅\n"
            f"Sequence: _{seq_filename}_\n"
            f"N-CEST: {n_cest_pools}\n"
            f"N-Iso: {self.n_isochromats}\n"
            "```\n"
            "+----------------+--------------+------------+\n"
            "| Step           | Progress (%) | Status     |\n"
            "+----------------+--------------+------------+\n"
            f"| {total_blocks}/{total_blocks:<12} | {100:<12.2f} | {'Completed':<10} |\n"
            "+----------------+--------------+------------+\n"
            f"Duration: {elapsed_time}\n"
            "```"
        )
    

    def get_mag(self, return_cest_pool: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Rturns the complex transverse magnetization of water pool or ONE cest pool. No implementation for MT pools.
        If return_cest_pool is True, only returns magnetization of the first cest pool.
        ----------
        Parameters
        ----------
        return_zmag: bool, optional
            If True, returns z-magnetization
        return_cest_pool: bool, optional
            If True, returns magnetization of the first cest pool
        """
        n_total_pools = len(self.params.cest_pools) + 1

        if return_cest_pool and self.params.cest_pools:

            m_x = self.m_out[:, n_total_pools + 1, :]
            m_y =self.m_out[:, 1, :]
            m_z = self.m_out[:, self.params.mz_loc + 1, :]

            
            m_x_total = np.sum(m_x, axis=0)
            m_y_total = np.sum(m_y, axis=0)
            m_z_total = np.sum(m_z, axis=0)

            norm_factor = np.max(np.sqrt(m_x_total**2 + m_y_total**2 + m_z_total**2))

            m_z_total /= norm_factor if norm_factor != 0 else 1
            
            m_x_total /= norm_factor if norm_factor != 0 else 1
            m_y_total /= norm_factor if norm_factor != 0 else 1

            m_trans_c = m_x + 1j * m_y
            m_trans_c_total = m_x_total + 1j * m_y_total

            self.total_vec = np.vstack((m_x_total, m_y_total, m_z_total)).T

            return self.t, np.abs(m_z), abs(m_z_total), m_trans_c, m_trans_c_total
        
        else:
            
            m_x = self.m_out[:, 0, :]
            m_y = self.m_out[:, n_total_pools, :]
            m_z = self.m_out[:, self.params.mz_loc, :]

            m_x_total = np.sum(m_x, axis=0)
            m_y_total = np.sum(m_y, axis=0)
            m_z_total = np.sum(m_z, axis=0)

            norm_factor = np.max(np.sqrt(m_x_total**2 + m_y_total**2 + m_z_total**2))

            m_z_total /= norm_factor if norm_factor != 0 else 1
            
            m_x_total /= norm_factor if norm_factor != 0 else 1
            m_y_total /= norm_factor if norm_factor != 0 else 1

            m_trans_c = m_x + 1j * m_y
            m_trans_c_total = m_x_total + 1j * m_y_total

            self.total_vec = np.vstack((m_x_total, m_y_total, m_z_total)).T

            return self.t, np.abs(m_z), abs(m_z_total), m_trans_c, m_trans_c_total
    


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

        time = self.t[::step]
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