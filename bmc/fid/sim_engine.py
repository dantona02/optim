from pathlib import Path
import numpy as np
from tqdm import tqdm
import subprocess
import json

from typing import Tuple, Union
from bmc.bmc_solver import BlochMcConnellSolver
from bmc.bmc_tool import BMCTool
from bmc.bmc_tool import prep_rf_simulation, prep_grad_simulation
from bmc.params import Params
from IPython import get_ipython
from manim import *
from manim.opengl import *


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
            self.defs["num_meas"] = self.params.options["max_pulse_samples"] * len(self.seq.block_events)
        else:
            self.defs["num_meas"] = self.params.options["max_pulse_samples"]

        if "num_meas" in self.defs:
            self.n_measure = int(self.defs["num_meas"]) #redefining n_measure to max_pulse_samples
        else:
            self.n_measure = self.n_offsets

        
        

        print(self.n_isochromats)

        self.m_out = np.zeros([self.n_isochromats, self.m_init.shape[0], self.n_measure]) #expanding m_out to the number of max_pulse_samples
        self.dt_adc = self.adc_time / self.params.options["max_pulse_samples"]

        self.t = np.array([])

        


    def run_adc(self, block, current_adc, accum_phase, mag) -> Tuple[int, float, np.ndarray]:
        #adc with time dt and max_pulse_sampels
        if block.adc is not None:
            
            start_time = self.t[-1] if self.t.size > 0 else 0
            print(f'adc at {start_time:.4f}s')
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
                print(f'rf at {start_time:.4f}s')
                time_array = start_time + np.arange(self.params.options["max_pulse_samples"]) * dtp_
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
                print(f'gz at {start_time:.4f}s')
                time_array = start_time + np.arange(self.params.options["max_pulse_samples"]) * dtp_
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

        # code for pypulseq >= 1.4.0:
        try:
            for block_event in loop_block_events:
                block = self.seq.get_block(block_event)
                current_adc, accum_phase, mag = self.run_adc(block, current_adc, accum_phase, mag)
        except AttributeError:
            for block_event in loop_block_events:
                block = self.seq.get_block(block_event)
                current_adc, accum_phase, mag = self.run_1_3_0(block, current_adc, accum_phase, mag)

    def get_mag(self, return_zmag: bool = False, return_cest_pool: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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


        if return_cest_pool and self.params.cest_pools:
            if return_zmag:
                m_z = self.m_out[:, self.params.mz_loc + 1, :]
                return self.t, np.abs(m_z)
            else:
                n_total_pools = len(self.params.cest_pools) + 1
                m_trans_c = self.m_out[:, 1, :] + 1j * self.m_out[:, n_total_pools + 1, :]
                return self.t, m_trans_c
        else:
            if return_zmag:
                m_z = self.m_out[:, self.params.mz_loc, :]
                return self.t, m_z
            elif self.params.cest_pools:
                n_total_pools = len(self.params.cest_pools) + 1
                m_trans_c = self.m_out[:, 0, :] + 1j * self.m_out[:, n_total_pools, :]
                return self.t, m_trans_c
            else:
                m_trans_c = self.m_out[:, 0, :] + 1j * self.m_out[:, 1, :]
                return self.t, m_trans_c
    
    def animate(self, step: int = 1, run_time = 0.1, track_path=False, ie=False, timing=False, **addParams) -> None:
        """
        Animates the magnetization vector in a 3D plot.
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
            Flag to activate interactive embedding, by default False. Jupyter kernel must be restarted after using this flag.
        """

        time = self.t

        if self.params.cest_pools:
            n_total_pools = len(self.params.cest_pools) + 1
            m_vec_water = np.stack(
                (self.m_out[0, :],
                 self.m_out[n_total_pools, :],
                 self.m_out[self.params.mz_loc, :]),
                 axis=1) #stack x,y,z magnetization to [[x1,y1,z1],[x2,y2,z2],...
            m_vec_cest = np.stack(
                (self.m_out[1, :],
                 self.m_out[n_total_pools + 1, :],
                 self.m_out[self.params.mz_loc + 1, :]),
                 axis=1)

        else:
            m_vec_water = np.stack(
                (self.m_out[0, :],
                 self.m_out[1, :],
                 self.m_out[self.params.mz_loc, :]),
                 axis=1)
        
        m_vec_water = m_vec_water[::step]#.tolist()
        mag_vector = m_vec_water
        render_params = {
            'quality': '-ql',
            'write': '',
        }
        render_params.update(addParams)
        
        class Vector3DScene(ThreeDScene):
            def construct(self):

                if mag_vector is None:
                    raise ValueError("No magnetization data available.")

                axes = ThreeDAxes(
                x_range=(-1.4, 1.4, .2),
                y_range=(-1.4, 1.4, .2),
                z_range=(-1.4, 1.4, .2)
                )

                decimal = Text("0", font_size=36)

                time_tracker = ValueTracker(0)

                def update_decimal(d):
                    # Finden Sie den aktuellen Index basierend auf der Animation
                    current_index = int(time_tracker.get_value())
                    # Verwenden Sie den Wert aus dem Array
                    current_time = time[current_index] if current_index < len(time) else time[-1]
                    
                    d.become(Text(f"t = {current_time:.4f} s", font_size=36))
                    d.fix_in_frame()
                    d.to_corner(UR).scale(0.7)

                # Text als OpenGL-kompatibles Objekt initialisieren
                if timing:
                    self.add(decimal)
                decimal.fix_in_frame()
                decimal.add_updater(update_decimal)

                scale_factor_xy = axes.x_length / (axes.x_range[1] - axes.x_range[0])
                scale_factor_z = axes.z_length / (axes.z_range[1] - axes.z_range[0])
                scaling_array = np.array([scale_factor_xy, scale_factor_xy, scale_factor_z])

                labels = axes.get_axis_labels(Text("x").scale(.7), Text("y").scale(.7), Text("z").scale(.7))

                #vector initialization
                vector = Vector(
                    mag_vector[0] * scaling_array,
                    color=RED
                )
                
                x_tracker = ValueTracker(mag_vector[0][0] * scaling_array[0])
                y_tracker = ValueTracker(mag_vector[0][1] * scaling_array[1])
                z_tracker = ValueTracker(mag_vector[0][2] * scaling_array[2])

                def update_vector(v):
                    new_vector = Vector([x_tracker.get_value(), 
                                        y_tracker.get_value(), 
                                        z_tracker.get_value()], 
                                        color=RED)
                    v.become(new_vector)

                vector.add_updater(update_vector)

                path = TracedPath(vector.get_end, stroke_color=RED, stroke_width=1)
                self.add(axes, labels, vector)
                if track_path:
                    self.add(path)

                self.set_camera_orientation(phi=65 * DEGREES, theta=135 * DEGREES)
                
                self.wait(0.5)

                for i, pos in enumerate(mag_vector[1:]):
                    self.play(
                        x_tracker.animate.set_value(pos[0] * scaling_array[0]),
                        y_tracker.animate.set_value(pos[1] * scaling_array[1]),
                        z_tracker.animate.set_value(pos[2] * scaling_array[2]),
                        time_tracker.animate.set_value(i+1),
                        run_time=run_time, rate_func=linear)
                
                if ie:
                    self.interactive_embed()
                
        ipython = get_ipython()
        if ipython:
            ipython.run_line_magic('manim', f'-v WARNING {render_params["quality"]} --disable_caching --renderer=opengl {render_params["write"]} Vector3DScene')
        else:
            print("Magic commands are not supported outside Jupyter notebooks.")
        
    def get_time(self) -> np.ndarray:
        """
        Returns the time array.
        """
        return self.t