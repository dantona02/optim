from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import time
from bmc.utils.global_device import GLOBAL_DEVICE
from datetime import timedelta

from typing import Tuple, Union, Callable, Optional
# from bmc.bmc_solver import BlochMcConnellSolver
from bmc.solver import BlochMcConnellSolver
from bmc.bmc_tool import BMCTool
from bmc.bmc_tool import prep_rf_simulation, prep_grad_simulation
from bmc.params import Params
from IPython import get_ipython
from manim import *
from manim.opengl import *

from bmc.utils.webhook import DiscordNotifier

# torch.backends.cuda.preferred_linalg_library('magma')

class BMCSim(BMCTool):
    def __init__(self, adc_time: float, params: Params, seq_file: str | Path, z_positions: torch.Tensor, n_backlog: str | int, verbose: bool = True, webhook: bool = False, progress_callback: Optional[Callable[[int, int], None]] = None, **kwargs) -> None:
        super().__init__(params, seq_file, verbose, **kwargs)
        self.z_positions = z_positions.to(GLOBAL_DEVICE)  # Torch-Tensor
        self.n_isochromats = len(self.z_positions)
        self.bm_solver = BlochMcConnellSolver(params=self.params, n_offsets=self.n_offsets, z_positions=self.z_positions)
        
        self.adc_time = adc_time
        self.n_backlog = n_backlog

        if self.n_backlog == "ALL":
            self.n_backlog = len(self.seq.block_events)
            self.defs["num_meas"] = (self.params.options["max_pulse_samples"] + 1) * len(self.seq.block_events)
        else:
            self.defs["num_meas"] = (self.params.options["max_pulse_samples"] + 2) * (self.n_backlog + 1)

        if "num_meas" in self.defs:
            self.n_measure = int(self.defs["num_meas"]) #redefining n_measure to max_pulse_samples
        else:
            self.n_measure = self.n_offsets

        self.m_out = torch.zeros(self.n_isochromats, self.m_init.shape[0], self.n_measure, dtype=torch.float64, device=GLOBAL_DEVICE)
        self.m_out[:, :, 0] = torch.tensor(self.m_init, dtype=torch.float64, device=GLOBAL_DEVICE).unsqueeze(0)

        self.dt_adc = self.adc_time / self.params.options["max_pulse_samples"]
        self.t = torch.tensor([0], dtype=torch.float64, device=GLOBAL_DEVICE)
        self.total_vec = None
        self.events = []
        self.time_sampling_size = torch.tensor([], device=GLOBAL_DEVICE)

        self.webhook = webhook
        # Füge Callback für Fortschrittsbenachrichtigung hinzu
        self.progress_callback = progress_callback


    def run_adc(self, block, current_adc, mag, counter) -> Tuple[int, torch.Tensor]:
        """
        Handles the simulation of ADC, RF, and gradient blocks.
        Updates the time array, magnetization output, and accumulated phase.

        Parameters
        ----------
        block : Block
            The current block in the sequence.
        current_adc : int
            The current ADC index.
        mag : torch.Tensor
            The current magnetization vector.

        Returns
        -------
        current_adc : int
            Updated ADC index.
        mag : torch.Tensor
            Updated magnetization vector.
        """
        # ADC
        if block.adc is not None:
            new_events = self.events.copy()
            new_m_out = self.m_out.clone()
            new_t = self.t.clone()
            new_current_adc = current_adc

            start_time = new_t[-1]
            new_events.append(f'adc at {start_time.item():.4f}s')

            time_array = start_time + torch.arange(
                1,
                self.params.options["max_pulse_samples"] + 1,
                dtype=torch.float64,
                device=GLOBAL_DEVICE
            ) * self.dt_adc
            self.time_sampling_size = torch.cat((self.time_sampling_size, torch.tensor([len(time_array)], device=GLOBAL_DEVICE)))
            new_t = torch.cat((new_t, time_array))

            for _ in range(len(time_array)):
                self.bm_solver.update_matrix(0, 0, 0)
                mag = self.bm_solver.solve_equation(mag=mag, dtp=self.dt_adc)
                mag = mag.to(new_m_out.dtype)

                index_for_scatter = torch.full(
                    (new_m_out.size(0), new_m_out.size(1), 1),
                    new_current_adc,
                    dtype=torch.long,
                    device=GLOBAL_DEVICE
                )
               
                src_for_scatter = mag.squeeze().unsqueeze(-1)

                new_m_out = new_m_out.scatter(dim=2, index=index_for_scatter, src=src_for_scatter)
                new_current_adc += 1

            self.events = new_events
            self.m_out = new_m_out
            self.t = new_t
            current_adc = new_current_adc

        elif block.rf and block.gz is not None:

            amp_rf, ph_, dtp_rf, delay_after_pulse = prep_rf_simulation(block, self.params.options["max_pulse_samples"])
            amp_gz, _, delay_after_grad = prep_grad_simulation(block, self.params.options["max_pulse_samples"], dtp_rf)

            if len(amp_gz) != len(amp_rf):
                raise Exception(f"Length of RF and gradient amplitudes must be equal, shapes {amp_rf.shape} and {amp_gz.shape}")
            if counter <= self.n_backlog:
                start_time = self.t[-1]
                self.events.append(f'rf and gz at {start_time.item():.4f}s')
                time_array = start_time + torch.arange(1, amp_rf.numel() + 1, dtype=torch.float64, device=GLOBAL_DEVICE) * dtp_rf
                self.time_sampling_size = torch.cat((self.time_sampling_size, torch.tensor([len(time_array)], device=GLOBAL_DEVICE)))
                self.t = torch.cat((self.t, time_array))

            for i in range(amp_rf.numel()):
                self.bm_solver.update_matrix(
                    rf_amp=amp_rf[i],
                    rf_phase=ph_[i],  # statt -ph_[i] + block.rf.phase_offset - accum_phase
                    grad_amp=amp_gz[i],
                    rf_freq=0,                        # Frequenzoffset immer 0
                )
                mag = self.bm_solver.solve_equation(mag=mag, dtp=dtp_rf)
                if counter <= self.n_backlog:
                    self.m_out[:, :, current_adc] = mag.squeeze()
                    current_adc += 1

            if delay_after_pulse > 0:
                self.bm_solver.update_matrix(0, 0, 0)
                mag = self.bm_solver.solve_equation(mag=mag, dtp=delay_after_pulse)
                if counter <= self.n_backlog:
                    start_time = self.t[-1]
                    time_array = start_time + torch.arange(1, 2, dtype=torch.float64, device=GLOBAL_DEVICE) * delay_after_pulse
                    self.time_sampling_size = torch.cat((self.time_sampling_size, torch.tensor([len(time_array)], device=GLOBAL_DEVICE)))
                    self.t = torch.cat((self.t, time_array))
                    self.m_out[:, :, current_adc] = mag.squeeze()
                    current_adc += 1


        # RF pulse
        elif block.rf is not None:
            amp_, ph_, dtp_, delay_after_pulse = prep_rf_simulation(block, self.params.options["max_pulse_samples"])

            if counter <= self.n_backlog:
                start_time = self.t[-1]
                self.events.append(f'rf at {start_time.item():.4f}s')
                time_array = start_time + torch.arange(1, amp_.numel() + 1, dtype=torch.float64, device=GLOBAL_DEVICE) * dtp_
                self.time_sampling_size = torch.cat((self.time_sampling_size, torch.tensor([len(time_array)], device=GLOBAL_DEVICE)))
                self.t = torch.cat((self.t, time_array))
            
            for i in range(amp_.numel()):
                self.bm_solver.update_matrix(
                    rf_amp=amp_[i],
                    rf_phase=ph_[i],  # statt -ph_[i] + block.rf.phase_offset - accum_phase
                    rf_freq=0,                        # Frequenzoffset immer 0
                )
                
                mag = self.bm_solver.solve_equation(mag=mag, dtp=dtp_)
                if counter <= self.n_backlog:
                    self.m_out[:, :, current_adc] = mag.squeeze()
                    current_adc += 1

            if delay_after_pulse > 0:
                self.bm_solver.update_matrix(0, 0, 0)
                mag = self.bm_solver.solve_equation(mag=mag, dtp=delay_after_pulse)
                if counter <= self.n_backlog:
                    start_time = self.t[-1]
                    time_array = start_time + torch.arange(1, 2, dtype=torch.float64, device=GLOBAL_DEVICE) * delay_after_pulse
                    self.time_sampling_size = torch.cat((self.time_sampling_size, torch.tensor([len(time_array)], device=GLOBAL_DEVICE)))
                    self.t = torch.cat((self.t, time_array))
                    self.m_out[:, :, current_adc] = mag.squeeze()
                    current_adc += 1


        # spoiling gradient
        elif block.gz is not None:
            amp_, dtp_, delay_after_grad = prep_grad_simulation(block, self.params.options["max_pulse_samples"])

            if counter <= self.n_backlog:
                start_time = self.t[-1]
                self.events.append(f'gz at {start_time.item():.4f}s')
                time_array = start_time + torch.arange(1, amp_.numel() + 1, dtype=torch.float64, device=GLOBAL_DEVICE) * dtp_
                self.time_sampling_size = torch.cat((self.time_sampling_size, torch.tensor([len(time_array)], device=GLOBAL_DEVICE)))
                self.t = torch.cat((self.t, time_array))

            for i in range(amp_.numel()):
                self.bm_solver.update_matrix(0, 0, 0, grad_amp=amp_[i])
                mag = self.bm_solver.solve_equation(mag=mag, dtp=dtp_)
                if counter <= self.n_backlog:
                    self.m_out[:, :, current_adc] = mag.squeeze()
                    current_adc += 1

            if delay_after_grad > 0:
                self.bm_solver.update_matrix(0, 0, 0)
                mag = self.bm_solver.solve_equation(mag=mag, dtp=delay_after_grad)
                if counter <= self.n_backlog:
                    start_time = self.t[-1]
                    time_array = start_time + torch.arange(1, 2, dtype=torch.float64, device=GLOBAL_DEVICE) * delay_after_grad
                    self.time_sampling_size = torch.cat((self.time_sampling_size, torch.tensor([len(time_array)], device=GLOBAL_DEVICE)))
                    self.t = torch.cat((self.t, time_array))
                    self.m_out[:, :, current_adc] = mag.squeeze()
                    current_adc += 1

        # delay
        elif hasattr(block, "block_duration") and block.block_duration != "0":
            delay = block.block_duration
            sample_factor_delay = int(self.params.options["max_pulse_samples"] / 10)
            dt_delay = delay / sample_factor_delay
            if counter <= self.n_backlog:
                start_time = self.t[-1]
                self.events.append(f'delay at {start_time.item():.4f}s')
                time_array = start_time + torch.arange(1, sample_factor_delay + 1, dtype=torch.float64, device=GLOBAL_DEVICE) * dt_delay
                self.time_sampling_size = torch.cat((self.time_sampling_size, torch.tensor([len(time_array)], device=GLOBAL_DEVICE)))
                self.t = torch.cat((self.t, time_array))

                for step in range(len(time_array)):
                    self.bm_solver.update_matrix(0, 0, 0)
                    mag = self.bm_solver.solve_equation(mag=mag, dtp=dt_delay)
                    self.m_out[:, :, current_adc] = mag.squeeze()
                    current_adc += 1
            else:
                self.bm_solver.update_matrix(0, 0, 0)
                mag = self.bm_solver.solve_equation(mag=mag, dtp=delay)

        else:
            raise Exception("Unknown case")

        return current_adc, mag

    def run_fid(self) -> None:
        """
        Start simulation process.
        """
        if self.n_offsets != self.n_measure:
            self.run_m0_scan = True

        current_adc = 1

        mag = torch.tensor(
            self.m_init[np.newaxis, np.newaxis, :, np.newaxis], 
            dtype=torch.float64,
            device=GLOBAL_DEVICE
        )

        try:
            block_events = self.seq.block_events
        except AttributeError:
            block_events = self.seq.dict_block_events

        # Bestimme den Iterationsmodus basierend auf verbose-Flag und progress_callback
        total_events = len(block_events)
        if self.verbose:
            loop_block_events = tqdm(enumerate(block_events, start=1), total=total_events, desc="BMCTool simulation")
        else:
            loop_block_events = enumerate(block_events, start=1)

        # Discord-Webhook-Benachrichtigungen einrichten
        if self.webhook:
            notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1321535040164728932/LcXvJRZFlns6w18hN4mDkCWuQawYTcWao1GLUOnDYK9QpVUw3lxLLUNl0zIsXdcVBNWK",
                                    total_steps=total_events,
                                    seq_file=self.seq_file,
                                    n_cest_pools=len(self.params.cest_pools),
                                    n_isochromats=self.n_isochromats,
                                    device=f"{GLOBAL_DEVICE.type}"
                                    )
            notifier.send_initial_embed()

        try:
            start_time = time.time()
            
            # Hauptsimulationsschleife
            for i, block_event in loop_block_events:
                counter = np.abs(total_events - i)
                block = self.seq.get_block(block_event)
                current_adc, mag = self.run_adc(block, current_adc, mag, counter)
                
                # Progress updates über WebSocket senden, wenn Callback vorhanden
                if self.progress_callback:
                    self.progress_callback(i, total_events)
                
                # Discord-Webhook-Updates senden
                if self.webhook:
                    notifier.update_progress(i)
            
            # Ergebnisse zuschneiden
            self.m_out = self.m_out[:, :, :self.t.numel()]
            
            # Abschließende Benachrichtigung für Discord-Webhook
            if self.webhook:
                end_time = time.time()
                elapsed_time = timedelta(seconds=end_time - start_time)
                notifier.send_completion_embed(elapsed_time)
                
        except Exception as e:
            # Fehlerbenachrichtigung für Discord-Webhook
            if self.webhook:
                notifier.send_failed_embed(str(e))
            raise
    

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

            m_x = self.m_out[:, 1, :]
            m_y = self.m_out[:, n_total_pools + 1, :]
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

            norm_factor = self.n_isochromats #torch.max(torch.sqrt(m_x_total**2 + m_y_total**2 + m_z_total**2))

            m_z_total /= norm_factor if norm_factor != 0 else 1
            m_x_total /= norm_factor if norm_factor != 0 else 1
            m_y_total /= norm_factor if norm_factor != 0 else 1

            m_trans_c = m_x + 1j * m_y
            m_trans_c_total = m_x_total + 1j * m_y_total

            self.total_vec = torch.stack((m_x_total, m_y_total, m_z_total), dim=-1)

            return self.t, torch.abs(m_z), torch.abs(m_z_total), m_trans_c, m_trans_c_total
        
    

    def get_exact(self) -> tuple:
        """
        Gibt zwei Listen zurück:
        - time_slices: Liste von Torch-Tensoren, die jeweils einen Zeitslice enthalten
        - magnetization_slices: Liste von Torch-Tensoren, die jeweils den entsprechenden Slice der Magnetisierung enthalten
        """
        time_slices = []
        magnetization_slices = []
        start = 0

        print("\n=== Debug: get_exact() ===")
        print(f"Total time samples: {len(self.t)}")
        print(f"Time sampling sizes: {self.time_sampling_size}")
        print(f"Events: {self.events}")
    
        _, _, _, _, magnetization = self.get_mag()
        
        t = self.t if isinstance(self.t, torch.Tensor) else torch.tensor(self.t)
        magnetization = (magnetization if isinstance(magnetization, torch.Tensor)
                        else torch.tensor(magnetization))
        
        for i, size in enumerate(self.time_sampling_size):
            end = int(start + size)
            time_slice = t[start:end]
            mag_slice = magnetization[start:end]
            print(f"\nSlice {i}:")
            print(f"  Event: {self.events[i] if i < len(self.events) else 'No event'}")
            print(f"  Size: {size}")
            print(f"  Time range: {time_slice[0].item():.4f}s - {time_slice[-1].item():.4f}s")
            print(f"  Sample count: {len(time_slice)}")
            
            time_slices.append(time_slice)
            magnetization_slices.append(mag_slice)
            start = end

        print("\n=== End Debug ===\n")
        return time_slices, magnetization_slices
    


    def get_time(self) -> np.ndarray:
        """
        Returns the time array.
        """
        return self.t


    def save_results(self, label: str = "sim", results_root=None):
        """Save all raw simulation data to a timestamped directory under results/simulations/.

        Parameters
        ----------
        label : str
            Short human-readable label for the run, e.g. ``"fisp_30deg"``.
        results_root : Path | str | None
            Override the default results root directory.

        Returns
        -------
        Path
            Absolute path to the created output directory.
        """
        from bmc.utils.results import save_simulation
        return save_simulation(self, label=label, results_root=results_root)


    def animate(self, step: int = 1, run_time=0.1, track_path=False, ie=False, timing=False, total_mag: bool = False, animate_cest: bool = False, fan_surface: bool = False, fan_opacity: float = 0.6, show_vectors: bool = True, **addParams) -> None:
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
        total_mag: bool, optional
            If True, displays total magnetization vector, by default False
        animate_cest: bool, optional
            If True, animates CEST pool magnetization, by default False
        fan_surface: bool, optional
            If True, displays a colored fan-shaped surface connecting vector tips, by default False
        fan_opacity: float, optional
            Opacity of the fan surface, by default 0.6
        show_vectors: bool, optional
            If True, shows individual vectors along with fan surface, by default True
        """

        time = self.t[::step].cpu()
        self.m_out = self.m_out.cpu()
        self.total_vec = self.total_vec.cpu()
        self.z_positions = self.z_positions.cpu()
        isochromats = self.n_isochromats

        # Magnetisierung für Wasserpools vorbereiten
        if animate_cest:
            n_total_pools = len(self.params.cest_pools) + 1
            m_vec_water = np.stack(
                (self.m_out[:, 1, :],
                 self.m_out[:, n_total_pools + 1, :],
                self.m_out[:, self.params.mz_loc + 1, :]),
                axis=2
            )
            # Normiere jeden Vektor in m_vec_water auf Länge 1
            norms = np.linalg.norm(m_vec_water, axis=2, keepdims=True)
            norms[norms == 0] = 1  # Division durch 0 vermeiden
            m_vec_water = m_vec_water / norms
        else:
            if self.params.cest_pools is not None:
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
        if m_vec_total is not None:
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

                    # Hilfsfunktion zum Erstellen der Fächerfläche
                    def create_fan_surface(t_idx, scaling, cols, opacity):
                        """Erstellt eine fächerförmige Fläche aus Dreiecken zwischen benachbarten Vektoren."""
                        triangles = VGroup()
                        origin = np.array([0.0, 0.0, 0.0])
                        
                        for i in range(isochromats - 1):
                            # Vektorspitzen für benachbarte Isochromaten
                            tip1 = np.array([
                                m_vec_water[i, t_idx, 0] * scaling[0],
                                m_vec_water[i, t_idx, 1] * scaling[1],
                                m_vec_water[i, t_idx, 2] * scaling[2]
                            ])
                            tip2 = np.array([
                                m_vec_water[i + 1, t_idx, 0] * scaling[0],
                                m_vec_water[i + 1, t_idx, 1] * scaling[1],
                                m_vec_water[i + 1, t_idx, 2] * scaling[2]
                            ])
                            
                            # Dreieck vom Ursprung zu beiden Vektorspitzen
                            triangle = Polygon(
                                origin, tip1, tip2,
                                fill_opacity=opacity,
                                stroke_width=0.5,
                                stroke_opacity=0.3
                            )
                            
                            # Farbverlauf: Mischung der Farben beider benachbarter Vektoren
                            triangle.set_fill(color=[cols[i], cols[i + 1]])
                            triangle.set_stroke(color=cols[i])
                            
                            triangles.add(triangle)
                        
                        return triangles

                    # Tracker für Vektoren initialisieren
                    for i in range(isochromats):
                        x_tracker = ValueTracker(m_vec_water[i, 0, 0] * scaling_array[0])
                        y_tracker = ValueTracker(m_vec_water[i, 0, 1] * scaling_array[1])
                        z_tracker = ValueTracker(m_vec_water[i, 0, 2] * scaling_array[2])
                        trackers.append((x_tracker, y_tracker, z_tracker))

                        if show_vectors or not fan_surface:
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

                    # Fächerfläche erstellen, falls aktiviert
                    if fan_surface and isochromats > 1:
                        fan_group = create_fan_surface(0, scaling_array, colors, fan_opacity)
                        time_index_tracker = ValueTracker(0)
                        
                        def update_fan(fg, tracker=time_index_tracker, scaling=scaling_array, cols=colors, opacity=fan_opacity):
                            t_idx = int(tracker.get_value())
                            new_fan = create_fan_surface(t_idx, scaling, cols, opacity)
                            fg.become(new_fan)
                        
                        fan_group.add_updater(update_fan)
                        self.add(fan_group)

                    self.add(axes, labels)
                    if show_vectors or not fan_surface:
                        self.add(*vectors)
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
                        animations = []
                        
                        # Vektor-Animationen hinzufügen, wenn Vektoren angezeigt werden
                        if show_vectors or not fan_surface:
                            animations.extend([
                                trackers[i][0].animate.set_value(m_vec_water[i, t, 0] * scaling_array[0]) for i in range(isochromats)
                            ])
                            animations.extend([
                                trackers[i][1].animate.set_value(m_vec_water[i, t, 1] * scaling_array[1]) for i in range(isochromats)
                            ])
                            animations.extend([
                                trackers[i][2].animate.set_value(m_vec_water[i, t, 2] * scaling_array[2]) for i in range(isochromats)
                            ])
                        
                        # Fächerflächen-Animation hinzufügen
                        if fan_surface and isochromats > 1:
                            animations.append(time_index_tracker.animate.set_value(t))
                        
                        # Zeit-Animation hinzufügen
                        if timing:
                            animations.append(time_tracker.animate.set_value(t))
                        
                        self.play(*animations, run_time=run_time, rate_func=linear)

                if ie:
                    self.interactive_embed()

        ipython = get_ipython()
        if ipython:
            ipython.run_line_magic('manim', f'-v WARNING {render_params["quality"]} --disable_caching --renderer=opengl {render_params["write"]} Vector3DScene')
        else:
            print("Magic commands are not supported outside Jupyter notebooks.")