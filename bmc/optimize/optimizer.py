import torch
import torch.nn as nn
from bmc.utils.global_device import GLOBAL_DEVICE
from bmc.fid.engine import BMCSim
import numpy as np
from bmc.bmc_tool import prep_rf_simulation, prep_grad_simulation
from bmc.set_params import load_params
from pathlib import Path

#torch.autograd.set_detect_anomaly(True)

class DifferentiableBMCSimWrapper(nn.Module):
    def __init__(self, sim_engine: BMCSim):
        """
        sim_engine: Eine Instanz deiner BMCSim-Klasse, die den kompletten Simulationsablauf kapselt.
                    (Sie enthält z. B. Attribute wie m_init, t, seq, bm_solver, dt_adc usw.)
        """
        super(DifferentiableBMCSimWrapper, self).__init__()
        self.sim_engine = sim_engine

    def reset_simulation(self):
        """
        Setzt den Simulationszustand zurück.
        Wir nehmen an, dass self.sim_engine.m_init als NumPy‑Array oder Tensor vorliegt.
        """
        mag = torch.tensor(
            self.sim_engine.m_init[np.newaxis, np.newaxis, :, np.newaxis], 
            dtype=torch.float64,
            device=GLOBAL_DEVICE
        )
        self.sim_engine.t = torch.tensor([0.0], dtype=torch.float64, device=GLOBAL_DEVICE)
        self.sim_engine.bm_solver.update_params(self.sim_engine.params)

        return mag

    def forward(self, rf_params=None, grad_params=None):
        """
        Extended Forward Pass:
        - rf_params: List of [amplitude_tensor, phase_tensor] for each RF pulse
        - grad_params: (Optional) Gradient parameters (Tensor)
        """
        rf_freq_offset = [1, -1]
        signals = []

        for offset in rf_freq_offset:
            mag = self.reset_simulation()
            current_adc = 1
            rf_counter = 0
            grad_counter = 0
            
            total_events = len(self.sim_engine.seq.block_events)
            for i, block_event in enumerate(self.sim_engine.seq.block_events, start=1):
                block = self.sim_engine.seq.get_block(block_event)
                counter = np.abs(total_events - i)
                if block.rf is not None:
                    if rf_params is None:
                        block.rf.freq_offset = offset if block.rf.freq_offset != 0.0 else 1
                        current_adc, mag = self.sim_engine.run_adc(
                            block, current_adc, mag, counter
                        )
                    else:
                        _, _, dtp_, delay_after_pulse = prep_rf_simulation(
                            block, self.sim_engine.params.options["max_pulse_samples"]
                        )
                        rf_offset = offset if block.rf.freq_offset != 0.0 else 0.0
                        
                        amp_params, phase_params = rf_params[rf_counter]
                        for step_idx in range(amp_params.numel()):
                            self.sim_engine.bm_solver.update_matrix(
                                rf_amp=amp_params[step_idx],
                                rf_phase=rf_offset * phase_params[step_idx],
                                rf_freq=0,
                            )
                            mag = self.sim_engine.bm_solver.solve_equation(mag=mag, dtp=dtp_)
                            if counter <= self.sim_engine.n_backlog:
                                self.sim_engine.m_out[:, :, current_adc] = mag.squeeze()
                                current_adc += 1

                        if delay_after_pulse > 0:
                            self.sim_engine.bm_solver.update_matrix(0, 0, 0)
                            mag = self.sim_engine.bm_solver.solve_equation(mag=mag, dtp=delay_after_pulse)
                            if counter <= self.sim_engine.n_backlog:
                                start_time = self.t[-1]
                                time_array = start_time + torch.arange(1, 2, dtype=torch.float64, device=GLOBAL_DEVICE) * delay_after_pulse
                                self.t = torch.cat((self.t, time_array))
                                self.sim_engine.m_out[:, :, current_adc] = mag.squeeze()
                                current_adc += 1
                        
                        rf_counter += 1

                elif block.gz is not None:
                    if grad_params is None or grad_counter >= len(grad_params):
                        current_adc, mag = self.sim_engine.run_adc(block, current_adc, mag, counter)
                        continue

                    amp_gz, dtp_gz, delay_after_grad = prep_grad_simulation(
                        block, self.sim_engine.params.options["max_pulse_samples"]
                    )
                    custom_grad = grad_params[grad_counter] if grad_params is not None else amp_gz
                    for g_idx in range(amp_gz.numel()):
                        self.sim_engine.bm_solver.update_matrix(0, 0, 0, grad_amp=custom_grad[g_idx])
                        mag = self.sim_engine.bm_solver.solve_equation(mag=mag, dtp=dtp_gz)
                        if counter <= self.sim_engine.n_backlog:
                            self.sim_engine.m_out[:, :, current_adc] = mag.squeeze()
                            current_adc += 1

                    if delay_after_grad > 0:
                        self.sim_engine.bm_solver.update_matrix(0, 0, 0)
                        mag = self.sim_engine.bm_solver.solve_equation(mag=mag, dtp=delay_after_grad)
                        if counter <= self.sim_engine.n_backlog:
                            start_time = self.t[-1]
                            time_array = start_time + torch.arange(1, 2, dtype=torch.float64, device=GLOBAL_DEVICE) * delay_after_grad
                            self.t = torch.cat((self.t, time_array))
                            self.sim_engine.m_out[:, :, current_adc] = mag.squeeze()
                            current_adc += 1

                    grad_counter += 1

                else:
                    current_adc, mag = self.sim_engine.run_adc(
                        block, current_adc, mag, counter
                    )

            _, _, _, _, m_trans = self.sim_engine.get_mag()
            signals.append(m_trans.abs())

        difference = torch.max(signals[0] - signals[1])
        return difference

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    config_file = base_dir / "sim_lib" / "config_1pool.yaml"
    seq_file = base_dir / "seq_lib" / "RACETE.seq"

    if not Path(config_file).exists():
        raise FileNotFoundError(f"File {config_file} not found.")

    if not Path(seq_file).exists():
        raise FileNotFoundError(f"File {seq_file} not found.")
    
    sim_params = load_params(config_file)

    low = -1e-3
    high = 1e-3
    n_iso = 100
    z_pos = np.linspace(low, high, n_iso)
    z_pos = torch.tensor(z_pos)
    z_pos = torch.cat((z_pos, torch.tensor([0.0]))) 

    sim_engine_instance = BMCSim(adc_time=5e-3,
                                params=sim_params,
                                seq_file=seq_file,
                                z_positions=z_pos,
                                n_backlog=1,
                                verbose=True,
                                webhook=False)
    diff_sim = DifferentiableBMCSimWrapper(sim_engine_instance)
    
    rf_parameters_list = []
    grad_parameters_list = []
    for i, block_event in enumerate(diff_sim.sim_engine.seq.block_events, start=1):
        block = diff_sim.sim_engine.seq.get_block(block_event)
        if block.rf is not None:
            amp_rf, ph_rf, dtp_rf, _ = prep_rf_simulation(
                block, diff_sim.sim_engine.params.options["max_pulse_samples"]
            )
            # Store both amplitude and phase parameters
            rf_parameters_list.append([amp_rf, ph_rf])
        if block.gz is not None:
            amp_gz, dtp_gz, delay_after_grad = prep_grad_simulation(
                block, diff_sim.sim_engine.params.options["max_pulse_samples"]
            )
            grad_parameters_list.append(amp_gz)

    if rf_parameters_list:
        # Convert list of [amp, phase] pairs into a list of tensors that require gradients
        rf_amp_tensors = []
        rf_phase_tensors = []
        for amp, phase in rf_parameters_list:
            amp_tensor = amp.clone().detach().requires_grad_(True)
            phase_tensor = phase.clone().detach().requires_grad_(True)
            rf_amp_tensors.append(amp_tensor)
            rf_phase_tensors.append(phase_tensor)
        rf_parameters = list(zip(rf_amp_tensors, rf_phase_tensors))
    else:
        rf_parameters = None

    if grad_parameters_list:
        grad_parameters_tensor = torch.stack(grad_parameters_list)
        grad_parameters_tensor.requires_grad_(True)
    else:
        grad_parameters_tensor = None

    end_signal = diff_sim(rf_parameters, grad_params=grad_parameters_tensor)
    print("Endsignal:", end_signal.item())
    end_signal.backward()
    
    if rf_parameters is not None:
        print("\nGradienten für RF:")
        for i, (amp_tensor, phase_tensor) in enumerate(rf_parameters):
            print(f"RF Pulse {i}:")
            print("Amplitude grad:", amp_tensor.grad)
            print("Phase grad:", phase_tensor.grad)
    
    if grad_parameters_tensor is not None:
        print("\nGradienten für GZ:", grad_parameters_tensor.grad)