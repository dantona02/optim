import torch
import torch.nn as nn
from bmc.utils.global_device import GLOBAL_DEVICE
from bmc.fid.engine import BMCSim
import numpy as np
from bmc.bmc_tool import prep_rf_simulation
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

    def forward(self, pulse_params):
        """
        pulse_params: Tensor der Form (N, 3) – 
                      jede Zeile enthält [rf_amp, rf_phase, rf_freq] für einen RF-Block,
                      die du optimieren möchtest.
        
        Dieser Forward-Pass:
          1. Setzt den Simulationszustand zurück.
          2. Geht über alle Sequenzblöcke. Wenn ein Block ein RF-Block ist, wird er mit den
             aktuellen Parametern aus pulse_params bearbeitet.
          3. Für andere Blöcke wird der vorhandene Code (z. B. run_adc) aufgerufen.
          4. Am Ende wird das Endsignal (hier z. B. die Transversalmagnetisierung) zurückgegeben.
        """
        rf_block = self.sim_engine.seq.get_block(1)
        rf_freq_offset = [rf_block.rf.freq_offset, -rf_block.rf.freq_offset] # get offet of the rf block
        signals = []

        for rf_offset in rf_freq_offset:

            mag = self.reset_simulation()
            current_adc = 1
            accum_phase = 0.0
            rf_counter = 0
        
            for i, block_event in enumerate(self.sim_engine.seq.block_events, start=1):
                block = self.sim_engine.seq.get_block(block_event)
                total_events = len(self.sim_engine.seq.block_events)
                counter = np.abs(total_events - i)
                if block.rf is not None:
                    # _, _, dtp_, delay_after_pulse = prep_rf_simulation(block, self.sim_engine.params.options["max_pulse_samples"])
                    _, ph_, dtp_, delay_after_pulse = prep_rf_simulation(block, self.sim_engine.params.options["max_pulse_samples"])

                    if block.rf.freq_offset == 0.0:
                        rf_offset = 0.0

                    if counter <= self.sim_engine.n_backlog:
                        start_time = self.t[-1]
                        self.sim_engine.events.append(f'rf at {start_time.item():.4f}s')
                        # time_array = start_time + torch.arange(1, pulse_params[rf_counter, 0].numel() + 1, dtype=torch.float64, device=GLOBAL_DEVICE) * dtp_
                        time_array = start_time + torch.arange(1, pulse_params[rf_counter].numel() + 1, dtype=torch.float64, device=GLOBAL_DEVICE) * dtp_
                        self.t = torch.cat((self.t, time_array))
                    # for i in range(pulse_params[rf_counter, 0].numel()):
                    for i in range(pulse_params[rf_counter].numel()):
                        self.sim_engine.bm_solver.update_matrix(
                            # rf_amp=pulse_params[rf_counter, 0][i],
                            rf_amp=pulse_params[rf_counter, i],
                            # rf_amp=amp_[i],
                            # rf_phase=-pulse_params[rf_counter, 1][i] + block.rf.phase_offset - accum_phase,
                            rf_phase=-ph_[i] + block.rf.phase_offset - accum_phase,
                            rf_freq=rf_offset,
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

                    # phase_degree = dtp_ * pulse_params[rf_counter, 0].numel() * 360 * block.rf.freq_offset
                    phase_degree = dtp_ * pulse_params[rf_counter].numel() * 360 * block.rf.freq_offset
                    phase_degree %= 360
                    accum_phase += phase_degree / 180 * torch.pi
                    rf_counter += 1

                else:
                    current_adc, accum_phase, mag = self.sim_engine.run_adc(block, current_adc, accum_phase, mag, counter)
            
            _, _, _, _, m_trans = self.sim_engine.get_mag()
            signal = torch.max(m_trans.abs())
            signals.append(signal)
        return tuple(signals)

# if __name__ == "__main__":
  
#     seq_file = 'seq_lib/10_ETM.seq'
#     config_file = 'sim_lib/config_1pool.yaml'
#     seq_file = Path(seq_file).resolve()
#     config_file = Path(config_file).resolve()

#     if not Path(config_file).exists():
#         raise FileNotFoundError(f"File {config_file} not found.")

#     if not Path(seq_file).exists():
#         raise FileNotFoundError(f"File {seq_file} not found.")
    
#     sim_params = load_params(config_file)

#     low = -20e-6
#     high = 20e-6
#     n_iso = 10
#     z_pos = np.linspace(low, high, n_iso)
#     z_pos = torch.tensor(z_pos)
#     z_pos = torch.cat((z_pos, torch.tensor([0.0]))) 

#     sim_engine_instance = BMCSim(adc_time=5e-3,
#                                  params=sim_params,
#                                  seq_file=seq_file,
#                                  z_positions=z_pos,
#                                  n_backlog=0,
#                                  verbose=True,
#                                  webhook=False)
#     diff_sim = DifferentiableBMCSimWrapper(sim_engine_instance)
    
#     rf_parameters_list = []
#     for i, block_event in enumerate(diff_sim.sim_engine.seq.block_events, start=1):
#             block = diff_sim.sim_engine.seq.get_block(block_event)
#             if block.rf is not None:
#                 amp_, ph_, _, _ = prep_rf_simulation(block, diff_sim.sim_engine.params.options["max_pulse_samples"])
#                 block_params = torch.stack([amp_, ph_]) #[rf_amp, rf_phase, rf_freq, delay_after_pulse]
#                 rf_parameters_list.append(block_params)
#     if rf_parameters_list:
#         rf_parameters_tensor = torch.stack(rf_parameters_list)
#         rf_parameters_tensor.requires_grad_(True)
#     else:
#         print("No RF blocks found.")

#     end_signal = diff_sim(rf_parameters_tensor)
#     print("Endsignal:", end_signal)
#     end_signal.backward()
#     print("Gradienten für RF-Parameter:", rf_parameters_tensor.grad)

    # def loss_function(end_signal, pulse_params, lambda_smooth=1e-3):
    #     primary_loss = -end_signal
    #     smoothness_loss = torch.mean(torch.abs(pulse_params[:, 1:] - pulse_params[:, :-1]))
    #     total_loss = primary_loss + lambda_smooth * smoothness_loss
    #     return total_loss
        
    # diff_sim.reset_simulation()
    # loss.backward()