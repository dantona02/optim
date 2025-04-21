import torch
import torch.nn as nn
from bmc.utils.global_device import GLOBAL_DEVICE
from bmc.fid.engine import BMCSim
import numpy as np
from bmc.bmc_tool import prep_rf_simulation, prep_grad_simulation
from pathlib import Path

class SingleSimBMCWrapper(nn.Module):
    """
    Differenzierbarer Wrapper für BMC-Simulation mit nur einem Simulationsdurchlauf.
    Optimiert für die direkte Signalmaximierung ohne Differenzbildung.
    """
    def __init__(self, sim_engine: BMCSim):
        """
        Initialisiert den differenzierbaren Wrapper für die BMC-Simulation.

        Args:
            sim_engine: Eine Instanz der BMCSim-Klasse, die den gesamten Simulationsprozess kapselt.
        """
        super(SingleSimBMCWrapper, self).__init__()
        self.sim_engine = sim_engine

    def reset_simulation(self):
        """
        Setzt alle Simulationszustände auf ihre Anfangswerte zurück.
        Dies stellt die exakte Äquivalenz mit einer frischen Initialisierung sicher.
        
        Returns:
            Anfangsmagnetisierungstensor
        """
        # Magnetisierung auf Anfangszustand zurücksetzen
        mag = torch.tensor(
            self.sim_engine.m_init[np.newaxis, np.newaxis, :, np.newaxis], 
            dtype=torch.float64,
            device=GLOBAL_DEVICE
        )
        
        # Zeit auf 0 zurücksetzen
        self.sim_engine.t = torch.tensor([0.0], dtype=torch.float64, device=GLOBAL_DEVICE)
        
        # Solver-Parameter zurücksetzen
        self.sim_engine.bm_solver.update_params(self.sim_engine.params)
        
        # Ausgabetensor auf Anfangszustand zurücksetzen
        self.sim_engine.m_out = torch.zeros(
            self.sim_engine.n_isochromats, 
            self.sim_engine.m_init.shape[0], 
            self.sim_engine.n_measure, 
            dtype=torch.float64, 
            device=GLOBAL_DEVICE
        )
        self.sim_engine.m_out[:, :, 0] = torch.tensor(
            self.sim_engine.m_init, 
            dtype=torch.float64, 
            device=GLOBAL_DEVICE
        ).unsqueeze(0)
        
        # Event-Tracking zurücksetzen
        self.sim_engine.events = []
        self.sim_engine.time_sampling_size = torch.tensor([], device=GLOBAL_DEVICE)
        self.sim_engine.total_vec = None

        return mag

    def forward(self, rf_params=None, grad_params=None):
        """
        Forward-Pass für BMC-Simulation mit einem einzelnen Durchlauf.
        
        Args:
            rf_params: Tupel aus (Amplituden-Tensor, Phasen-Tensor) für jeden RF-Puls
            grad_params: (Optional) Gradienten-Parameter (Tensor)
            
        Returns:
            Signal am Ende der Simulation
        """
        # Simulation zurücksetzen und initialisieren
        mag = self.reset_simulation()
        current_adc = 1
        rf_counter = 0
        grad_counter = 0
        
        total_events = len(self.sim_engine.seq.block_events)
        for i, block_event in enumerate(self.sim_engine.seq.block_events, start=1):
            block = self.sim_engine.seq.get_block(block_event)
            counter = np.abs(total_events - i)

            # Behandle RF und Gradient gleichzeitig
            if block.rf is not None and block.gz is not None:
                if rf_params is None:
                    current_adc, mag = self.sim_engine.run_adc(block, current_adc, mag, counter)
                    continue

                amp_rf, ph_rf, dtp_rf, delay_after_pulse = prep_rf_simulation(
                    block, self.sim_engine.params.options["max_pulse_samples"]
                )
                amp_gz, dtp_gz, delay_after_grad = prep_grad_simulation(
                    block, self.sim_engine.params.options["max_pulse_samples"],
                    dtp_rf
                )

                if len(amp_gz) != len(amp_rf):
                    raise Exception(f"Length of RF and gradient amplitudes must be equal, shapes {amp_rf.shape} and {amp_gz.shape}")

                amp_params, phase_params = rf_params[rf_counter]
                grad_param = grad_params[grad_counter] if grad_params is not None else amp_gz

                if counter <= self.sim_engine.n_backlog:
                    start_time = self.sim_engine.t[-1]
                    time_array = start_time + torch.arange(1, amp_params.numel() + 1, dtype=torch.float64, device=GLOBAL_DEVICE) * dtp_rf
                    self.sim_engine.t = torch.cat((self.sim_engine.t, time_array))

                for step_idx in range(amp_params.numel()):
                    self.sim_engine.bm_solver.update_matrix(
                        rf_amp=amp_params[step_idx],
                        rf_phase=phase_params[step_idx],  # Kein Offset-Faktor mehr
                        rf_freq=0,
                        grad_amp=grad_param[step_idx]
                    )
                    mag = self.sim_engine.bm_solver.solve_equation(mag=mag, dtp=dtp_rf)
                    if counter <= self.sim_engine.n_backlog:
                        self.sim_engine.m_out[:, :, current_adc] = mag.squeeze()
                        current_adc += 1

                if delay_after_pulse > 0:
                    self.sim_engine.bm_solver.update_matrix(0, 0, 0)
                    mag = self.sim_engine.bm_solver.solve_equation(mag=mag, dtp=delay_after_pulse)
                    if counter <= self.sim_engine.n_backlog:
                        start_time = self.sim_engine.t[-1]
                        time_array = start_time + torch.arange(1, 2, dtype=torch.float64, device=GLOBAL_DEVICE) * delay_after_pulse
                        self.sim_engine.t = torch.cat((self.sim_engine.t, time_array))
                        self.sim_engine.m_out[:, :, current_adc] = mag.squeeze()
                        current_adc += 1

                rf_counter += 1
                grad_counter += 1

            # Nur RF Pulse
            elif block.rf is not None:
                if rf_params is None:
                    current_adc, mag = self.sim_engine.run_adc(
                        block, current_adc, mag, counter
                    )
                else:
                    _, _, dtp_, delay_after_pulse = prep_rf_simulation(
                        block, self.sim_engine.params.options["max_pulse_samples"]
                    )
                    
                    amp_params, phase_params = rf_params[rf_counter]
                    for step_idx in range(amp_params.numel()):
                        self.sim_engine.bm_solver.update_matrix(
                            rf_amp=amp_params[step_idx],
                            rf_phase=phase_params[step_idx],  # Kein Offset-Faktor mehr
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
                            start_time = self.sim_engine.t[-1]
                            time_array = start_time + torch.arange(1, 2, dtype=torch.float64, device=GLOBAL_DEVICE) * delay_after_pulse
                            self.sim_engine.t = torch.cat((self.sim_engine.t, time_array))
                            self.sim_engine.m_out[:, :, current_adc] = mag.squeeze()
                            current_adc += 1
                    
                    rf_counter += 1

            # Nur Gradient
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
                        start_time = self.sim_engine.t[-1]
                        time_array = start_time + torch.arange(1, 2, dtype=torch.float64, device=GLOBAL_DEVICE) * delay_after_grad
                        self.sim_engine.t = torch.cat((self.sim_engine.t, time_array))
                        self.sim_engine.m_out[:, :, current_adc] = mag.squeeze()
                        current_adc += 1

                grad_counter += 1

            else:
                current_adc, mag = self.sim_engine.run_adc(
                    block, current_adc, mag, counter
                )

        # Magnetisierung abrufen und transversales Signal zurückgeben
        _, _, _, _, m_trans = self.sim_engine.get_mag()
        signal = m_trans.abs()
        signal = torch.max(signal)
        
        # Signal direkt zurückgeben (kein Maximum einer Differenz)
        return signal

if __name__ == "__main__":
    from bmc.set_params import load_params
    
    # Test-Code für den SingleSimBMCWrapper
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    config_file = base_dir / "sim_lib" / "config_1pool.yaml"
    seq_file = base_dir / "seq_lib" / "50.seq"  # Wir verwenden die 50.seq Datei

    if not Path(config_file).exists():
        raise FileNotFoundError(f"File {config_file} not found.")

    if not Path(seq_file).exists():
        raise FileNotFoundError(f"File {seq_file} not found.")
    
    sim_params = load_params(config_file)

    # Erstelle Positionen für die Simulation
    low = -1e-3
    high = 1e-3
    n_iso = 100
    z_pos = np.linspace(low, high, n_iso)
    z_pos = torch.tensor(z_pos)
    z_pos = torch.cat((z_pos, torch.tensor([0.0]))) 

    # Initialisiere Simulationsengine mit n_backlog=0 für Signal am Beginn des ADC
    sim_engine_instance = BMCSim(adc_time=5e-3,
                                params=sim_params,
                                seq_file=seq_file,
                                z_positions=z_pos,
                                n_backlog=0,  # Signal am Beginn des ADC
                                verbose=True,
                                webhook=False)
    
    # Unseren neuen Wrapper verwenden
    single_sim = SingleSimBMCWrapper(sim_engine_instance)
    
    # RF-Parameter extrahieren
    rf_parameters_list = []
    for i, block_event in enumerate(single_sim.sim_engine.seq.block_events, start=1):
        block = sim_engine_instance.seq.get_block(block_event)
        if block.rf:
            if hasattr(block, "block_duration") and block.block_duration != "0":
                amp_, ph_, dtp_rf, _ = prep_rf_simulation(
                    block, sim_engine_instance.params.options["max_pulse_samples"])
                print(f"RF-Block {i}: Form mit {len(amp_)} Samples, dtp_rf: {dtp_rf}")
                
                # Amplitude und Phase für den RF-Puls speichern
                rf_parameters_list.append([amp_, ph_])

    if rf_parameters_list:
        # Parameter für Optimierung vorbereiten
        rf_amp_tensors = []
        rf_phase_tensors = []
        for amp, phase in rf_parameters_list:
            amp_tensor = amp.clone().detach().requires_grad_(True)
            phase_tensor = phase.clone().detach()  # Phase benötigt keinen Gradienten
            rf_amp_tensors.append(amp_tensor)
            rf_phase_tensors.append(phase_tensor)
        rf_parameters = list(zip(rf_amp_tensors, rf_phase_tensors))
        
        # Test-Signal berechnen
        signal = single_sim(rf_parameters)
        print(f"Signal: {signal.item()}")
    else:
        print("Keine RF-Blöcke in der Sequenz gefunden.")
