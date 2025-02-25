import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import os
import json
from datetime import datetime
from bmc.fid.engine import BMCSim
from bmc.optimize.optimizer import DifferentiableBMCSimWrapper
from bmc.bmc_tool import prep_rf_simulation, prep_grad_simulation
from bmc.set_params import load_params
from bmc.utils.global_device import GLOBAL_DEVICE
import matplotlib.pyplot as plt

def l2_loss_function(signal_diff):
    return -torch.norm(signal_diff, p=2)

def save_checkpoint(checkpoint_dir, rf_parameters, grad_parameters, optimizer, epoch, loss, signal, 
                   checkpoint_name=None):
    """
    Speichert einen Checkpoint mit allen relevanten Trainingsparametern
    """
    if checkpoint_name is None:
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
    
    # Erstelle Checkpoint-Verzeichnis falls nicht vorhanden
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Bereite RF-Parameter für Speicherung vor
    rf_state = []
    if rf_parameters:
        for amp, phase in rf_parameters:
            rf_state.append({
                'amplitude': amp.detach(),
                'phase': phase.detach()
            })
    
    # Bereite Gradienten-Parameter für Speicherung vor
    grad_state = grad_parameters.detach() if grad_parameters is not None else None
    
    # Erstelle Checkpoint-Dictionary
    checkpoint = {
        'epoch': epoch,
        'rf_parameters': rf_state,
        'grad_parameters': grad_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'signal': signal
    }
    
    # Speichere Checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)
    
    # Speichere zusätzliche Metadaten
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'epoch': epoch,
        'loss': float(loss),
        'signal': float(signal)
    }
    metadata_path = os.path.join(checkpoint_dir, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def load_checkpoint(checkpoint_path, optimizer=None):
    """
    Lädt einen Checkpoint und stellt den Trainingszustand wieder her
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    # Erstelle neue Parameter-Listen
    rf_parameters = []
    if checkpoint['rf_parameters']:
        for rf_state in checkpoint['rf_parameters']:
            amp = rf_state['amplitude'].requires_grad_(True)
            phase = rf_state['phase'].requires_grad_(True)
            rf_parameters.append((amp, phase))
    
    grad_parameters = None
    if checkpoint['grad_parameters'] is not None:
        grad_parameters = checkpoint['grad_parameters'].requires_grad_(True)
    
    # Lade Optimizer-Zustand falls vorhanden
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return rf_parameters, grad_parameters, checkpoint['epoch'], checkpoint['loss'], checkpoint['signal']

def get_params():
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

    return rf_parameters, grad_parameters_tensor, sim_params, seq_file, z_pos

def train(num_epochs=50, learning_rate=0.1, checkpoint_dir='checkpoints', 
          checkpoint_frequency=10, resume_from=None):
    """
    Training mit Checkpoint-Unterstützung
    
    Args:
        num_epochs: Anzahl der Trainings-Epochen
        learning_rate: Lernrate für den Optimizer
        checkpoint_dir: Verzeichnis für Checkpoints
        checkpoint_frequency: Speicherintervall für Checkpoints (in Epochen)
        resume_from: Optional, Pfad zu einem Checkpoint zum Fortsetzen des Trainings
    """
    # Initialisiere Parameter und Optimizer
    if resume_from is None:
        rf_parameters, grad_parameters_tensor, sim_params, seq_file, z_pos = get_params()
        start_epoch = 0
        best_loss = float('inf')
    else:
        # Lade Parameter aus Checkpoint
        sim_params, seq_file, z_pos = get_params()[2:]  # Hole nur die Sim-Parameter
        rf_parameters, grad_parameters_tensor, start_epoch, best_loss, _ = load_checkpoint(resume_from)
    
    # Sammle alle Parameter für den Optimizer
    all_parameters = []
    if rf_parameters:
        for amp_tensor, phase_tensor in rf_parameters:
            all_parameters.extend([amp_tensor, phase_tensor])
    if grad_parameters_tensor is not None:
        all_parameters.append(grad_parameters_tensor)

    optimizer = optim.Adam(all_parameters, lr=learning_rate)
    
    if resume_from:
        # Lade Optimizer-Zustand wenn wir von Checkpoint fortsetzen
        _, _, _, _, _ = load_checkpoint(resume_from, optimizer)
    
    # Initialize plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    loss_values = []
    signal_values = []
    epochs = []

    for epoch in range(start_epoch, num_epochs):
        sim_engine_instance = BMCSim(adc_time=5e-3,
                                   params=sim_params,
                                   seq_file=seq_file,
                                   z_positions=z_pos,
                                   n_backlog=0,
                                   verbose=True,
                                   webhook=False)
        diff_sim = DifferentiableBMCSimWrapper(sim_engine_instance)
        
        optimizer.zero_grad()
        end_signal = diff_sim(rf_parameters, grad_params=grad_parameters_tensor)
        loss = l2_loss_function(end_signal)
        loss.backward()
        optimizer.step()
        
        # Update plots
        loss_values.append(loss.item())
        signal_values.append(end_signal.item())
        epochs.append(epoch)
        
        ax1.clear()
        ax2.clear()
        
        ax1.plot(epochs, loss_values, 'b-')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        
        ax2.plot(epochs, signal_values, 'r-')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('End Signal')
        ax2.set_title('End Signal')
        
        plt.tight_layout()
        plt.pause(0.01)
        
        print(f"Epoch {epoch}: Loss = {loss.item()}, End Signal = {end_signal.item()}")
        
        # Speichere Checkpoint
        if (epoch + 1) % checkpoint_frequency == 0:
            save_checkpoint(
                checkpoint_dir,
                rf_parameters,
                grad_parameters_tensor,
                optimizer,
                epoch,
                loss.item(),
                end_signal.item()
            )
            
        # Speichere zusätzlich besten Checkpoint
        if loss.item() < best_loss:
            best_loss = loss.item()
            save_checkpoint(
                checkpoint_dir,
                rf_parameters,
                grad_parameters_tensor,
                optimizer,
                epoch,
                loss.item(),
                end_signal.item(),
                'best_checkpoint.pt'
            )
    
    # Speichere finalen Zustand
    save_checkpoint(
        checkpoint_dir,
        rf_parameters,
        grad_parameters_tensor,
        optimizer,
        num_epochs-1,
        loss.item(),
        end_signal.item(),
        'final_checkpoint.pt'
    )
    
    return rf_parameters, grad_parameters_tensor

if __name__ == "__main__":
    # Beispiel für Training mit Checkpoint-Unterstützung
    rf_params_optimized, grad_params_optimized = train(
        num_epochs=15,
        checkpoint_frequency=5,
        # resume_from='checkpoints/checkpoint_epoch_20.pt'  # Optional zum Fortsetzen
    )
