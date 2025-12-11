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

def signal_objective_function(signal, rf_parameters=None, edge_penalty_weight=0.0, smoothness_weight=0.0, negative_penalty_weight=0.0):
    """
    Computes the objective value for optimizing the signal.
    We maximize the signal, so the goal is to minimize the negative signal value.
    With an optional penalty for non-zero values at the ends of the pulse.
    
    Args:
        signal: Signal tensor
        rf_parameters: List of RF parameters [amp, phase] (optional)
        edge_penalty_weight: Weight for the penalty on non-zero ends (0.0 = disabled)
        smoothness_weight: Weight for smoothing the pulse shape (0.0 = disabled)
        negative_penalty_weight: Weight for penalizing negative amplitude values (0.0 = disabled)
        
    Returns:
        Loss: Negative signal (+ optional penalties)
    """

    signal_loss = -torch.abs(signal)
    
    total_penalty = 0.0
    
    if rf_parameters is not None:
        if edge_penalty_weight > 0:
            edge_penalty = 0.0
            for amp, _ in rf_parameters:
                n_samples = 1
                first_samples = amp[:n_samples]
                last_samples = amp[-n_samples:]
                
                edge_penalty += torch.sum(first_samples**2) + torch.sum(last_samples**2)
            
            total_penalty += edge_penalty_weight * edge_penalty
        
        if smoothness_weight > 0:
            smoothness_penalty = 0.0
            for amp, _ in rf_parameters:
                diffs = amp[1:] - amp[:-1]
                smoothness_penalty += torch.sum(diffs**2)
            
            total_penalty += smoothness_weight * smoothness_penalty
        
        if negative_penalty_weight > 0:
            negative_penalty = 0.0
            for amp, _ in rf_parameters:
                negative_values = torch.nn.functional.relu(-amp)
                negative_penalty += torch.sum(negative_values**2)
            
            total_penalty += negative_penalty_weight * negative_penalty
    
    return signal_loss + total_penalty

def save_checkpoint(checkpoint_dir, rf_parameters, grad_parameters, optimizer, epoch, loss, signal, 
                   checkpoint_name=None):
    """
    Saves a checkpoint with all relevant training parameters.
    
    Args:
        checkpoint_dir: Directory for storing checkpoints
        rf_parameters: RF pulse parameters (amplitude and phase)
        grad_parameters: Gradient parameters
        optimizer: Current optimizer state
        epoch: Current epoch number
        loss: Current loss value
        signal: Current signal value
        checkpoint_name: Optional custom name for the checkpoint file
    """
    if checkpoint_name is None:
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    rf_state = []
    if rf_parameters:
        for amp, phase in rf_parameters:
            rf_state.append({
                'amplitude': amp.detach(),
                'phase': phase.detach()
            })
    
    grad_state = [grad.detach() for grad in grad_parameters] if grad_parameters is not None else None
    
    checkpoint = {
        'epoch': epoch,
        'rf_parameters': rf_state,
        'grad_parameters': grad_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'signal': signal
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)
    
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
    Loads a checkpoint and restores the training state.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        optimizer: Optional optimizer to load state into
        
    Returns:
        Tuple of (rf_parameters, grad_parameters, epoch, loss, signal)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    rf_parameters = []
    if checkpoint['rf_parameters']:
        for rf_state in checkpoint['rf_parameters']:
            amp = rf_state['amplitude'].requires_grad_(True)
            phase = rf_state['phase'].requires_grad_(True)
            rf_parameters.append((amp, phase))
    
    grad_parameters = None
    if checkpoint['grad_parameters'] is not None:
        grad_parameters = [grad.requires_grad_(True) for grad in checkpoint['grad_parameters']]
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return rf_parameters, grad_parameters, checkpoint['epoch'], checkpoint['loss'], checkpoint['signal']

def get_params(config_file, seq_file):
    """
    Initializes and returns all necessary parameters for training.
    
    Returns:
        Tuple of (rf_parameters, grad_parameters_list, sim_params, seq_file, z_pos)
    """

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
            amp_rf, ph_rf, _, _ = prep_rf_simulation(
                block, diff_sim.sim_engine.params.options["max_pulse_samples"]
            )
            # Store both amplitude and phase parameters
            rf_parameters_list.append([amp_rf, ph_rf])
        if block.gz is not None:
            amp_gz, _, _ = prep_grad_simulation(
                block, diff_sim.sim_engine.params.options["max_pulse_samples"]
            )
            # Store each gradient as a separate tensor with gradients enabled
            grad_tensor = amp_gz.clone().detach().requires_grad_(True)
            grad_parameters_list.append(grad_tensor)

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

    # Instead of stacking, we return the list of individual gradient tensors
    if not grad_parameters_list:
        grad_parameters_list = None

    return rf_parameters, grad_parameters_list, sim_params, z_pos

def compute_gradient_flow(parameters):
    """
    Calculates the L2 norm of gradients for all optimizable parameters.
    
    Args:
        parameters: List of parameters with gradients
        
    Returns:
        Dict with parameter names and their L2 gradient norms
    """
    grad_norms = {}
    
    if isinstance(parameters, list):
        for i, param in enumerate(parameters):
            if hasattr(param, 'grad') and param.grad is not None:
                norm = param.grad.norm(2).item()
                grad_norms[f'param_{i}'] = norm
    else:
        # If parameters is a dict of named parameters
        for name, param in parameters.items():
            if param.grad is not None:
                norm = param.grad.norm(2).item()
                grad_norms[name] = norm
                
    return grad_norms

def train(config_file, seq_file, num_epochs=50, learning_rate=0.1, checkpoint_dir='checkpoints', 
          checkpoint_frequency=10, resume_from=None,
          edge_penalty_weight=0.0, smoothness_weight=0.0, negative_penalty_weight=0.0, only_grad=False):
    """
    Training with checkpoint support and configurable loss penalties.
    
    Args:
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        checkpoint_dir: Directory for storing checkpoints
        checkpoint_frequency: Checkpoint saving frequency in epochs
        resume_from: Optional path to checkpoint for resuming training
        edge_penalty_weight: Weight for edge penalty in loss function
        smoothness_weight: Weight for smoothness penalty in loss function
        negative_penalty_weight: Weight for negative amplitude penalty in loss function
        
    Returns:
        Tuple of (optimized_rf_parameters, optimized_gradient_parameters)
    """

    if resume_from is None:
        rf_parameters, grad_parameters_list, sim_params, z_pos = get_params(config_file=config_file, seq_file=seq_file)
        start_epoch = 0
        best_loss = float('inf')
    else:
        sim_params, seq_file, z_pos = get_params()[2:]
        rf_parameters, grad_parameters_list, start_epoch, best_loss, _ = load_checkpoint(resume_from)
    
    sim_engine_instance = BMCSim(adc_time=5e-3,
                               params=sim_params,
                               seq_file=seq_file,
                               z_positions=z_pos,
                               n_backlog=1,
                               verbose=True,
                               webhook=False)
    diff_sim = DifferentiableBMCSimWrapper(sim_engine_instance)

    # Organize all parameters and provide labels for more informative monitoring
    all_parameters = []
    parameter_names = {}
    
    if rf_parameters:
        for i, (amp_tensor, phase_tensor) in enumerate(rf_parameters):
            all_parameters.extend([amp_tensor, phase_tensor])
            parameter_names[len(all_parameters)-2] = f"rf_{i+1}_amplitude"
            parameter_names[len(all_parameters)-1] = f"rf_{i+1}_phase"
    
    if grad_parameters_list is not None:
        all_parameters.extend(grad_parameters_list)
        for i, grad_tensor in enumerate(grad_parameters_list):
            parameter_names[len(all_parameters)-len(grad_parameters_list)+i] = f"gradient_{i+1}"

    if only_grad and grad_parameters_list is not None:
        optimizer = optim.Adam(grad_parameters_list, lr=learning_rate)
    else:
        optimizer = optim.Adam(all_parameters, lr=learning_rate)
    
    if resume_from:
        _, _, _, _, _ = load_checkpoint(resume_from, optimizer)
    
    # Create figure with 3 subplots: loss, signal, and gradient flow
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    loss_values = []
    signal_values = []
    epochs = []
    
    # Dictionary to store gradient norms for each parameter over time
    gradient_history = {name: [] for name in parameter_names.values()}

    for epoch in range(start_epoch, num_epochs):
        optimizer.zero_grad()
        end_signal = diff_sim(rf_parameters, grad_params=grad_parameters_list)
        loss = signal_objective_function(
            end_signal, 
            rf_parameters=rf_parameters, 
            edge_penalty_weight=edge_penalty_weight, 
            smoothness_weight=smoothness_weight, 
            negative_penalty_weight=negative_penalty_weight
        )
        loss.backward()
        
        # Compute and save gradient norms before optimization step
        named_parameters = {parameter_names[i]: param for i, param in enumerate(all_parameters)}
        gradient_norms = compute_gradient_flow(named_parameters)
        
        for name, norm in gradient_norms.items():
            gradient_history[name].append(norm)
            
        optimizer.step()
        
        # Update plots
        loss_values.append(loss.item())
        signal_values.append(torch.abs(end_signal).item())
        epochs.append(epoch)
        
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        ax1.plot(epochs, loss_values, 'b-')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        
        ax2.plot(epochs, signal_values, 'r-')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('End Signal')
        ax2.set_title('End Signal')
        
        # Plot gradient flow (L2 norm of gradients) for each parameter
        for name, norms in gradient_history.items():
            if norms:  # Only plot if we have data
                ax3.plot(epochs[-len(norms):], norms, label=name)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Gradient L2 Norm')
        ax3.set_title('Gradient Flow')
        ax3.set_yscale('log')  # Log scale often helps visualize gradient norms better
        ax3.legend(loc='upper right', fontsize='small')
        
        plt.tight_layout()
        plt.pause(0.01)
        
        # Log information including gradient norms
        grad_info = ", ".join([f"{name}: {norm:.6f}" for name, norm in gradient_norms.items()])
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, End Signal = {torch.abs(end_signal).item():.6f}")
        print(f"Gradient L2 Norms: {grad_info}")
        
        if (epoch + 1) % checkpoint_frequency == 0:
            save_checkpoint(
                checkpoint_dir,
                rf_parameters,
                grad_parameters_list,
                optimizer,
                epoch,
                loss.item(),
                torch.abs(end_signal).item()
            )
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            save_checkpoint(
                checkpoint_dir,
                rf_parameters,
                grad_parameters_list,
                optimizer,
                epoch,
                loss.item(),
                torch.abs(end_signal).item(),
                'best_checkpoint.pt'
            )
    
    save_checkpoint(
        checkpoint_dir,
        rf_parameters,
        grad_parameters_list,
        optimizer,
        num_epochs-1,
        loss.item(),
        torch.abs(end_signal).item(),
        'final_checkpoint.pt'
    )
    
    return rf_parameters, grad_parameters_list

if __name__ == "__main__":
    rf_params_optimized, grad_params_optimized = train(
        config_file='/Users/danielmiksch/JupyterLab/optim/sim_lib/config_1pool.yaml',
        seq_file='/Users/danielmiksch/JupyterLab/optim/seq_lib/RACETE.seq',
        num_epochs=30,
        # learning_rate=500e7,
        learning_rate=5,
        checkpoint_frequency=10,
        edge_penalty_weight=0.001,
        smoothness_weight=0.0001,
        negative_penalty_weight=0.001,
        only_grad=False
        # resume_from='checkpoints/best_checkpoint.pt'  # Optional for resuming
    )
    print("Training finished.")