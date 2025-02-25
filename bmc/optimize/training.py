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
    """
    Calculates L2 loss for the signal difference.
    
    Args:
        signal_diff: Signal difference tensor
        
    Returns:
        Negative L2 norm of the signal difference
    """
    return -torch.norm(signal_diff, p=2)

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
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Prepare RF parameters for storage
    rf_state = []
    if rf_parameters:
        for amp, phase in rf_parameters:
            rf_state.append({
                'amplitude': amp.detach(),
                'phase': phase.detach()
            })
    
    # Prepare gradient parameters for storage
    grad_state = grad_parameters.detach() if grad_parameters is not None else None
    
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'rf_parameters': rf_state,
        'grad_parameters': grad_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'signal': signal
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)
    
    # Save additional metadata
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
    
    # Create new parameter lists
    rf_parameters = []
    if checkpoint['rf_parameters']:
        for rf_state in checkpoint['rf_parameters']:
            amp = rf_state['amplitude'].requires_grad_(True)
            phase = rf_state['phase'].requires_grad_(True)
            rf_parameters.append((amp, phase))
    
    grad_parameters = None
    if checkpoint['grad_parameters'] is not None:
        grad_parameters = checkpoint['grad_parameters'].requires_grad_(True)
    
    # Load optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return rf_parameters, grad_parameters, checkpoint['epoch'], checkpoint['loss'], checkpoint['signal']

def get_params():
    """
    Initializes and returns all necessary parameters for training.
    
    Returns:
        Tuple of (rf_parameters, grad_parameters_tensor, sim_params, seq_file, z_pos)
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    config_file = base_dir / "sim_lib" / "config_1pool.yaml"
#     seq_file = base_dir / "seq_lib" / "RACETE.seq"
    seq_file = base_dir / "seq_lib" / "custom_ETM.seq"

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
    Training with checkpoint support.
    
    Args:
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        checkpoint_dir: Directory for storing checkpoints
        checkpoint_frequency: Checkpoint saving frequency in epochs
        resume_from: Optional path to checkpoint for resuming training
        
    Returns:
        Tuple of (optimized_rf_parameters, optimized_gradient_parameters)
    """
    # Initialize parameters and optimizer
    if resume_from is None:
        rf_parameters, grad_parameters_tensor, sim_params, seq_file, z_pos = get_params()
        start_epoch = 0
        best_loss = float('inf')
    else:
        # Load parameters from checkpoint
        sim_params, seq_file, z_pos = get_params()[2:]  # Get only sim params
        rf_parameters, grad_parameters_tensor, start_epoch, best_loss, _ = load_checkpoint(resume_from)
    
    # Create simulation instance
    sim_engine_instance = BMCSim(adc_time=5e-3,
                               params=sim_params,
                               seq_file=seq_file,
                               z_positions=z_pos,
                               n_backlog=1,
                               verbose=True,
                               webhook=False)
    diff_sim = DifferentiableBMCSimWrapper(sim_engine_instance)
    
    # Collect all parameters for optimizer
    all_parameters = []
    if rf_parameters:
        for amp_tensor, phase_tensor in rf_parameters:
            all_parameters.extend([amp_tensor, phase_tensor])
    if grad_parameters_tensor is not None:
        all_parameters.append(grad_parameters_tensor)

    optimizer = optim.Adam(all_parameters, lr=learning_rate)
    
    if resume_from:
        # Load optimizer state if resuming from checkpoint
        _, _, _, _, _ = load_checkpoint(resume_from, optimizer)
    
    # Initialize plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    loss_values = []
    signal_values = []
    epochs = []

    for epoch in range(start_epoch, num_epochs):
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
        
        # Save checkpoint
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
            
        # Save best checkpoint
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
    
    # Save final state
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
    # Example for training with checkpoint support
    rf_params_optimized, grad_params_optimized = train(
        num_epochs=10,
        checkpoint_frequency=5,
        # resume_from='checkpoints/checkpoint_epoch_20.pt'  # Optional for resuming
    )
