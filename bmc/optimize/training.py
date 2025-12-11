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
from bmc.utils.eval import calculate_flip_angle

def l2_loss_function(signal_diff):
    """
    Calculates L2 loss for the signal difference.
    
    Args:
        signal_diff: Signal difference tensor
        
    Returns:
        Negative L2 norm of the signal difference
    """
    return -torch.norm(signal_diff, p=2)

def signal_objective_function(signal, rf_parameters=None, edge_penalty_weight=0.0, smoothness_weight=0.0, negative_penalty_weight=0.0):
    """
    Calculates the objective value for signal optimization.
    We maximize the signal, so the goal is to minimize the negative signal value.
    With optional penalties for non-zero values at pulse edges, smoothness, and negative values.
    
    Args:
        signal: Signal tensor
        rf_parameters: List of RF parameters [amp, phase] (optional)
        edge_penalty_weight: Weight for penalty on non-zero edges (0.0 = disabled)
        smoothness_weight: Weight for penalty on pulse shape smoothness (0.0 = disabled)
        negative_penalty_weight: Weight for penalty on negative amplitude values (0.0 = disabled)
        
    Returns:
        Loss: Negative signal (+ penalties if applicable)
    """
    # For optimization, we want to maximize the absolute signal value
    signal_loss = -torch.abs(signal)
    
    total_penalty = 0.0
    
    # Apply penalties if rf_parameters are provided
    if rf_parameters is not None:
        # Penalty for non-zero edges
        if edge_penalty_weight > 0:
            edge_penalty = 0.0
            for amp, _ in rf_parameters:
                # Calculate penalty for first and last N samples
                n_samples = 1  # Number of samples at start and end that should be close to 0
                first_samples = amp[:n_samples]  # First N samples
                last_samples = amp[-n_samples:]  # Last N samples
                
                # Calculate penalty (squared to penalize larger deviations more)
                edge_penalty += torch.sum(first_samples**2) + torch.sum(last_samples**2)
            
            total_penalty += edge_penalty_weight * edge_penalty
        
        # Penalty for non-smooth transitions
        if smoothness_weight > 0:
            smoothness_penalty = 0.0
            for amp, _ in rf_parameters:
                # Calculate differences between consecutive samples
                diffs = amp[1:] - amp[:-1]
                
                # Penalize large differences (squared)
                smoothness_penalty += torch.sum(diffs**2)
            
            total_penalty += smoothness_weight * smoothness_penalty
        
        # Penalty for negative amplitude values
        if negative_penalty_weight > 0:
            negative_penalty = 0.0
            for amp, _ in rf_parameters:
                # Find all negative values (ReLU negates)
                negative_values = torch.nn.functional.relu(-amp)
                # Square penalty for negative values
                negative_penalty += torch.sum(negative_values**2)
            
            total_penalty += negative_penalty_weight * negative_penalty
    
    return signal_loss + total_penalty

def save_checkpoint(checkpoint_dir, rf_parameters, grad_parameters, optimizer, epoch, loss, signal, 
                   checkpoint_name=None, dtp_rf=None):
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
        dtp_rf: RF pulse sampling time (important for correct reconstruction)
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
    
    # Add pulse parameters if available
    if dtp_rf is not None:
        checkpoint['parameters'] = {
            'dt': float(dtp_rf),  # Sample time (very important for correct pulse usage)
            'num_samples': len(rf_parameters[0][0]) if rf_parameters else 0  # Number of samples in pulse
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
    
    # Add dt to metadata
    if dtp_rf is not None:
        metadata['dt'] = float(dtp_rf)
        
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
        Tuple of (rf_parameters, grad_parameters_tensor, sim_params, seq_file, z_pos, diff_sim, dtp_rf)
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    config_file = base_dir / "sim_lib" / "config_1pool.yaml"
    seq_file = base_dir / "seq_lib" / "custom_ETM.seq"  # Use SE.seq for Spin-Echo optimization

    if not Path(config_file).exists():
        raise FileNotFoundError(f"Configuration file {config_file} not found.")

    if not Path(seq_file).exists():
        raise FileNotFoundError(f"Sequence file {seq_file} not found.")
    
    print(f"Using sequence file: {seq_file}")
    print(f"Using configuration file: {config_file}")
    
    sim_params = load_params(config_file)

    # Create positions for simulation with 500 isochromates
    low = -1e-3
    high = 1e-3
    n_iso = 500  # Increased to 500 isochromates for better accuracy
    z_pos = np.linspace(low, high, n_iso)
    z_pos = torch.tensor(z_pos)
    z_pos = torch.cat((z_pos, torch.tensor([0.0]))) 
    
    print(f"Number of isochromates: {len(z_pos)}")

    # Initialize simulation engine with n_backlog=0 for signal at beginning of ADC
    sim_engine_instance = BMCSim(adc_time=50e-3,  # Longer echo time for SE sequence
                               params=sim_params,
                               seq_file=seq_file,
                               z_positions=z_pos,
                               n_backlog=0,  # Signal at beginning of ADC
                               verbose=True,
                               webhook=False)
    diff_sim = DifferentiableBMCSimWrapper(sim_engine_instance)
    
    rf_parameters_list = []
    grad_parameters_list = []
    dtp_rf = None
    
    for i, block_event in enumerate(diff_sim.sim_engine.seq.block_events, start=1):
        block = diff_sim.sim_engine.seq.get_block(block_event)
        if block.rf is not None:
            amp_rf, ph_rf, dtp_rf, _ = prep_rf_simulation(
                block, diff_sim.sim_engine.params.options["max_pulse_samples"]
            )
            print(f"RF Block {i}: Shape with {len(amp_rf)} samples, dtp_rf: {dtp_rf}")
            
            # Make amplitudes require gradients
            amp_rf.requires_grad_(True)
            
            # Store amplitude and phase parameters
            rf_parameters_list.append([amp_rf, ph_rf])
            
        if block.gz is not None:
            amp_gz, dtp_gz, delay_after_grad = prep_grad_simulation(
                block, diff_sim.sim_engine.params.options["max_pulse_samples"]
            )
            # Make gradients require gradients
            amp_gz.requires_grad_(True)
            grad_parameters_list.append(amp_gz)

    if rf_parameters_list:
        # For SE optimization, we should check we have the expected number of pulses
        if len(rf_parameters_list) < 2:
            print("Warning: Found fewer than 2 RF pulses in the SE sequence.")
        else:
            print(f"Number of RF pulses: {len(rf_parameters_list)}")
            print(f"RF Pulse 1 Amplitude Shape: {rf_parameters_list[0][0].shape}")
            print(f"RF Pulse 2 Amplitude Shape: {rf_parameters_list[1][0].shape}")
            
        rf_parameters = rf_parameters_list
    else:
        rf_parameters = None
        raise ValueError("No RF blocks found in the sequence.")

    if grad_parameters_list:
        grad_parameters_tensor = torch.stack(grad_parameters_list)
        grad_parameters_tensor.requires_grad_(True)
    else:
        grad_parameters_tensor = None

    return rf_parameters, grad_parameters_tensor, sim_params, seq_file, z_pos, diff_sim, dtp_rf

def visualize_pulses(parameters, dtp, title="RF-Pulse"):
    """
    Visualizes multiple RF pulses.
    
    Args:
        parameters: List of [amp, phase] pairs
        dtp: Sampling time of the pulses
        title: Base title for the plot
    """
    plt.figure(figsize=(12, 8))
    
    # For each pulse in parameters
    for i, [amp, _] in enumerate(parameters):
        pulse = amp.detach().cpu().numpy()
        time_axis = np.arange(len(pulse)) * dtp * 1000  # Convert to ms
        
        plt.subplot(len(parameters), 1, i+1)
        plt.plot(time_axis, pulse, linewidth=2)
        plt.title(f"{title} {i+1}" + (" (90°)" if i == 0 else " (180°)" if i == 1 else ""))
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.grid(True)
    
    plt.tight_layout()

def compare_pulses(original_parameters, optimized_parameters, dtp_rf):
    """
    Compares original and optimized pulse shapes.
    
    Args:
        original_parameters: List of original [amp, phase] pairs
        optimized_parameters: List of optimized [amp, phase] pairs
        dtp_rf: Sampling time
    """
    plt.figure(figsize=(14, 10))
    
    # For each pulse
    for i in range(len(optimized_parameters)):
        plt.subplot(len(optimized_parameters), 1, i+1)
        
        # Original pulse
        pulse_orig = original_parameters[i][0].detach().cpu().numpy()
        time_axis = np.arange(len(pulse_orig)) * dtp_rf * 1000  # Convert to ms
        plt.plot(time_axis, pulse_orig, 'b-', linewidth=1.5, alpha=0.7, label='Original')
        
        # Optimized pulse
        pulse_opt = optimized_parameters[i][0].detach().cpu().numpy()
        plt.plot(time_axis, pulse_opt, 'r-', linewidth=1.5, alpha=0.7, label='Optimized')
        
        plt.title(f"Comparison Pulse {i+1}" + (" (90°)" if i == 0 else " (180°)" if i == 1 else ""))
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    
    return plt.gcf()

def plot_progress(epochs, losses, signals, save_path=None):
    """
    Plots training progress.
    
    Args:
        epochs: List of epoch numbers
        losses: List of loss values
        signals: List of signal values
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Loss plot
    ax1.plot(epochs, losses, 'b-', marker='o')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress')
    ax1.grid(True)
    
    # Signal plot
    ax2.plot(epochs, signals, 'r-', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Signal')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def initialize_custom_rf_pulse(original_rf_parameters, init_method='random', scaling_factor=0.5):
    """
    Initializes RF pulses with different methods to give the optimizer more freedom.
    
    Args:
        original_rf_parameters: Original RF parameters as list of [amp, phase] pairs
        init_method: Initialization method ('random', 'flat', 'sine', 'mixed', 'original')
        scaling_factor: Amplitude scaling factor (for 'random', 'sine', 'mixed')
        
    Returns:
        New RF parameters with initialized pulse shape
    """
    custom_rf_parameters = []
    
    for i, [amp, phase] in enumerate(original_rf_parameters):
        # Keep original phase
        new_phase = phase.clone().detach()
        
        # Get dimensions of original pulse
        pulse_length = len(amp)
        max_amp_value = torch.max(amp).item()
        
        # Initialize based on chosen method
        if init_method == 'random':
            # Random initialization (uniformly distributed)
            new_amp = torch.rand_like(amp) * max_amp_value * scaling_factor
            
        elif init_method == 'flat':
            # Flat initialization (constant amplitude)
            mean_amp = torch.mean(amp).item()
            new_amp = torch.ones_like(amp) * mean_amp
            
        elif init_method == 'sine':
            # Sine initialization (a different shape than Gaussian)
            x = torch.linspace(0, 2*torch.pi, pulse_length, device=amp.device, dtype=amp.dtype)
            new_amp = torch.sin(x) * max_amp_value * scaling_factor
            # Ensure values are positive
            new_amp = torch.abs(new_amp)
            
        elif init_method == 'mixed':
            # Mix of original and randomness
            random_component = torch.rand_like(amp) * max_amp_value * scaling_factor
            # Detach original to separate computation graph
            original_component = amp.clone().detach() * (1 - scaling_factor)
            new_amp = random_component + original_component
            
        else:  # 'original' or other values
            # Keep original shape
            new_amp = amp.clone().detach()
        
        # Ensure we have a leaf tensor (detach)
        # and create a new copy (clone)
        new_amp = new_amp.detach().clone()
        # Then activate requires_grad
        new_amp.requires_grad_(True)
        
        # Add new pair to list
        custom_rf_parameters.append([new_amp, new_phase])
    
    return custom_rf_parameters

def train(num_epochs=200, learning_rate=5.0, 
          edge_penalty_weight=0.0001, smoothness_weight=0.0001, negative_penalty_weight=0.001,
          checkpoint_dir='checkpoints/SE_optimization', checkpoint_frequency=10,
          resume_from=None, init_method='mixed', custom_init_scaling=0.5,
          lr_decay=0.998, momentum=0.9, use_scheduler=True, use_adam=True):
    """
    Training for RF pulse optimization with extended options.
    For the SE sequence, both pulses (90° and 180°) are optimized simultaneously.
    
    Args:
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        edge_penalty_weight: Weight for penalty on non-zero edges
        smoothness_weight: Weight for penalty on pulse smoothness
        negative_penalty_weight: Weight for penalty on negative amplitude values
        checkpoint_dir: Directory for storing checkpoints
        checkpoint_frequency: Checkpoint saving frequency in epochs
        resume_from: Optional path to checkpoint for resuming training
        init_method: Method for initializing the RF pulse
        custom_init_scaling: Scaling factor for custom initialization
        lr_decay: Factor for learning rate decay when using a scheduler
        momentum: Momentum for SGD optimizer (if use_adam=False)
        use_scheduler: Whether to use a learning rate scheduler
        use_adam: Whether to use Adam instead of SGD
        
    Returns:
        Tuple of (best_parameters, best_signal, final_checkpoint_path, progress_plot)
    """
    # Initialize parameters
    if resume_from is None:
        # Get fresh parameters
        rf_parameters, grad_parameters_tensor, sim_params, seq_file, z_pos, diff_sim, dtp_rf = get_params()
        
        # Initialize RF parameters with the chosen method if not 'original'
        if init_method != 'original':
            rf_parameters = initialize_custom_rf_pulse(rf_parameters, init_method, custom_init_scaling)
            print(f"RF pulses reinitialized with method '{init_method}', scaling factor: {custom_init_scaling}")
            
        # Store original parameters for later comparison
        original_pulse_parameters = []
        for amp, phase in rf_parameters:
            original_pulse_parameters.append([amp.detach().clone(), phase.detach().clone()])
            
    else:
        # Load parameters from checkpoint
        print(f"Resuming training from checkpoint: {resume_from}")
        
        # Only get simulation environment components
        _, _, sim_params, seq_file, z_pos, diff_sim, dtp_rf = get_params()
        
        # Load checkpoint data
        checkpoint = torch.load(resume_from)
        
        # Create RF parameters from checkpoint
        rf_parameters = []
        for rf_state in checkpoint['rf_parameters']:
            amp = rf_state['amplitude'].requires_grad_(True)
            phase = rf_state['phase'].requires_grad_(True)
            rf_parameters.append([amp, phase])
        
        # Get original parameters for comparison
        original_pulse_parameters = get_params()[0]
        
        # Gradient parameters (if any)
        grad_parameters_tensor = None
        if 'grad_parameters' in checkpoint and checkpoint['grad_parameters'] is not None:
            grad_parameters_tensor = checkpoint['grad_parameters'].requires_grad_(True)
            
        # Start from next epoch
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Prepare optimizable parameters
    optimizable_params = []
    for amp, _ in rf_parameters:
        optimizable_params.append(amp)
    if grad_parameters_tensor is not None:
        optimizable_params.append(grad_parameters_tensor)
    
    # Initialize optimizer
    if use_adam:
        optimizer = optim.Adam(optimizable_params, lr=learning_rate)
    else:
        optimizer = optim.SGD(optimizable_params, lr=learning_rate, momentum=momentum)
        
    # Optional learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
        
    # Load optimizer state if resuming
    if resume_from and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate  # Reset learning rate
            print("Successfully loaded optimizer state")
        except:
            print("Could not load optimizer state, using newly initialized optimizer")
    
    # Variables for tracking
    best_signal = float('-inf')
    best_parameters = None
    start_epoch = 0 if resume_from is None else start_epoch
    epochs_list = []
    losses_list = []
    signals_list = []
    learning_rates = []
    
    # Measure initial signal
    with torch.no_grad():
        original_signal = diff_sim(original_pulse_parameters, grad_params=grad_parameters_tensor)
        print(f"Original signal: {original_signal.item():.6f}")
        
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Forward pass: get signal from simulation
        signal = diff_sim(rf_parameters, grad_params=grad_parameters_tensor)
        
        # Calculate loss with custom objective function
        loss = signal_objective_function(
            signal, 
            rf_parameters, 
            edge_penalty_weight, 
            smoothness_weight, 
            negative_penalty_weight
        )
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Extract pure signal value (negative loss without penalties)
        pure_signal_value = torch.abs(signal).item()
        
        # Tracking
        epochs_list.append(epoch)
        losses_list.append(loss.item())
        signals_list.append(pure_signal_value)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Output
        print(f"Epoch {epoch}/{num_epochs - 1} - Signal: {pure_signal_value:.6f}, Loss: {loss.item():.6f}, LR: {current_lr:.6f}")
        
        # Save the best state (based on pure signal)
        if pure_signal_value > best_signal:
            best_signal = pure_signal_value
            # Clone current parameters
            best_parameters = []
            for amp, phase in rf_parameters:
                best_parameters.append([amp.detach().clone(), phase.detach().clone()])
                
            save_checkpoint(
                checkpoint_dir, 
                best_parameters, 
                grad_parameters_tensor, 
                optimizer, 
                epoch, 
                loss.item(), 
                pure_signal_value,
                checkpoint_name="best_checkpoint.pt",
                dtp_rf=dtp_rf
            )
            print(f"  New best value found and saved!")
            
        # Save regular checkpoints
        if epoch % checkpoint_frequency == 0 or epoch == num_epochs - 1:
            current_parameters = []
            for amp, phase in rf_parameters:
                current_parameters.append([amp.detach().clone(), phase.detach().clone()])
                
            save_checkpoint(
                checkpoint_dir, 
                current_parameters, 
                grad_parameters_tensor, 
                optimizer, 
                epoch, 
                loss.item(), 
                pure_signal_value,
                dtp_rf=dtp_rf
            )
            
        # Update scheduler if used
        if scheduler and epoch < num_epochs - 1:  # Don't update in the last epoch
            scheduler.step()
    
    # Save final state
    final_parameters = []
    for amp, phase in rf_parameters:
        final_parameters.append([amp.detach().clone(), phase.detach().clone()])
        
    final_path = save_checkpoint(
        checkpoint_dir, 
        final_parameters, 
        grad_parameters_tensor, 
        optimizer, 
        num_epochs - 1, 
        losses_list[-1], 
        signals_list[-1],
        checkpoint_name="final_checkpoint.pt",
        dtp_rf=dtp_rf
    )
    
    # Plot progress
    progress_fig = plot_progress(
        epochs_list, 
        losses_list, 
        signals_list,
        save_path=os.path.join(checkpoint_dir, "training_progress.png")
    )
    
    # Additional plot for learning rates
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_list, learning_rates, 'g-')
    plt.title('Learning Rate Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "learning_rate_progress.png"))
    
    # Compare original and optimized pulses
    with torch.no_grad():
        original_signal = diff_sim(original_pulse_parameters, grad_params=grad_parameters_tensor)
        optimized_signal = diff_sim(best_parameters, grad_params=grad_parameters_tensor)
        
    print("\nTraining completed!")
    print(f"Original signal: {original_signal.item():.6f}")
    print(f"Optimized signal: {optimized_signal.item():.6f}")
    
    if abs(original_signal.item()) > 0:
        improvement = (optimized_signal.item() - original_signal.item()) / abs(original_signal.item()) * 100
        print(f"Improvement: {improvement:.2f}%")
    
    # Calculate flip angles of optimized pulses
    for i, [amp, _] in enumerate(best_parameters):
        flip_angle = calculate_flip_angle(amp, dtp_rf)
        pulse_type = "90°" if i == 0 else "180°" if i == 1 else f"#{i+1}"
        print(f"Calculated flip angle of {pulse_type} pulse: {flip_angle:.2f}°")
    
    # Visualize the optimized pulses
    visualize_pulses(best_parameters, dtp_rf, "Optimized Pulse")
    
    # Compare pulses
    compare_plot = compare_pulses(original_pulse_parameters, best_parameters, dtp_rf)
    compare_plot.savefig(os.path.join(checkpoint_dir, "pulse_comparison.png"))
    
    return best_parameters, best_signal, final_path, progress_fig

if __name__ == "__main__":
    # Training für die Spin-Echo-Sequenz mit optimierten Parametern
    best_parameters, best_signal, final_checkpoint_path, progress_plot = train(
        num_epochs=200,
        learning_rate=5.0,
        edge_penalty_weight=0.0001,
        smoothness_weight=0.0001,
        negative_penalty_weight=0.001,
        checkpoint_dir='checkpoints/SE_optimization',
        checkpoint_frequency=10,
        init_method='mixed',
        custom_init_scaling=0.5,
        lr_decay=0.998,
        use_scheduler=True,
        use_adam=True
        # Zum Fortsetzen des Trainings:
        # resume_from='checkpoints/SE_optimization/best_checkpoint.pt'
    )
    
    print(f"Optimierung abgeschlossen! Bestes Signal: {best_signal:.6f}")