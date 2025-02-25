import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional

def load_checkpoint(checkpoint_path: str) -> Dict:
    """
    Loads a checkpoint and returns its contained parameters.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt file)
        
    Returns:
        Dictionary containing the checkpoint data
    """
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

def load_training_history(checkpoint_dir: str) -> List[Dict]:
    """
    Loads training history from all available checkpoints.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        
    Returns:
        List of dictionaries containing the training history
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    checkpoint_files.extend([checkpoint_dir / "best_checkpoint.pt", checkpoint_dir / "final_checkpoint.pt"])
    
    history = []
    for checkpoint_file in checkpoint_files:
        if checkpoint_file.exists():
            checkpoint = load_checkpoint(str(checkpoint_file))
            history.append({
                'epoch': checkpoint['epoch'],
                'loss': checkpoint['loss'],
                'signal': checkpoint['signal']
            })
    
    # Sort by epoch
    history.sort(key=lambda x: x['epoch'])
    return history

def get_best_checkpoint(checkpoint_dir: str) -> Tuple[str, Dict]:
    """
    Finds the best checkpoint based on the loss value.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        
    Returns:
        Tuple of (path to best checkpoint, checkpoint data)
    """
    best_checkpoint_path = Path(checkpoint_dir) / 'best_checkpoint.pt'
    if not best_checkpoint_path.exists():
        raise FileNotFoundError(f"No best_checkpoint.pt found in {checkpoint_dir}")
        
    checkpoint = load_checkpoint(str(best_checkpoint_path))
    return str(best_checkpoint_path), checkpoint

def plot_rf_parameters(checkpoint: Dict, pulse_idx: Optional[int] = None):
    """
    Plots RF parameters (amplitude and phase) from a checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint data
        pulse_idx: Optional, index of RF pulse to plot. 
                  If None, all pulses will be plotted.
    """
    rf_parameters = checkpoint['rf_parameters']
    
    if pulse_idx is not None:
        if pulse_idx >= len(rf_parameters):
            raise ValueError(f"Pulse index {pulse_idx} out of valid range")
        rf_parameters = [rf_parameters[pulse_idx]]
    
    num_pulses = len(rf_parameters)
    fig, axes = plt.subplots(num_pulses, 2, figsize=(12, 4*num_pulses))
    if num_pulses == 1:
        axes = axes.reshape(1, -1)
    
    for i, rf_pulse in enumerate(rf_parameters):
        amp = rf_pulse['amplitude']
        phase = rf_pulse['phase']
        
        # Plot Amplitude
        axes[i, 0].plot(amp.numpy())
        axes[i, 0].set_title(f'RF Pulse {i+1} - Amplitude')
        axes[i, 0].set_xlabel('Sample')
        axes[i, 0].set_ylabel('Amplitude')
        
        # Plot Phase
        axes[i, 1].plot(phase.numpy())
        axes[i, 1].set_title(f'RF Pulse {i+1} - Phase')
        axes[i, 1].set_xlabel('Sample')
        axes[i, 1].set_ylabel('Phase')
    
    plt.tight_layout()
    plt.show()

def plot_gradient_parameters(checkpoint: Dict):
    """
    Plots gradient parameters from a checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint data
    """
    grad_parameters = checkpoint['grad_parameters']
    
    if grad_parameters is None:
        print("No gradient parameters found in checkpoint")
        return
        
    # Convert to numpy for plotting
    grad_parameters = grad_parameters.numpy()
    
    # Plot for each gradient
    num_gradients = grad_parameters.shape[0]
    fig, axes = plt.subplots(num_gradients, 1, figsize=(12, 4*num_gradients))
    if num_gradients == 1:
        axes = [axes]
    
    for i, grad in enumerate(grad_parameters):
        axes[i].plot(grad, 'g-')
        axes[i].set_title(f'Gradient {i+1}')
        axes[i].set_xlabel('Sample')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_training_progress(checkpoint_dir: str):
    """
    Plots training progress based on checkpoint history.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
    """
    history = load_training_history(checkpoint_dir)
    
    if not history:
        raise ValueError("No checkpoints found")
    
    epochs = [entry['epoch'] for entry in history]
    losses = [entry['loss'] for entry in history]
    signals = [entry['signal'] for entry in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(epochs, losses, 'b-o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    ax2.plot(epochs, signals, 'r-o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Signal')
    ax2.set_title('End Signal')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    checkpoint_dir = "checkpoints"
    
    # Load and show training progress
    print("Plotting training progress...")
    plot_training_progress(checkpoint_dir)
    
    # Load best checkpoint
    print("\nLoading best checkpoint...")
    best_checkpoint_path, best_checkpoint = get_best_checkpoint(checkpoint_dir)
    print(f"Best checkpoint from epoch {best_checkpoint['epoch']}")
    print(f"Loss: {best_checkpoint['loss']}")
    print(f"Signal: {best_checkpoint['signal']}")
    
    # Plot RF parameters from best checkpoint
    print("\nPlotting RF parameters...")
    plot_rf_parameters(best_checkpoint)
    
    # Plot gradient parameters
    print("\nPlotting gradient parameters...")
    plot_gradient_parameters(best_checkpoint)