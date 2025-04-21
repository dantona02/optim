"""
Gauss Pulse Signal Optimization

Dieses Modul bietet eine vereinfachte Optimierungsimplementierung für Gaußpulse.
Im Gegensatz zur allgemeinen Implementierung in training.py wird hier das Signal
zu Beginn des ADC-Events maximiert, anstatt eine Signaldifferenz zu optimieren.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from bmc.fid.engine import BMCSim
from bmc.optimize.optimizer import DifferentiableBMCSimWrapper
from bmc.bmc_tool import prep_rf_simulation, prep_grad_simulation
from bmc.set_params import load_params
from bmc.utils.global_device import GLOBAL_DEVICE

def signal_objective_function(signal):
    """
    Berechnet den Zielwert für die Optimierung des Signals.
    Wir maximieren das Signal, also ist das Ziel, den negativen Signalwert zu minimieren.
    
    Args:
        signal: Signal-Tensor
        
    Returns:
        Negatives Signal (für Minimierungsproblem)
    """
    # Nehmen wir an, dass der letzte Wert des Signals (ADC-Wert) wichtig ist
    return -torch.abs(signal)

def save_checkpoint(checkpoint_dir, rf_parameters, optimizer, epoch, loss, signal, 
                  checkpoint_name=None):
    """
    Speichert einen Checkpoint mit allen relevanten Trainingsparametern.
    
    Args:
        checkpoint_dir: Verzeichnis zum Speichern der Checkpoints
        rf_parameters: RF-Pulsparameter (Amplitude)
        optimizer: Aktueller Optimierer-Zustand
        epoch: Aktuelle Epochennummer
        loss: Aktueller Verlustwert
        signal: Aktueller Signalwert
        checkpoint_name: Optionaler Name für die Checkpoint-Datei
    """
    if checkpoint_name is None:
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
    
    # Erstelle Checkpoint-Verzeichnis, falls es nicht existiert
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Bereite RF-Parameter für die Speicherung vor
    rf_state = rf_parameters.detach().clone() if rf_parameters is not None else None
    
    # Erstelle Checkpoint-Dictionary
    checkpoint = {
        'epoch': epoch,
        'rf_parameters': rf_state,
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
    metadata_path = os.path.join(checkpoint_dir, 'gauss_training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def load_checkpoint(checkpoint_path, optimizer=None):
    """
    Lädt einen Checkpoint und stellt den Trainingszustand wieder her.
    
    Args:
        checkpoint_path: Pfad zur Checkpoint-Datei
        optimizer: Optionaler Optimierer, um den Zustand zu laden
        
    Returns:
        Tuple aus (rf_parameters, epoch, loss, signal)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint-Datei {checkpoint_path} wurde nicht gefunden.")
    
    checkpoint = torch.load(checkpoint_path)
    
    # RF-Parameter laden
    rf_parameters = checkpoint['rf_parameters']
    
    # Optimierer-Zustand laden, falls vorhanden
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return rf_parameters, checkpoint['epoch'], checkpoint['loss'], checkpoint['signal']

def get_params():
    """
    Initialisiert und gibt alle notwendigen Parameter für das Training zurück.
    
    Returns:
        Tuple aus (rf_parameters_tensor, sim_params, seq_file, z_pos)
    """
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    config_file = base_dir / "sim_lib" / "config_1pool.yaml"
    seq_file = base_dir / "seq_lib" / "90.seq"

    if not Path(config_file).exists():
        raise FileNotFoundError(f"Konfigurationsdatei {config_file} nicht gefunden.")

    if not Path(seq_file).exists():
        raise FileNotFoundError(f"Sequenzdatei {seq_file} nicht gefunden.")
    
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
                                n_backlog=0,  # Wichtig: Signal am Beginn des ADC
                                verbose=True,
                                webhook=False)
    diff_sim = DifferentiableBMCSimWrapper(sim_engine_instance)
    
    # Extrahiere RF-Parameter aus der Sequenz
    rf_parameters_list = []
    for i, block_event in enumerate(diff_sim.sim_engine.seq.block_events, start=1):
        block = sim_engine_instance.seq.get_block(block_event)
        if block.rf:
            if hasattr(block, "block_duration") and block.block_duration != "0":
                amp_, ph_, dtp_rf, delay_after_pulse = prep_rf_simulation(
                    block, sim_engine_instance.params.options["max_pulse_samples"])
                rf_parameters_list.append(amp_)

    if rf_parameters_list:
        rf_parameters_tensor = torch.stack(rf_parameters_list)  # Form: (N, T)
        rf_parameters_tensor.requires_grad_(True)  # Mache Parameter differenzierbar
    else:
        raise ValueError("Keine RF-Blöcke in der Sequenz gefunden.")

    return rf_parameters_tensor, sim_params, seq_file, z_pos, diff_sim

def plot_progress(epochs, losses, signals, save_path=None):
    """
    Plottet den Fortschritt des Trainings.
    
    Args:
        epochs: Liste der Epochen
        losses: Liste der Verlustwerte
        signals: Liste der Signalwerte
        save_path: Optionaler Pfad zum Speichern des Plots
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot für Verlust
    ax1.plot(epochs, losses, 'b-', marker='o')
    ax1.set_ylabel('Verlust')
    ax1.set_title('Trainingsverlauf')
    ax1.grid(True)
    
    # Plot für Signal
    ax2.plot(epochs, signals, 'r-', marker='o')
    ax2.set_xlabel('Epoche')
    ax2.set_ylabel('Signal')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def train(num_epochs=50, learning_rate=0.01, checkpoint_dir='checkpoints/gauss', 
          checkpoint_frequency=5, resume_from=None):
    """
    Führt das Training für die Gaußpuls-Optimierung durch.
    
    Args:
        num_epochs: Anzahl der Trainingsiterationen
        learning_rate: Lernrate für den Optimierer
        checkpoint_dir: Verzeichnis zum Speichern der Checkpoints
        checkpoint_frequency: Häufigkeit der Checkpoint-Erstellung
        resume_from: Optionaler Pfad zu einem Checkpoint, von dem aus das Training fortgesetzt wird
    """
    # Hole Parameter und initialisiere Simulation
    rf_parameters, sim_params, seq_file, z_pos, diff_sim = get_params()
    
    # Initialisiere den Optimierer
    optimizer = torch.optim.Adam([rf_parameters], lr=learning_rate)
    
    # Variablen für Tracking
    start_epoch = 0
    best_signal = float('-inf')
    best_parameters = None
    epochs_list = []
    losses_list = []
    signals_list = []
    
    # Setze Training von einem Checkpoint fort, falls angegeben
    if resume_from is not None:
        rf_parameters, start_epoch, _, last_signal = load_checkpoint(resume_from, optimizer)
        print(f"Training wird von Epoche {start_epoch} mit Signal {last_signal:.6f} fortgesetzt.")
        
    # Trainingsschleife
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Führe die Simulation mit aktuellen Parametern durch
        signal = diff_sim(rf_parameters)
        
        # Berechne den Verlust
        loss = signal_objective_function(signal)
        
        # Führe Backpropagation durch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Signal (negierter Verlust) für die Ausgabe
        current_signal = -loss.item()
        
        # Tracking
        epochs_list.append(epoch)
        losses_list.append(loss.item())
        signals_list.append(current_signal)
        
        # Ausgabe
        print(f"Epoche {epoch}/{start_epoch + num_epochs - 1} - "
              f"Signal: {current_signal:.6f}")
        
        # Speichere den besten Zustand
        if current_signal > best_signal:
            best_signal = current_signal
            best_parameters = rf_parameters.detach().clone()
            save_checkpoint(
                checkpoint_dir, 
                best_parameters, 
                optimizer, 
                epoch, 
                loss.item(), 
                current_signal,
                checkpoint_name="best_checkpoint.pt"
            )
            print(f"  Neuer Bestwert gefunden und gespeichert!")
        
        # Speichere regelmäßige Checkpoints
        if epoch % checkpoint_frequency == 0 or epoch == start_epoch + num_epochs - 1:
            save_checkpoint(
                checkpoint_dir, 
                rf_parameters, 
                optimizer, 
                epoch, 
                loss.item(), 
                current_signal
            )
    
    # Speichere den finalen Zustand
    save_checkpoint(
        checkpoint_dir, 
        rf_parameters, 
        optimizer, 
        start_epoch + num_epochs - 1, 
        losses_list[-1], 
        signals_list[-1],
        checkpoint_name="final_checkpoint.pt"
    )
    
    # Plotte den Fortschritt
    plot_progress(
        epochs_list, 
        losses_list, 
        signals_list,
        save_path=os.path.join(checkpoint_dir, "training_progress.png")
    )
    
    print("\nTraining abgeschlossen!")
    print(f"Bestes Signal: {best_signal:.6f}")
    
    return best_parameters, best_signal

if __name__ == "__main__":
    # Führe ein einfaches Training mit 20 Epochen durch
    train(num_epochs=20, checkpoint_frequency=5)
