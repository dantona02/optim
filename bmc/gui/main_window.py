import sys
import os
from pathlib import Path
import yaml
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QFileDialog, QMessageBox, QStackedWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import numpy as np
import torch
import tempfile
import copy  # Import copy-Modul für deepcopy

from bmc.fid.engine import BMCSim
from bmc.set_params import load_params
from bmc.utils.global_device import GLOBAL_DEVICE
from bmc.gui.control_panel import ControlPanel
from bmc.gui.plot_panel import PlotPanel
from bmc.gui.config_dialog import ConfigDialog
from bmc.gui.side_navigation import SideNavigation
from bmc.gui.about_page import AboutPage
from bmc.gui.animation_control_panel import AnimationControlPanel
from bmc.gui.video_panel import VideoPanel
from bmc.gui.animation_worker import AnimationWorker


class SimulationWorker(QThread):
    """Worker thread for running BMC simulations without freezing the UI."""
    progress_updated = pyqtSignal(int, int)  # (current, total)
    status_updated = pyqtSignal(str)
    simulation_completed = pyqtSignal(object)  # Sends the sim_engine
    error_occurred = pyqtSignal(str)
    
    def __init__(self, seq_file, sim_params, sim_settings):
        super().__init__()
        self.seq_file = seq_file
        self.sim_params = sim_params
        self.sim_settings = sim_settings
        self.sim_engine = None
        
    def run(self):
        """Run the simulation in a separate thread."""
        try:
            # Prepare z-positions
            n_iso = self.sim_settings['n_iso']
            low = -1e-3
            high = 1e-3
            z_pos = np.linspace(low, high, n_iso)
            z_pos = torch.tensor(z_pos)
            z_pos = torch.cat((z_pos, torch.tensor([0.0])))
            
            # Initialize simulation engine
            self.sim_engine = BMCSim(
                adc_time=self.sim_settings['adc_time'] / 1000.0,
                params=self.sim_params,
                seq_file=self.seq_file,
                z_positions=z_pos,
                n_backlog=self.sim_settings['n_backlog'],
                verbose=False,
                webhook=False
            )
            
            # Setup progress tracking
            total_events = len(self.sim_engine.seq.block_events)
            self.progress_updated.emit(0, total_events)
            
            # Initialize magnetization
            current_adc = 1
            mag = torch.tensor(
                self.sim_engine.m_init[np.newaxis, np.newaxis, :, np.newaxis], 
                dtype=torch.float64,
                device=GLOBAL_DEVICE
            )
            
            # Run simulation with progress tracking
            for i, block_event in enumerate(self.sim_engine.seq.block_events, start=1):
                counter = np.abs(total_events - i)
                block = self.sim_engine.seq.get_block(block_event)
                current_adc, mag = self.sim_engine.run_adc(block, current_adc, mag, counter)
                
                # Update progress bar
                self.progress_updated.emit(i, total_events)
                
                # Update status text periodically
                if i % 10 == 0 or i == total_events:
                    self.status_updated.emit("Running simulation...")
                
            # Trim the magnetization data to the correct length
            self.sim_engine.m_out = self.sim_engine.m_out[:, :, :self.sim_engine.t.numel()]
            
            # Signal completion
            self.simulation_completed.emit(self.sim_engine)
            
        except Exception as e:
            # Handle errors
            self.error_occurred.emit(str(e))


class BMCSimulatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BMC Simulator")
        
        # Get the available screen geometry
        screen = QApplication.primaryScreen().availableGeometry()
        
        # Set reasonable default size
        width = int(screen.width() * 0.7)  # 70% of screen width
        height = int(screen.height() * 0.7)  # 70% of screen height
        
        # Set minimum size to maintain usability
        self.setMinimumSize(1200, 800)
        
        # Center the window on screen
        x = (screen.width() - width) // 2
        y = (screen.height() - height) // 2
        self.setGeometry(x, y, width, height)
        
        # Initialize important variables
        self.sim_engine = None
        self.current_config = None
        self.current_seq = None
        self.config_params = {}
        self.simulation_thread = None
        self.animation_thread = None
        self.temp_config_path = None
        self.latest_video_path = None
        self.simulation_counter = {}  # Track number of simulations per sequence
        
        # Load default configuration values
        self._load_default_config()
        
        # Set the modern dark theme
        self.setStyleSheet("""
            QMainWindow, QScrollArea, QWidget {
                background-color: #1E1E1E;
            }
            QWidget {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid #383838;
                border-radius: 8px;
                background-color: #2A2A2A;
                margin-top: -1px;
            }
            QTabBar::tab {
                background-color: transparent;
                color: #888888;
                padding: 8px 16px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                color: #2962FF;
                background-color: #2A2A2A;
                border: 1px solid #383838;
                border-bottom: none;
            }
            QTabBar::tab:hover:!selected {
                color: #E0E0E0;
                background-color: rgba(41, 98, 255, 0.05);
            }
        """)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Add side navigation
        self.side_nav = SideNavigation()
        self.side_nav.navigationChanged.connect(self._on_navigation_changed)
        layout.addWidget(self.side_nav)
        
        # Create stacked widget for different pages
        self.stack = QStackedWidget()
        
        # Create main simulation page
        self.main_page = QWidget()
        main_page_layout = QHBoxLayout(self.main_page)
        main_page_layout.setContentsMargins(12, 12, 12, 12)
        main_page_layout.setSpacing(12)
        
        # Create control panel for main page
        self.control_panel = ControlPanel()
        self.control_panel.setFixedWidth(420)  # Fixed width for control panel
        
        # Connect control panel signals
        self.control_panel.loadSequenceClicked.connect(self._load_sequence)
        self.control_panel.loadConfigClicked.connect(self._load_config)
        self.control_panel.editConfigClicked.connect(self._open_config_dialog)
        self.control_panel.runSimulationClicked.connect(self._run_simulation)
        self.control_panel.saveResultsClicked.connect(self._save_results)
        
        # Create plot panel for main page
        self.plot_panel = PlotPanel()
        
        # Add panels to main page layout
        main_page_layout.addWidget(self.control_panel)
        main_page_layout.addWidget(self.plot_panel)
        main_page_layout.setStretch(0, 0)  # Control panel - no stretch
        main_page_layout.setStretch(1, 1)  # Plot panel - should stretch
        
        # Create animation page
        self.animation_page = QWidget()
        animation_layout = QHBoxLayout(self.animation_page)
        animation_layout.setContentsMargins(12, 12, 12, 12)
        animation_layout.setSpacing(12)
        
        # Create animation control panel
        self.animation_control_panel = AnimationControlPanel()
        self.animation_control_panel.setFixedWidth(420)  # Fixed width
        
        # Connect animation control panel signals
        self.animation_control_panel.startAnimationClicked.connect(self._run_animation)
        self.animation_control_panel.saveVideoClicked.connect(self._save_video)
        
        # Create video panel for animation page
        self.video_panel = VideoPanel()
        
        # Add panels to animation page layout
        animation_layout.addWidget(self.animation_control_panel)
        animation_layout.addWidget(self.video_panel)
        animation_layout.setStretch(0, 0)  # Control panel - no stretch
        animation_layout.setStretch(1, 1)  # Video panel - should stretch
        
        # Create placeholders for other pages
        self.pulseseq_page = QWidget()
        pulseseq_layout = QHBoxLayout(self.pulseseq_page)
        pulseseq_layout.addWidget(QWidget())  # Placeholder
        
        self.settings_page = QWidget()
        settings_layout = QHBoxLayout(self.settings_page)
        settings_layout.addWidget(QWidget())  # Placeholder
        
        self.about_page = AboutPage()
        
        # Add pages to stack
        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.animation_page)
        self.stack.addWidget(self.pulseseq_page)
        self.stack.addWidget(self.settings_page)
        self.stack.addWidget(self.about_page)
        
        # Add stack to main layout
        layout.addWidget(self.stack)
        
        # Set default page
        self.stack.setCurrentIndex(0)  # Show main page by default

    def _on_navigation_changed(self, page_name):
        """Handle navigation changes"""
        page_index = {
            "simulation": 0,
            "animation": 1,
            "pulseseq": 2,
            "settings": 3,
            "about": 4
        }.get(page_name, 0)
        
        # Setze den korrekten Index und stelle sicher, dass der Button aktiv bleibt
        self.stack.setCurrentIndex(page_index)
        self.side_nav.setCurrentPage(page_name)
        
        # Update animation panel with current sequence if available
        if page_name == "animation" and self.sim_engine is not None and self.current_seq:
            self.animation_control_panel.set_sequence_info(self.current_seq)
        
    def _load_default_config(self):
        """Load default values for configuration"""
        self.config_params = {
            "water_pool": {
                "f": 1.0,
                "t1": 2.5,
                "t2": 0.071
            },
            "cest_pool": {
                "amide": {
                    "f": 0.00064865,
                    "t1": 1.3,
                    "t2": 0.1,
                    "k": 50,
                    "dw": 8
                }
            },
            "b0": 17,
            "gamma": 267.5153,
            "b0_inhom": 0.0,
            "rel_b1": 1,
            "verbose": False,
            "reset_init_mag": True,
            "scale": 1,
            "max_pulse_samples": 1500
        }

    def _open_config_dialog(self):
        """Opens the configuration dialog"""
        # Erstelle eine tiefe Kopie der Konfigurationsparameter, damit Änderungen
        # vollständig übernommen werden können, einschließlich verschachtelter Dictionaries
        dialog = ConfigDialog(copy.deepcopy(self.config_params), self)
        if dialog.exec():
            # Übernimm die vollständige neue Konfiguration
            self.config_params = dialog.get_config()
            # Zeige an, dass eine dynamische Konfiguration verwendet wird
            self.control_panel.set_config_loaded("Dynamic configuration")
            
    def _load_sequence(self, filename=None):
        if not filename:
            return
            
        self.current_seq = filename
        self._check_start_enabled()
        
        # Update animation control panel if we're on the animation page
        if self.stack.currentIndex() == 1 and self.sim_engine is not None:
            self.animation_control_panel.set_sequence_info(filename)
            
    def _load_config(self, filename=None):
        if not filename:
            return
            
        try:
            with open(filename, 'r') as f:
                self.config_params = yaml.safe_load(f)
            
            self.current_config = filename
            self._check_start_enabled()
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Config", f"Error loading configuration: {str(e)}")
            
    def _check_start_enabled(self):
        # Enable the start button when we have a sequence
        # Config can be either loaded externally or created via GUI
        self.control_panel.start_btn.setEnabled(self.current_seq is not None)
            
    def _run_simulation(self):
        if not self.current_seq:
            QMessageBox.warning(self, "Missing Data", "Please load a sequence file.")
            return
        
        # Stop any ongoing simulation
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.terminate()
            self.simulation_thread.wait()
            
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(self.config_params, tmp, default_flow_style=False)
            self.temp_config_path = tmp.name
        
        # Load and validate parameters
        try:
            sim_params = load_params(self.temp_config_path)
        except (AssertionError, AttributeError) as e:
            QMessageBox.critical(self, "Invalid Parameters", f"Error in parameters: {str(e)}")
            self._cleanup_temp_file()
            return
        
        # Get simulation parameters from control panel
        sim_settings = self.control_panel.get_simulation_parameters()
        
        # Update GUI status before simulation
        self.control_panel.enable_controls(False)
        self.control_panel.set_status("Starting simulation...")
        
        # Create simulation thread
        self.simulation_thread = SimulationWorker(
            seq_file=self.current_seq,
            sim_params=sim_params,
            sim_settings=sim_settings
        )
        
        # Connect signals from worker
        self.simulation_thread.progress_updated.connect(self.control_panel.update_progress)
        self.simulation_thread.status_updated.connect(self.control_panel.set_status)
        self.simulation_thread.simulation_completed.connect(self._simulation_finished)
        self.simulation_thread.error_occurred.connect(self._simulation_error)
        
        # Start the thread
        self.simulation_thread.start()
    
    @pyqtSlot(object)
    def _simulation_finished(self, sim_engine):
        """Handle successful simulation completion."""
        # Store the sim_engine
        self.sim_engine = sim_engine
        
        # Get base name from sequence file
        base_name = Path(self.current_seq).stem
        
        # Update simulation counter for this sequence
        if base_name not in self.simulation_counter:
            self.simulation_counter[base_name] = 0
        self.simulation_counter[base_name] += 1
        
        # Generate dataset name from sequence file and config
        dataset_name = base_name
        if self.simulation_counter[base_name] > 1:
            dataset_name += f"_{self.simulation_counter[base_name]}"
            
        # Add config parameters to name if available
        if self.config_params:
            config_suffix = []
            if 'cest' in self.config_params:
                if 'offset' in self.config_params['cest']:
                    config_suffix.append(f"off_{self.config_params['cest']['offset']}")
                if 'exchange_rate' in self.config_params['cest']:
                    config_suffix.append(f"k_{self.config_params['cest']['exchange_rate']}")
            if config_suffix:
                dataset_name += "_" + "_".join(config_suffix)
        
        # Update plots with dataset name
        self.plot_panel.plot_results(self.sim_engine, dataset_name)
        
        # Update UI status
        self.control_panel.set_status("Simulation completed successfully", is_success=True)
        self.control_panel.enable_save_button(True)
        self.control_panel.enable_controls(True)
        
        # Update animation control panel with sequence info
        if self.current_seq:
            self.animation_control_panel.set_sequence_info(self.current_seq)
        
        # Clean up
        self._cleanup_temp_file()
    
    @pyqtSlot(str)
    def _simulation_error(self, error_message):
        """Handle simulation errors."""
        self.control_panel.set_status(f"Error: {error_message[:100]}...", is_error=True)
        self.control_panel.enable_controls(True)
        print(f"Simulation error: {error_message}")
        
        # Clean up
        self._cleanup_temp_file()
    
    def _cleanup_temp_file(self):
        """Clean up temporary configuration file."""
        if self.temp_config_path and os.path.exists(self.temp_config_path):
            try:
                os.unlink(self.temp_config_path)
                self.temp_config_path = None
            except Exception as e:
                print(f"Could not delete temporary file: {e}")

    def _run_animation(self, animation_params):
        """Run animation with the specified parameters."""
        if not self.sim_engine:
            QMessageBox.warning(self, "No Simulation Data", "Please run a simulation first.")
            return
        
        # Stop any ongoing animation
        if self.animation_thread and self.animation_thread.isRunning():
            self.animation_thread.terminate()
            self.animation_thread.wait()
        
        # Update UI status before animation
        self.animation_control_panel.enable_controls(False)
        self.animation_control_panel.set_status("Starting animation...")
        
        # Create animation thread
        self.animation_thread = AnimationWorker(
            sim_engine=self.sim_engine,
            animation_params=animation_params
        )
        
        # Connect signals from worker
        self.animation_thread.progress_updated.connect(self.animation_control_panel.update_progress)
        self.animation_thread.status_updated.connect(self.animation_control_panel.set_status)
        self.animation_thread.animation_completed.connect(self._animation_finished)
        self.animation_thread.error_occurred.connect(self._animation_error)
        
        # Start the thread
        self.animation_thread.start()
    
    @pyqtSlot(str)
    def _animation_finished(self, video_path):
        """Handle successful animation completion."""
        # Store the video path
        self.latest_video_path = video_path
        
        # Update UI status
        self.animation_control_panel.set_status("Animation completed successfully", is_success=True)
        self.animation_control_panel.enable_controls(True)
        self.animation_control_panel.enable_save_button(True)
        
        # Set the video in the video panel
        self.video_panel.set_video(video_path)
    
    @pyqtSlot(str)
    def _animation_error(self, error_message):
        """Handle animation errors."""
        self.animation_control_panel.set_status(f"Error: {error_message[:100]}...", is_error=True)
        self.animation_control_panel.enable_controls(True)
        print(f"Animation error: {error_message}")
    
    def _save_video(self):
        """Save the current animation video."""
        if self.latest_video_path and os.path.exists(self.latest_video_path):
            self.video_panel.download_video()
        else:
            QMessageBox.warning(self, "No Video Available", "No animation video available to save.")
        
    def _save_results(self):
        """Save simulation results to a file"""
        if not self.sim_engine:
            return
            
        # Get magnetization data
        t, m_z, m_z_total, m_trans, m_trans_total = self.sim_engine.get_mag()
        
        # Save the data
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "",
            "NPZ Files (*.npz)"
        )
        
        if filename:
            np.savez(
                filename,
                time=t.cpu().numpy(),
                m_z=m_z.cpu().numpy(),
                m_z_total=m_z_total.cpu().numpy(),
                m_trans=m_trans.cpu().numpy(),
                m_trans_total=m_trans_total.cpu().numpy()
            )
            
    def closeEvent(self, event):
        """Handle window close event to clean up resources."""
        # Stop any ongoing simulation
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.terminate()
            self.simulation_thread.wait()
        
        # Stop any ongoing animation
        if self.animation_thread and self.animation_thread.isRunning():
            self.animation_thread.terminate()
            self.animation_thread.wait()
        
        # Clean up temporary files
        self._cleanup_temp_file()
        
        # Accept the event to close the window
        event.accept()
    
    def reset_simulation_counter(self, base_name):
        """Reset the simulation counter for a specific sequence"""
        if base_name in self.simulation_counter:
            # Find remaining datasets with this base name
            remaining_count = 0
            for dataset in self.plot_panel.dataset_panel.datasets.keys():
                if dataset.startswith(base_name):
                    remaining_count += 1
            
            if remaining_count == 0:
                # If no datasets with this base name remain, reset the counter
                del self.simulation_counter[base_name]
            else:
                # Update counter to match the number of remaining datasets
                self.simulation_counter[base_name] = remaining_count

