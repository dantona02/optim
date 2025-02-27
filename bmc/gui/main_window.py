import sys
import os
from pathlib import Path
import yaml
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QFileDialog, QMessageBox, QStackedWidget
)
from PyQt6.QtCore import Qt
import numpy as np
import torch
import tempfile

from bmc.fid.engine import BMCSim
from bmc.set_params import load_params
from bmc.utils.global_device import GLOBAL_DEVICE
from bmc.gui.control_panel import ControlPanel
from bmc.gui.plot_panel import PlotPanel
from bmc.gui.config_dialog import ConfigDialog
from bmc.gui.side_navigation import SideNavigation
from bmc.gui.about_page import AboutPage

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
        self.setMinimumSize(900, 600)
        
        # Center the window on screen
        x = (screen.width() - width) // 2
        y = (screen.height() - height) // 2
        self.setGeometry(x, y, width, height)
        
        # Initialize important variables
        self.sim_engine = None
        self.current_config = None
        self.current_seq = None
        self.config_params = {}
        
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
        
        # Create other pages
        self.pulseseq_page = QWidget()
        pulseseq_layout = QHBoxLayout(self.pulseseq_page)
        pulseseq_layout.addWidget(QWidget())  # Placeholder
        
        self.animation_page = QWidget()
        animation_layout = QHBoxLayout(self.animation_page)
        animation_layout.addWidget(QWidget())  # Placeholder
        
        self.settings_page = QWidget()
        settings_layout = QHBoxLayout(self.settings_page)
        settings_layout.addWidget(QWidget())  # Placeholder
        
        self.about_page = AboutPage()
        
        # Add pages to stack
        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.pulseseq_page)
        self.stack.addWidget(self.animation_page)
        self.stack.addWidget(self.settings_page)
        self.stack.addWidget(self.about_page)
        
        # Add stack to main layout
        layout.addWidget(self.stack)
        
        # Set default page
        self.stack.setCurrentIndex(0)  # Show main page by default

    def _on_navigation_changed(self, page_name):
        """Handle navigation changes"""
        page_index = {
            "simulation": 0,  # Geändert von "main" zu "simulation"
            "pulseseq": 1,
            "animation": 2,
            "settings": 3,
            "about": 4
        }.get(page_name, 0)
        
        # Setze den korrekten Index und stelle sicher, dass der Button aktiv bleibt
        self.stack.setCurrentIndex(page_index)
        self.side_nav.setCurrentPage(page_name)
        
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
        dialog = ConfigDialog(self.config_params, self)
        if dialog.exec():
            # Update configuration if OK was clicked
            self.config_params = dialog.get_config()
            # Show that dynamic configuration is being used
            self.control_panel.set_config_loaded("Dynamic configuration")
            
    def _load_sequence(self, filename=None):
        if not filename:
            return
            
        self.current_seq = filename
        self._check_start_enabled()
            
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
            
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(self.config_params, tmp, default_flow_style=False)
            temp_config_path = tmp.name
        
        # Load and validate parameters
        try:
            sim_params = load_params(temp_config_path)
        except (AssertionError, AttributeError) as e:
            QMessageBox.critical(self, "Invalid Parameters", f"Error in parameters: {str(e)}")
            return
        
        # Get simulation parameters from control panel
        sim_settings = self.control_panel.get_simulation_parameters()
        
        # Prepare z-positions
        n_iso = sim_settings['n_iso']
        low = -1e-3
        high = 1e-3
        z_pos = np.linspace(low, high, n_iso)
        z_pos = torch.tensor(z_pos)
        z_pos = torch.cat((z_pos, torch.tensor([0.0])))
        
        # Update GUI status before simulation
        self.control_panel.enable_controls(False)
        self.control_panel.set_status("Simulation running...")
        
        try:
            # Initialize simulation engine
            self.sim_engine = BMCSim(
                adc_time=sim_settings['adc_time'] / 1000.0,
                params=sim_params,
                seq_file=self.current_seq,
                z_positions=z_pos,
                n_backlog=sim_settings['n_backlog'],
                verbose=False,
                webhook=False
            )
            
            # Setup progress tracking
            total_events = len(self.sim_engine.seq.block_events)
            self.control_panel.update_progress(0, total_events)
            
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
                self.control_panel.update_progress(i)
                
                # Update status text periodically
                if i % 10 == 0 or i == total_events:
                    self.control_panel.set_status("Running simulation...")
                
            # Trim the magnetization data to the correct length
            self.sim_engine.m_out = self.sim_engine.m_out[:, :, :self.sim_engine.t.numel()]
            
            # Update plots and GUI status
            self.plot_panel.plot_results(self.sim_engine)
            
            self.control_panel.set_status("Simulation completed successfully", is_success=True)
            self.control_panel.enable_save_button(True)
            
        except Exception as e:
            # Handle errors
            error_msg = str(e)
            self.control_panel.set_status(f"Error: {error_msg[:100]}...", is_error=True)
            print(f"Simulation error: {error_msg}")
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_config_path)
            except:
                pass
                
            # Re-enable controls
            self.control_panel.enable_controls(True)
        
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

