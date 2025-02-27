from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QSpinBox, QDoubleSpinBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from pathlib import Path

from bmc.gui.components import TitledGroupBox, AnimatedProgressBar


class ControlPanel(QWidget):
    """A class representing the control panel in the BMC Simulator GUI."""
    
    # Define signals for communicating with the main window
    loadSequenceClicked = pyqtSignal(str)
    loadConfigClicked = pyqtSignal(str)
    editConfigClicked = pyqtSignal()
    runSimulationClicked = pyqtSignal()
    saveResultsClicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.seq_path = None
        self.config_path = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface for the control panel."""
        # Main layout for the control panel
        control_layout = QVBoxLayout(self)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(0)
        
        # Scroll area for the control panel
        control_scroll = QScrollArea()
        control_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1E1E1E;
            }
            QScrollBar:vertical {
                border: none;
                background: #2A2A2A;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #3E3E3E;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                border: none;
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_layout.addWidget(control_scroll)
        
        # Container for actual control widgets
        control_panel = QWidget()
        control_panel.setObjectName("controlPanel")
        control_panel.setStyleSheet("#controlPanel { background-color: #1E1E1E; }")
        control_scroll.setWidget(control_panel)
        
        control_panel_layout = QVBoxLayout(control_panel)
        control_panel_layout.setSpacing(16)
        control_panel_layout.setContentsMargins(10, 0, 10, 20)
        
        # Add the different control sections
        self._setup_file_selection(control_panel_layout)
        self._setup_parameters(control_panel_layout)
        self._setup_configuration(control_panel_layout)
        self._setup_simulation_control(control_panel_layout)
        self._setup_export(control_panel_layout)
        
        # Add stretch at the end
        control_panel_layout.addStretch()
        
    def _setup_file_selection(self, layout):
        """Set up the file selection group."""
        file_group = TitledGroupBox("File Selection")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(12)
        
        # Sequence file loading
        seq_group = QWidget()
        seq_layout = QHBoxLayout(seq_group)
        seq_layout.setSpacing(12)
        self.seq_label = QLabel("No sequence loaded")
        self.seq_label.setMinimumWidth(200)
        self.load_seq_btn = QPushButton("Load Sequence")
        self.load_seq_btn.setFixedWidth(70)
        # Override the global stylesheet for this button
        self.load_seq_btn.setStyleSheet("""
            QPushButton {
                min-width: 100px;
                padding: 6px 10px;
            }
        """)
        self.load_seq_btn.clicked.connect(self._on_load_sequence)
        seq_layout.addWidget(self.seq_label)
        seq_layout.addWidget(self.load_seq_btn)
        file_layout.addWidget(seq_group)
        
        # Config file loading
        config_group = QWidget()
        config_layout = QHBoxLayout(config_group)
        config_layout.setSpacing(12)
        self.config_label = QLabel("No config loaded")
        self.config_label.setMinimumWidth(200)
        self.load_config_btn = QPushButton("Load Config")
        self.load_config_btn.setFixedWidth(70)
        # Override the global stylesheet for this button
        self.load_config_btn.setStyleSheet("""
            QPushButton {
                min-width: 100px;
                padding: 6px 10px;
            }
        """)
        self.load_config_btn.clicked.connect(self._on_load_config)
        config_layout.addWidget(self.config_label)
        config_layout.addWidget(self.load_config_btn)
        file_layout.addWidget(config_group)
        
        file_group.addLayout(file_layout)
        layout.addWidget(file_group)
    
    def _setup_parameters(self, layout):
        """Set up the parameters group."""
        param_group = TitledGroupBox("Parameters")
        param_layout = QVBoxLayout()
        param_layout.setSpacing(16)
        
        # ADC Time
        adc_layout = QHBoxLayout()
        adc_layout.setSpacing(12)
        adc_label = QLabel("ADC Time [ms]:")
        adc_label.setFixedWidth(150)
        self.adc_time = QDoubleSpinBox()
        self.adc_time.setRange(0.1, 100.0)
        self.adc_time.setValue(5.0)
        self.adc_time.setSingleStep(0.1)
        adc_layout.addWidget(adc_label)
        adc_layout.addWidget(self.adc_time)
        param_layout.addLayout(adc_layout)
        
        # Isochromates
        iso_layout = QHBoxLayout()
        iso_layout.setSpacing(12)
        iso_label = QLabel("Isochromates:")
        iso_label.setFixedWidth(150)
        self.n_iso = QSpinBox()
        self.n_iso.setRange(10, 1000)
        self.n_iso.setValue(100)
        iso_layout.addWidget(iso_label)
        iso_layout.addWidget(self.n_iso)
        param_layout.addLayout(iso_layout)

        # Backlog
        backlog_layout = QHBoxLayout()
        backlog_layout.setSpacing(12)
        backlog_label = QLabel("Backlog:")
        backlog_label.setFixedWidth(150)
        self.n_backlog = QSpinBox()
        self.n_backlog.setRange(0, 100)
        self.n_backlog.setValue(1)
        backlog_layout.addWidget(backlog_label)
        backlog_layout.addWidget(self.n_backlog)
        param_layout.addLayout(backlog_layout)
        
        param_group.addLayout(param_layout)
        layout.addWidget(param_group)
    
    def _setup_configuration(self, layout):
        """Set up the configuration group."""
        config_group = TitledGroupBox("Configuration")
        config_layout = QVBoxLayout()
        
        # Button to open configuration dialog
        config_btn = QPushButton("Edit Configuration")
        config_btn.clicked.connect(self._on_edit_config)
        config_layout.addWidget(config_btn)
        
        config_group.addLayout(config_layout)
        layout.addWidget(config_group)
    
    def _setup_simulation_control(self, layout):
        """Set up the simulation control group."""
        control_group = TitledGroupBox("Control")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(24)
        
        # Start Button
        self.start_btn = QPushButton("Start Simulation")
        self.start_btn.clicked.connect(self._on_run_simulation)
        self.start_btn.setEnabled(False)
        self.start_btn.setFixedHeight(40)
        control_layout.addWidget(self.start_btn)
        
        # Progress Bar
        self.progress = AnimatedProgressBar()
        self.progress.setFixedHeight(28)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v/%m (%p%)")
        self.progress.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #242424;
                min-height: 28px;
                max-height: 28px;
                border-radius: 14px;
                margin: 8px 0px;
                text-align: center;
                color: white;
                font-weight: 600;
                font-size: 13px;
                border: 2px solid #333333;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #0AC18E, stop:0.3 #22E09A, stop:1 #4ADE80);
                border-radius: 12px;
            }
        """)
        control_layout.addWidget(self.progress)
        
        # Status Label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #E0E0E0;
                padding: 10px;
                font-weight: 600;
                margin-top: 8px;
                font-size: 13px;
                background-color: transparent;
            }
        """)
        self.status_label.setMinimumHeight(40)
        control_layout.addWidget(self.status_label)
        
        control_group.addLayout(control_layout)
        layout.addWidget(control_group)
    
    def _setup_export(self, layout):
        """Set up the export group."""
        export_group = TitledGroupBox("Export")
        export_layout = QVBoxLayout()
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self._on_save_results)
        self.save_btn.setEnabled(False)
        export_layout.addWidget(self.save_btn)
        export_group.addLayout(export_layout)
        layout.addWidget(export_group)
    
    def _on_load_sequence(self):
        """Handle the load sequence button click."""
        seq_lib_path = str(Path(__file__).resolve().parent.parent.parent / "seq_lib")
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Sequence File",
            seq_lib_path,
            "Sequence Files (*.seq)"
        )
        if filename:
            self.seq_path = filename
            self.seq_label.setText(Path(filename).name)
            self._check_start_enabled()
            self.loadSequenceClicked.emit(filename)
    
    def _on_load_config(self):
        """Handle the load config button click."""
        sim_lib_path = str(Path(__file__).resolve().parent.parent.parent / "sim_lib")
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Config File",
            sim_lib_path,
            "YAML Files (*.yaml)"
        )
        if filename:
            self.config_path = filename
            self.config_label.setText(Path(filename).name)
            self._check_start_enabled()
            self.loadConfigClicked.emit(filename)
    
    def _on_edit_config(self):
        """Handle the edit config button click."""
        self.editConfigClicked.emit()
    
    def _on_run_simulation(self):
        """Handle the run simulation button click."""
        self.runSimulationClicked.emit()
    
    def _on_save_results(self):
        """Handle the save results button click."""
        self.saveResultsClicked.emit()
    
    def _check_start_enabled(self):
        """Check if the start button should be enabled."""
        self.start_btn.setEnabled(self.seq_path is not None)
    
    def set_config_loaded(self, config_name):
        """Set the config label when loaded from external source."""
        self.config_label.setText(config_name)
    
    def update_progress(self, value, total=None):
        """Update the progress bar."""
        self.progress.update_progress(value, total)
    
    def set_status(self, text, is_error=False, is_success=False):
        """Set the status label text and style."""
        self.status_label.setText(text)
        
        if is_error:
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #f44336; 
                    font-weight: 600; 
                    padding: 5px;
                }
            """)
        elif is_success:
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #4caf50; 
                    font-weight: 600; 
                    padding-top: 0px;
                    padding-bottom: 20px;
                    padding-left: 5px;
                    padding-right: 5px;
                }
            """)
        else:
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #42a5f5; 
                    font-weight: 600; 
                    padding-top: 0px;
                    padding-bottom: 20px;
                    padding-left: 5px;
                    padding-right: 5px;
                }
            """)
    
    def enable_controls(self, enabled=True):
        """Enable or disable control elements during simulation."""
        self.start_btn.setEnabled(enabled and self.seq_path is not None)
        self.load_seq_btn.setEnabled(enabled)
        self.load_config_btn.setEnabled(enabled)
    
    def enable_save_button(self, enabled=True):
        """Enable or disable the save button."""
        self.save_btn.setEnabled(enabled)
    
    def get_simulation_parameters(self):
        """Get the current simulation parameters."""
        return {
            'adc_time': self.adc_time.value(),
            'n_iso': self.n_iso.value(),
            'n_backlog': self.n_backlog.value()
        }