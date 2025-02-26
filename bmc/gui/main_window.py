import sys
import os
from pathlib import Path
import yaml
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QSpinBox,
    QDoubleSpinBox, QProgressBar, QTabWidget, QGroupBox, QStyleFactory,
    QFrame, QCheckBox, QMessageBox, QInputDialog
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPalette, QColor, QFont, QIcon
import matplotlib
matplotlib.use('QtAgg')  # Verwende QtAgg für PyQt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import torch
import tempfile

from bmc.fid.engine import BMCSim
from bmc.set_params import load_params
from bmc.utils.global_device import GLOBAL_DEVICE

STYLE_SHEET = """
QMainWindow {
    background-color: #2b2b2b;
}

QGroupBox {
    background-color: #333333;
    border: 2px solid #444444;
    border-radius: 5px;
    margin-top: 15px;  /* Reduzierter oberer Rand */
    padding: 15px;
    padding-top: 20px;  /* Mehr Platz für die Überschrift */
    color: white;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 7px;
    padding: 0 5px;
    color: white;
    font-weight: bold;
    background-color: #2b2b2b;  /* Gleiche Farbe wie MainWindow-Hintergrund */
    position: absolute;
    top: -12px;  /* Position nach oben angepasst */
    z-index: 1;  /* Bringt die Überschrift nach vorne */
}

QPushButton {
    background-color: #0d47a1;
    color: white;
    border: none;
    padding: 8px 15px;
    border-radius: 3px;
    font-weight: bold;
    min-width: 120px;
}

QPushButton:hover {
    background-color: #1565c0;
}

QPushButton:pressed {
    background-color: #0a3d91;
}

QPushButton:disabled {
    background-color: #666666;
    color: #999999;
}

QLabel {
    color: #ffffff;
    padding: 2px;
}

QProgressBar {
    border: 2px solid #444444;
    border-radius: 5px;
    text-align: center;
    color: white;
    background-color: #333333;
    min-height: 25px;  /* Höhere Progressbar */
    margin-bottom: 10px;  /* Abstand zum Status-Label */
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 #0d47a1, stop:1 #42a5f5);
    border-radius: 3px;
}

QSpinBox, QDoubleSpinBox {
    background-color: #424242;
    color: white;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 5px;
    min-width: 80px;
    min-height: 25px;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: #555555;
    border: 1px solid #666666;
    border-top-right-radius: 3px;
    width: 20px;
    height: 12px;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #555555;
    border: 1px solid #666666;
    border-bottom-right-radius: 3px;
    width: 20px;
    height: 12px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #666666;
}

QSpinBox::up-arrow {
    width: 0;
    height: 0;
    border-style: solid;
    border-width: 0 4px 4px 4px;
    border-color: transparent transparent white transparent;
}

QSpinBox::down-arrow {
    width: 0;
    height: 0;
    border-style: solid;
    border-width: 4px 4px 0 4px;
    border-color: white transparent transparent transparent;
}

QDoubleSpinBox::up-arrow {
    width: 0;
    height: 0;
    border-style: solid;
    border-width: 0 4px 4px 4px;
    border-color: transparent transparent white transparent;
}

QDoubleSpinBox::down-arrow {
    width: 0;
    height: 0;
    border-style: solid;
    border-width: 4px 4px 0 4px;
    border-color: white transparent transparent transparent;
}

QSpinBox::up-arrow, QSpinBox::down-arrow,
QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {
    border-style: solid;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    border-bottom: 6px solid white;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    border-top: 6px solid white;
}

QTabWidget::pane {
    border: 2px solid #444444;
    border-radius: 5px;
    background-color: #333333;
}

QTabBar::tab {
    background-color: #424242;
    color: white;
    padding: 8px 12px;
    margin-right: 2px;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
}

QTabBar::tab:selected {
    background-color: #0d47a1;
}

QTabBar::tab:hover {
    background-color: #1565c0;
}
"""

class AnimatedProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #2b2b2b;
                min-height: 30px;
                text-align: center;
                color: white;
                font-weight: bold;
                font-size: 13px;
                margin: 0px;
            }
            
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                stop:0 #1565C0, stop:1 #42a5f5);
                border-radius: 0px;
            }
        """)
        self._current = 0
        self._total = 100
        self.setFormat("%v/%m (%p%)")

    def update_progress(self, n, total=None):
        if total is not None:
            self._total = total
            self.setMaximum(total)
        self._current = n
        self.setValue(n)
        QApplication.processEvents()  # Aktualisiere die GUI

class TitledGroupBox(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        
        # Get absolute paths for the arrow images
        current_dir = Path(__file__).resolve().parent
        up_arrow_path = current_dir / 'images' / 'upload.png'
        down_arrow_path = current_dir / 'images' / 'download.png'
        
        # Convert to URL format that Qt understands
        up_arrow_url = f"{up_arrow_path.as_posix()}"
        down_arrow_url = f"{down_arrow_path.as_posix()}"
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 22, 0, 0)  # More space for the modern floating title
        layout.setSpacing(8)

        # Modern floating title label
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #E0E0E0;
                font-weight: 600;
                font-size: 13px;
                padding: 0px 12px;
                background-color: #1E1E1E;
                border: none;
                letter-spacing: 0.3px;
            }
        """)
        title_container = QWidget()
        title_container.setStyleSheet("background: transparent; border: none;")
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(16, 0, 0, 0)
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        
        # Modern box with subtle gradient and shadow effect
        self.box = QWidget()
        self.box.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #2A2A2A, stop:1 #2D2D2D);
                border: 1px solid #383838;
                border-radius: 8px;
            }}
            
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #2962FF, stop:1 #2979FF);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 13px;
                min-width: 120px;
            }}
            
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #2979FF, stop:1 #448AFF);
            }}
            
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #2962FF, stop:1 #2962FF);
            }}
            
            QPushButton:disabled {{
                background: #404040;
                color: #888888;
            }}
            
            QLabel {{
                color: #E0E0E0;
                padding: 2px;
                border: none;
                font-size: 13px;
            }}
            
            QProgressBar {{
                border: none;
                background-color: #2A2A2A;
                min-height: 6px;
                max-height: 6px;
                border-radius: 3px;
                margin: 12px 0px;
            }}
            
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #2962FF, stop:1 #448AFF);
                border-radius: 3px;
            }}
            
            QSpinBox, QDoubleSpinBox {{
                background-color: #2A2A2A;
                color: #E0E0E0;
                border: 1px solid #383838;
                border-radius: 6px;
                padding: 5px 8px;
                min-width: 100px;
                font-size: 13px;
            }}
            
            QSpinBox:hover, QDoubleSpinBox:hover {{
                border: 1px solid #424242;
                background-color: #2D2D2D;
            }}
            
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid #2962FF;
                background-color: #2D2D2D;
            }}
            
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 22px;
                height: 12px;
                border: none;
                background-color: transparent;
                margin: 1px 1px 0px 0px;
            }}
            
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 22px;
                height: 12px;
                border: none;
                background-color: transparent;
                margin: 0px 1px 1px 0px;
            }}
            
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: rgba(41, 98, 255, 0.1);
            }}
            
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                image: url("{up_arrow_url}");
                width: 10px;
                height: 10px;
            }}
            
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                image: url("{down_arrow_url}");
                width: 10px;
                height: 10px;
            }}
            
            QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
            QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
                image: none;
            }}
        """)
        
        self.box_layout = QVBoxLayout(self.box)
        self.box_layout.setContentsMargins(16, 16, 16, 16)
        self.box_layout.setSpacing(12)

        layout.addWidget(title_container)
        layout.addWidget(self.box)

    def addWidget(self, widget):
        self.box_layout.addWidget(widget)

    def addLayout(self, layout):
        self.box_layout.addLayout(layout)

class BMCSimulatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BMC Simulator")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialisierung der wichtigen Variablen
        self.sim_engine = None
        self.current_config = None
        self.current_seq = None
        self.config_params = {}  # Initialisiere config_params
        
        # Default-Konfigurationswerte laden
        self._load_default_config()
        
        # Set the modern dark theme
        self.setStyleSheet("""
            QMainWindow {
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
        
        # Hauptwidget und Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Linkes Panel für Kontrollelemente
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(400)  # Kontrollpanel breiter gemacht
        layout.addWidget(control_panel)
        
        # Rechtes Panel für Plots
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        layout.addWidget(plot_panel)
        
        # Setup der Kontrollelemente
        self._setup_controls(control_layout)
        
        # Setup der Plot-Bereiche
        self._setup_plots(plot_layout)
        
        # Initialisierung der Simulation
        self.sim_engine = None
        self.current_config = None
        self.current_seq = None
        self.config_params = {}  # Speichern der aktuellen Konfigurationsparameter
        
        # Default-Konfigurationswerte laden
        self._load_default_config()

    def _load_default_config(self):
        """Lädt Standardwerte für die Konfiguration"""
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

    def _setup_controls(self, layout):
        # File Selection Group
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
        self.load_seq_btn.setFixedWidth(150)
        self.load_seq_btn.clicked.connect(self._load_sequence)
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
        self.load_config_btn.setFixedWidth(150)
        self.load_config_btn.clicked.connect(self._load_config)
        config_layout.addWidget(self.config_label)
        config_layout.addWidget(self.load_config_btn)
        file_layout.addWidget(config_group)
        
        file_group.addLayout(file_layout)
        layout.addWidget(file_group)
        
        # Parameters Group
        param_group = TitledGroupBox("Parameters")
        param_layout = QVBoxLayout()
        param_layout.setSpacing(16)
        
        # ADC Time
        adc_layout = QHBoxLayout()
        adc_layout.setSpacing(12)
        adc_label = QLabel("ADC Time (ms):")
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
        
        # Zusätzlicher Tab für komplette Konfiguration
        config_tab = QTabWidget()
        config_tab.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #383838;
                border-radius: 6px;
                padding: 5px;
            }
            QTabBar::tab {
                background-color: #2D2D2D;
                color: #E0E0E0;
                padding: 8px 16px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #383838;
                color: #2962FF;
            }
            QTabBar::tab:hover:!selected {
                background-color: #333333;
                color: #FFFFFF;
            }
        """)
        
        # Tab für Wasserpool-Params
        water_tab = self._create_water_pool_tab()
        config_tab.addTab(water_tab, "Wasserpool")
        
        # Tab für CEST-Pool-Params
        cest_tab = self._create_cest_pool_tab()
        config_tab.addTab(cest_tab, "CEST-Pool")
        
        # Tab für Scanner-Einstellungen
        scanner_tab = self._create_scanner_tab()
        config_tab.addTab(scanner_tab, "Scanner")
        
        # Tab für erweiterte Einstellungen
        advanced_tab = self._create_advanced_tab()
        config_tab.addTab(advanced_tab, "Erweitert")
        
        # Configuration Group für den Tab
        config_group = TitledGroupBox("Konfiguration")
        config_layout = QVBoxLayout()
        config_layout.addWidget(config_tab)
        
        # Buttons für die Konfiguration
        config_btn_layout = QHBoxLayout()
        
        # Button zum Laden einer vorhandenen Konfiguration
        self.load_existing_config_btn = QPushButton("Konfig laden")
        self.load_existing_config_btn.setFixedWidth(120)
        self.load_existing_config_btn.clicked.connect(self._load_config)
        config_btn_layout.addWidget(self.load_existing_config_btn)
        
        # Button zum Speichern der aktuellen Konfiguration
        self.save_config_btn = QPushButton("Konfig speichern")
        self.save_config_btn.setFixedWidth(120)
        self.save_config_btn.clicked.connect(self._save_config)
        config_btn_layout.addWidget(self.save_config_btn)
        
        # Button zum Zurücksetzen auf Standardwerte
        self.reset_config_btn = QPushButton("Zurücksetzen")
        self.reset_config_btn.setFixedWidth(120)
        self.reset_config_btn.clicked.connect(self._reset_config)
        config_btn_layout.addWidget(self.reset_config_btn)
        
        config_layout.addLayout(config_btn_layout)
        config_group.addLayout(config_layout)
        layout.addWidget(config_group)
        
        # Simulation Control Group
        control_group = TitledGroupBox("Control")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(24)  # Erhöht den Abstand zwischen Elementen
        
        # Start Button
        self.start_btn = QPushButton("Start Simulation")
        self.start_btn.clicked.connect(self._run_simulation)
        self.start_btn.setEnabled(False)
        self.start_btn.setFixedHeight(40)  # Höherer Button für bessere Sichtbarkeit
        control_layout.addWidget(self.start_btn)
        
        # Progress Widget with Status - mehr Platz ohne Rahmen
        progress_widget = QWidget()
        progress_widget.setStyleSheet("background: transparent; border: none;") # Entfernt den Rahmen
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setSpacing(16)  # Mehr Abstand zwischen Progressbar und Label
        progress_layout.setContentsMargins(0, 8, 0, 8)  # Mehr Platz oben und unten
        
        # Verbesserte Progress Bar - mit abgestimmtem Hintergrund und sanft abgerundeten Ecken
        self.progress = AnimatedProgressBar()
        self.progress.setFixedHeight(25)  # Dickere Progress Bar
        self.progress.setTextVisible(True)  # Text auf der Progressbar anzeigen
        self.progress.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #1E1E1E;
                min-height: 25px;
                text-align: center;
                color: white;
                font-weight: bold;
                font-size: 13px;
                margin: 0px;
                border-radius: 8px;
                padding: 0px;
            }
            
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                              stop:0 #2E7D32, stop:0.5 #43A047, stop:1 #66BB6A);
                border-radius: 8px;
                margin: 0px;
            }
        """)
        progress_layout.addWidget(self.progress)
        
        # Status Label unter Progress Bar mit mehr Abstand
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
        self.status_label.setMinimumHeight(40)  # Feste Höhe für das Label
        progress_layout.addWidget(self.status_label)
        
        control_layout.addWidget(progress_widget)
        control_group.addLayout(control_layout)
        layout.addWidget(control_group)
        
        # Export Group
        export_group = TitledGroupBox("Export")
        export_layout = QVBoxLayout()
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self._save_results)
        self.save_btn.setEnabled(False)
        export_layout.addWidget(self.save_btn)
        export_group.addLayout(export_layout)
        layout.addWidget(export_group)
        
        # Add stretch at the end
        layout.addStretch()
        
    def _setup_plots(self, layout):
        # Create tabs for different plots
        tabs = QTabWidget()
        
        # Tab for Magnetization Plot
        mag_widget = QWidget()
        mag_layout = QVBoxLayout(mag_widget)
        self.mag_figure = Figure(figsize=(8, 6))
        self.mag_canvas = FigureCanvas(self.mag_figure)
        mag_layout.addWidget(self.mag_canvas)
        tabs.addTab(mag_widget, "Magnetization")
        
        # Tab for Phase Plot
        phase_widget = QWidget()
        phase_layout = QVBoxLayout(phase_widget)
        self.phase_figure = Figure(figsize=(8, 6))
        self.phase_canvas = FigureCanvas(self.phase_figure)
        phase_layout.addWidget(self.phase_canvas)
        tabs.addTab(phase_widget, "Phase")
        
        # Tab for Z-Magnetization
        mz_widget = QWidget()
        mz_layout = QVBoxLayout(mz_widget)
        self.mz_figure = Figure(figsize=(8, 6))
        self.mz_canvas = FigureCanvas(self.mz_figure)
        mz_layout.addWidget(self.mz_canvas)
        tabs.addTab(mz_widget, "Z-Magnetization")
        
        layout.addWidget(tabs)
        
    def _load_sequence(self):
        seq_lib_path = str(Path(__file__).resolve().parent.parent.parent / "seq_lib")
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Sequence File",
            seq_lib_path,
            "Sequence Files (*.seq)"
        )
        if filename:
            self.current_seq = filename
            self.seq_label.setText(os.path.basename(filename))
            self._check_start_enabled()
            
    def _load_config(self):
        sim_lib_path = str(Path(__file__).resolve().parent.parent.parent / "sim_lib")
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Config File",
            sim_lib_path,
            "YAML Files (*.yaml)"
        )
        if filename:
            self.current_config = filename
            self.config_label.setText(os.path.basename(filename))
            self._check_start_enabled()
            
    def _check_start_enabled(self):
        # Aktiviere den Start-Button, wenn eine Sequenz vorhanden ist
        # Die Konfiguration kann jetzt entweder extern geladen sein oder über die GUI erstellt werden
        self.start_btn.setEnabled(self.current_seq is not None)
            
    def _run_simulation(self):
        if not self.current_seq:
            QMessageBox.warning(self, "Fehlende Daten", "Bitte laden Sie eine Sequenzdatei.")
            return
            
        # Temporäre YAML-Datei erstellen
        import tempfile
        import yaml
        
        # Temporäre Datei im Text-Modus erstellen
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(self.config_params, tmp, default_flow_style=False)
            temp_config_path = tmp.name
        
        # Lade und validiere die Parameter
        try:
            sim_params = load_params(temp_config_path)
        except (AssertionError, AttributeError) as e:
            QMessageBox.critical(self, "Ungültige Parameter", f"Fehler in den Parametern: {str(e)}")
            return
            
        # Prepare z-positions
        n_iso = self.n_iso.value()
        low = -1e-3
        high = 1e-3
        z_pos = np.linspace(low, high, n_iso)
        z_pos = torch.tensor(z_pos)
        z_pos = torch.cat((z_pos, torch.tensor([0.0])))
        
        # Update GUI status before simulation
        self.start_btn.setEnabled(False)
        self.load_seq_btn.setEnabled(False)
        self.load_config_btn.setEnabled(False)
        self.status_label.setText("Simulation running...")
        self.status_label.setStyleSheet("color: #42a5f5; font-weight: 600; padding: 5px;")
        
        try:
            # Initialize simulation engine
            self.sim_engine = BMCSim(
                adc_time=self.adc_time.value() / 1000.0,
                params=sim_params,
                seq_file=self.current_seq,
                z_positions=z_pos,
                n_backlog=self.n_backlog.value(),
                verbose=False,
                webhook=False
            )
            
            # Setup progress tracking
            total_events = len(self.sim_engine.seq.block_events)
            self.progress.setRange(0, total_events)
            self.progress.setValue(0)
            QApplication.processEvents()
            
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
                self.progress.setValue(i)
                
                # Update status text periodically
                if i % 10 == 0 or i == total_events:
                    percent = int((i / total_events) * 100)
                    self.status_label.setText(f"Running simulation... {percent}%")
                
                QApplication.processEvents()
            
            # Trimme die Magnetisierungsdaten auf die korrekte Länge
            self.sim_engine.m_out = self.sim_engine.m_out[:, :, :self.sim_engine.t.numel()]
            
            # Update plots and GUI status
            self._plot_results()
            
            self.status_label.setText("Simulation completed successfully")
            self.status_label.setStyleSheet("color: #4caf50; font-weight: 600; padding: 5px;")
            self.save_btn.setEnabled(True)
            
        except Exception as e:
            # Handle errors
            error_msg = str(e)
            self.status_label.setText(f"Error: {error_msg[:100]}...")
            self.status_label.setStyleSheet("color: #f44336; font-weight: 600; padding: 5px;")
            print(f"Simulation error: {error_msg}")
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_config_path)
            except:
                pass
                
            # Re-enable controls
            self.start_btn.setEnabled(True)
            self.load_seq_btn.setEnabled(True)
            self.load_config_btn.setEnabled(True)
        
    def _plot_results(self):
        if not self.sim_engine:
            return
            
        # Hole die Zeitscheiben und Magnetisierungsdaten mit get_exact()
        time_slices, magnetization_slices = self.sim_engine.get_exact()
        
        # Hole auch die gesamten Magnetisierungsdaten für die Hintergrunddarstellung
        t, m_z, m_z_total, m_trans, m_trans_total = self.sim_engine.get_mag()
        
        # Konvertiere zu NumPy Arrays
        t_np = t.cpu().numpy()
        m_z_total_np = m_z_total.cpu().numpy()
        m_trans_total_np = m_trans_total.cpu().numpy()
        m_trans_np = m_trans.cpu().numpy()
        
        # Plot Magnetization
        self.mag_figure.clear()
        ax = self.mag_figure.add_subplot(111)
        # Plot komplettes Signal in grau
        ax.plot(t_np, abs(m_trans_total_np), '--o', markersize=2, linewidth=1, color='gray', label=r'$|M_{xy}|$')
        # Plot ADC-Punkte in blau
        # for t_slice, m_slice in zip(time_slices, magnetization_slices):
        #     if 'adc at' in self.sim_engine.events[len(self.sim_engine.events)-1]:  # Prüfe ob es sich um ADC-Daten handelt
        #         ax.plot(t_slice[-1].cpu().numpy(), abs(m_slice[-1].cpu().numpy()), 'b.', label='ADC')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Magnetization')
        ax.grid(True)
        ax.legend()
        self.mag_canvas.draw()
        
        # Plot Phase
        self.phase_figure.clear()
        ax = self.phase_figure.add_subplot(111)
        # Plot komplettes Signal in grau
        ax.plot(t_np, np.angle(m_trans_np[0, :]), 'gray', alpha=0.3, label='Full Signal')
        # Plot ADC-Punkte in rot
        # for t_slice, m_slice in zip(time_slices, magnetization_slices):
        #     if 'adc at' in self.sim_engine.events[len(self.sim_engine.events)-1]:  # Prüfe ob es sich um ADC-Daten handelt
        #         ax.plot(t_slice.cpu().numpy(), np.angle(m_slice.cpu().numpy()), 'r.', label='ADC')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Phase (rad)')
        ax.grid(True)
        ax.legend()
        self.phase_canvas.draw()
        
        # Plot Z-Magnetization
        self.mz_figure.clear()
        ax = self.mz_figure.add_subplot(111)
        # Plot komplettes Signal in grau
        ax.plot(t_np, m_z_total_np, 'gray', alpha=0.3, label='Full Signal')
        # Plot ADC-Punkte in grün
        # for t_slice, m_slice in zip(time_slices, magnetization_slices):
        #     if 'adc at' in self.sim_engine.events[len(self.sim_engine.events)-1]:  # Prüfe ob es sich um ADC-Daten handelt
        #         ax.plot(t_slice.cpu().numpy(), abs(m_slice.cpu().numpy()).real, 'g.', label='ADC')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z-Magnetization')
        ax.grid(True)
        ax.legend()
        self.mz_canvas.draw()
        
    def _save_results(self):
        """
        Save simulation results to a file when the save button is clicked
        """
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

    def _create_water_pool_tab(self):
        """Erstellt den Tab für Wasserpool-Parameter"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Faktoren
        f_layout = QHBoxLayout()
        f_label = QLabel("Relative Poolgröße (f):")
        f_label.setFixedWidth(150)
        self.water_f = QDoubleSpinBox()
        self.water_f.setRange(0.1, 10.0)
        self.water_f.setValue(self.config_params["water_pool"]["f"])
        self.water_f.setSingleStep(0.1)
        self.water_f.valueChanged.connect(lambda val: self._update_param("water_pool", "f", val))
        f_layout.addWidget(f_label)
        f_layout.addWidget(self.water_f)
        layout.addLayout(f_layout)
        
        # T1
        t1_layout = QHBoxLayout()
        t1_label = QLabel("T1 (s):")
        t1_label.setFixedWidth(150)
        self.water_t1 = QDoubleSpinBox()
        self.water_t1.setRange(0.1, 10.0)
        self.water_t1.setValue(self.config_params["water_pool"]["t1"])
        self.water_t1.setSingleStep(0.1)
        self.water_t1.valueChanged.connect(lambda val: self._update_param("water_pool", "t1", val))
        t1_layout.addWidget(t1_label)
        t1_layout.addWidget(self.water_t1)
        layout.addLayout(t1_layout)
        
        # T2
        t2_layout = QHBoxLayout()
        t2_label = QLabel("T2 (s):")
        t2_label.setFixedWidth(150)
        self.water_t2 = QDoubleSpinBox()
        self.water_t2.setRange(0.001, 5.0)
        self.water_t2.setValue(self.config_params["water_pool"]["t2"])
        self.water_t2.setSingleStep(0.01)
        self.water_t2.setDecimals(3)
        self.water_t2.valueChanged.connect(lambda val: self._update_param("water_pool", "t2", val))
        t2_layout.addWidget(t2_label)
        t2_layout.addWidget(self.water_t2)
        layout.addLayout(t2_layout)
        
        layout.addStretch()
        return tab

    def _create_cest_pool_tab(self):
        """Erstellt den Tab für CEST-Pool-Parameter"""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        
        # Dropdown für die Auswahl des CEST-Pools
        self.cest_pool_selection = QComboBox()
        self.cest_pool_selection.addItem("amide")
        self.cest_pool_selection.currentTextChanged.connect(self._update_cest_pool_display)
        
        pool_selection_layout = QHBoxLayout()
        pool_selection_layout.addWidget(QLabel("CEST-Pool:"))
        pool_selection_layout.addWidget(self.cest_pool_selection)
        
        # Buttons zum Hinzufügen/Entfernen von Pools
        add_pool_btn = QPushButton("Neuer Pool")
        add_pool_btn.clicked.connect(self._add_cest_pool)
        remove_pool_btn = QPushButton("Pool entfernen")
        remove_pool_btn.clicked.connect(self._remove_cest_pool)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(add_pool_btn)
        btn_layout.addWidget(remove_pool_btn)
        
        main_layout.addLayout(pool_selection_layout)
        main_layout.addLayout(btn_layout)
        
        # Container für die Pool-Parameter
        self.cest_param_container = QWidget()
        self.cest_param_layout = QVBoxLayout(self.cest_param_container)
        
        # Parameter-Felder
        # f - relative Poolgröße
        f_layout = QHBoxLayout()
        f_label = QLabel("Relative Poolgröße (f):")
        f_label.setFixedWidth(150)
        self.cest_f = QDoubleSpinBox()
        self.cest_f.setRange(0.00001, 0.1)
        self.cest_f.setDecimals(6)
        self.cest_f.setSingleStep(0.00001)
        self.cest_f.valueChanged.connect(lambda val: self._update_cest_param("f", val))
        f_layout.addWidget(f_label)
        f_layout.addWidget(self.cest_f)
        self.cest_param_layout.addLayout(f_layout)
        
        # T1
        t1_layout = QHBoxLayout()
        t1_label = QLabel("T1 (s):")
        t1_label.setFixedWidth(150)
        self.cest_t1 = QDoubleSpinBox()
        self.cest_t1.setRange(0.1, 10.0)
        self.cest_t1.setSingleStep(0.1)
        self.cest_t1.valueChanged.connect(lambda val: self._update_cest_param("t1", val))
        t1_layout.addWidget(t1_label)
        t1_layout.addWidget(self.cest_t1)
        self.cest_param_layout.addLayout(t1_layout)
        
        # T2
        t2_layout = QHBoxLayout()
        t2_label = QLabel("T2 (s):")
        t2_label.setFixedWidth(150)
        self.cest_t2 = QDoubleSpinBox()
        self.cest_t2.setRange(0.001, 5.0)
        self.cest_t2.setSingleStep(0.01)
        self.cest_t2.setDecimals(3)
        self.cest_t2.valueChanged.connect(lambda val: self._update_cest_param("t2", val))
        t2_layout.addWidget(t2_label)
        t2_layout.addWidget(self.cest_t2)
        self.cest_param_layout.addLayout(t2_layout)
        
        # k - Austauschrate
        k_layout = QHBoxLayout()
        k_label = QLabel("Austauschrate (k) [Hz]:")
        k_label.setFixedWidth(150)
        self.cest_k = QDoubleSpinBox()
        self.cest_k.setRange(0, 1000)
        self.cest_k.setSingleStep(10)
        self.cest_k.valueChanged.connect(lambda val: self._update_cest_param("k", val))
        k_layout.addWidget(k_label)
        k_layout.addWidget(self.cest_k)
        self.cest_param_layout.addLayout(k_layout)
        
        # dw - Chemische Verschiebung
        dw_layout = QHBoxLayout()
        dw_label = QLabel("Chem. Verschiebung (dw) [ppm]:")
        dw_label.setFixedWidth(150)
        self.cest_dw = QDoubleSpinBox()
        self.cest_dw.setRange(-10, 10)
        self.cest_dw.setSingleStep(0.5)
        self.cest_dw.setDecimals(2)
        self.cest_dw.valueChanged.connect(lambda val: self._update_cest_param("dw", val))
        dw_layout.addWidget(dw_label)
        dw_layout.addWidget(self.cest_dw)
        self.cest_param_layout.addLayout(dw_layout)
        
        main_layout.addWidget(self.cest_param_container)
        main_layout.addStretch()
        
        # Anfangsparameter setzen
        self._update_cest_pool_display("amide")
        
        return tab

    def _create_scanner_tab(self):
        """Erstellt den Tab für Scanner-Einstellungen"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # B0 - Feldstärke
        b0_layout = QHBoxLayout()
        b0_label = QLabel("B0-Feldstärke [T]:")
        b0_label.setFixedWidth(150)
        self.scanner_b0 = QDoubleSpinBox()
        self.scanner_b0.setRange(1.0, 20.0)
        self.scanner_b0.setValue(self.config_params["b0"])
        self.scanner_b0.setSingleStep(0.1)
        # Direkter Parameter, kein param_key
        self.scanner_b0.valueChanged.connect(lambda val: self._update_param("b0", None, val))
        b0_layout.addWidget(b0_label)
        b0_layout.addWidget(self.scanner_b0)
        layout.addLayout(b0_layout)
        
        # gamma - gyromagnetisches Verhältnis
        gamma_layout = QHBoxLayout()
        gamma_label = QLabel("Gyromagn. Verhältnis [rad/uT]:")
        gamma_label.setFixedWidth(150)
        self.scanner_gamma = QDoubleSpinBox()
        self.scanner_gamma.setRange(100.0, 300.0)
        self.scanner_gamma.setValue(self.config_params["gamma"])
        self.scanner_gamma.setSingleStep(0.1)
        self.scanner_gamma.setDecimals(4)
        # Direkter Parameter, kein param_key
        self.scanner_gamma.valueChanged.connect(lambda val: self._update_param("gamma", None, val))
        gamma_layout.addWidget(gamma_label)
        gamma_layout.addWidget(self.scanner_gamma)
        layout.addLayout(gamma_layout)
        
        # b0_inhom - Feld-Inhomogenität
        b0_inhom_layout = QHBoxLayout()
        b0_inhom_label = QLabel("B0-Inhomogenität [ppm]:")
        b0_inhom_label.setFixedWidth(150)
        self.scanner_b0_inhom = QDoubleSpinBox()
        self.scanner_b0_inhom.setRange(0.0, 1.0)
        self.scanner_b0_inhom.setValue(self.config_params["b0_inhom"])
        self.scanner_b0_inhom.setSingleStep(0.01)
        self.scanner_b0_inhom.setDecimals(3)
        # Direkter Parameter, kein param_key
        self.scanner_b0_inhom.valueChanged.connect(lambda val: self._update_param("b0_inhom", None, val))
        b0_inhom_layout.addWidget(b0_inhom_label)
        b0_inhom_layout.addWidget(self.scanner_b0_inhom)
        layout.addLayout(b0_inhom_layout)
        
        # rel_b1 - relative Amplituden-Inhomogenität
        rel_b1_layout = QHBoxLayout()
        rel_b1_label = QLabel("Relative B1-Inhomogenität:")
        rel_b1_label.setFixedWidth(150)
        self.scanner_rel_b1 = QDoubleSpinBox()
        self.scanner_rel_b1.setRange(0.1, 2.0)
        self.scanner_rel_b1.setValue(self.config_params["rel_b1"])
        self.scanner_rel_b1.setSingleStep(0.1)
        # Direkter Parameter, kein param_key
        self.scanner_rel_b1.valueChanged.connect(lambda val: self._update_param("rel_b1", None, val))
        rel_b1_layout.addWidget(rel_b1_label)
        rel_b1_layout.addWidget(self.scanner_rel_b1)
        layout.addLayout(rel_b1_layout)
        
        layout.addStretch()
        return tab

    def _create_advanced_tab(self):
        """Erstellt den Tab für erweiterte Einstellungen"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # verbose - ausführliche Ausgabe
        verbose_layout = QHBoxLayout()
        verbose_label = QLabel("Ausführliche Ausgabe:")
        verbose_label.setFixedWidth(150)
        self.advanced_verbose = QCheckBox()
        self.advanced_verbose.setChecked(self.config_params["verbose"])
        # Direkter Parameter, kein param_key
        self.advanced_verbose.stateChanged.connect(lambda state: self._update_param("verbose", None, state == Qt.CheckState.Checked))
        verbose_layout.addWidget(verbose_label)
        verbose_layout.addWidget(self.advanced_verbose)
        layout.addLayout(verbose_layout)
        
        # reset_init_mag - Magnetisierung zurücksetzen
        reset_layout = QHBoxLayout()
        reset_label = QLabel("Magnetisierung zurücksetzen:")
        reset_label.setFixedWidth(150)
        self.advanced_reset = QCheckBox()
        self.advanced_reset.setChecked(self.config_params["reset_init_mag"])
        # Direkter Parameter, kein param_key
        self.advanced_reset.stateChanged.connect(lambda state: self._update_param("reset_init_mag", None, state == Qt.CheckState.Checked))
        reset_layout.addWidget(reset_label)
        reset_layout.addWidget(self.advanced_reset)
        layout.addLayout(reset_layout)
        
        # scale - relative Magnetisierung
        scale_layout = QHBoxLayout()
        scale_label = QLabel("Relative Magnetisierung:")
        scale_label.setFixedWidth(150)
        self.advanced_scale = QDoubleSpinBox()
        self.advanced_scale.setRange(0.1, 10.0)
        self.advanced_scale.setValue(self.config_params["scale"])
        self.advanced_scale.setSingleStep(0.1)
        # Direkter Parameter, kein param_key
        self.advanced_scale.valueChanged.connect(lambda val: self._update_param("scale", None, val))
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.advanced_scale)
        layout.addLayout(scale_layout)
        
        # max_pulse_samples - maximale Anzahl Samples
        samples_layout = QHBoxLayout()
        samples_label = QLabel("Max. Pulse Samples:")
        samples_label.setFixedWidth(150)
        self.advanced_samples = QSpinBox()
        self.advanced_samples.setRange(100, 5000)
        self.advanced_samples.setValue(self.config_params["max_pulse_samples"])
        self.advanced_samples.setSingleStep(100)
        # Direkter Parameter, kein param_key
        self.advanced_samples.valueChanged.connect(lambda val: self._update_param("max_pulse_samples", None, val))
        samples_layout.addWidget(samples_label)
        samples_layout.addWidget(self.advanced_samples)
        layout.addLayout(samples_layout)
        
        layout.addStretch()
        return tab

    def _update_param(self, key, param_key=None, value=None):
        """Aktualisiert einen Parameter in der Konfiguration"""
        if param_key is not None:
            # Für verschachtelte Parameter wie water_pool
            if key not in self.config_params:
                self.config_params[key] = {}
            self.config_params[key][param_key] = value
        else:
            # Für direkte Parameter wie b0, gamma etc.
            self.config_params[key] = value
            
    def _update_cest_param(self, param_key, value):
        """Aktualisiert einen CEST-Pool-Parameter"""
        pool_name = self.cest_pool_selection.currentText()
        if "cest_pool" not in self.config_params:
            self.config_params["cest_pool"] = {}
        if pool_name not in self.config_params["cest_pool"]:
            self.config_params["cest_pool"][pool_name] = {}
        
        self.config_params["cest_pool"][pool_name][param_key] = value

    def _update_cest_pool_display(self, pool_name):
        """Aktualisiert die Anzeige der CEST-Pool-Parameter"""
        if "cest_pool" in self.config_params and pool_name in self.config_params["cest_pool"]:
            pool = self.config_params["cest_pool"][pool_name]
            self.cest_f.setValue(pool.get("f", 0.00064865))
            self.cest_t1.setValue(pool.get("t1", 1.3))
            self.cest_t2.setValue(pool.get("t2", 0.1))
            self.cest_k.setValue(pool.get("k", 50))
            self.cest_dw.setValue(pool.get("dw", 8))
        else:
            # Standardwerte setzen
            self.cest_f.setValue(0.00064865)
            self.cest_t1.setValue(1.3)
            self.cest_t2.setValue(0.1)
            self.cest_k.setValue(50)
            self.cest_dw.setValue(8)
            
            # Pool zur Konfiguration hinzufügen
            if "cest_pool" not in self.config_params:
                self.config_params["cest_pool"] = {}
            
            self.config_params["cest_pool"][pool_name] = {
                "f": 0.00064865,
                "t1": 1.3,
                "t2": 0.1,
                "k": 50,
                "dw": 8
            }

    def _add_cest_pool(self):
        """Fügt einen neuen CEST-Pool hinzu"""
        pool_name, ok = QInputDialog.getText(self, "Neuer CEST-Pool", "Name des neuen Pools:")
        if ok and pool_name:
            if "cest_pool" not in self.config_params:
                self.config_params["cest_pool"] = {}
                
            if pool_name not in self.config_params["cest_pool"]:
                # Standard-Pool-Parameter
                self.config_params["cest_pool"][pool_name] = {
                    "f": 0.00064865,
                    "t1": 1.3,
                    "t2": 0.1,
                    "k": 50,
                    "dw": 5
                }
                
                # Pool zur Dropdown-Liste hinzufügen
                self.cest_pool_selection.addItem(pool_name)
                self.cest_pool_selection.setCurrentText(pool_name)
            else:
                QMessageBox.warning(self, "Doppelter Pool", f"Ein Pool mit dem Namen '{pool_name}' existiert bereits.")

    def _remove_cest_pool(self):
        """Entfernt den aktuell ausgewählten CEST-Pool"""
        pool_name = self.cest_pool_selection.currentText()
        if pool_name and self.cest_pool_selection.count() > 1:
            reply = QMessageBox.question(self, 'Pool entfernen', 
                                       f"Sind Sie sicher, dass Sie den Pool '{pool_name}' entfernen möchten?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                        QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                # Pool aus der Konfiguration entfernen
                if "cest_pool" in self.config_params and pool_name in self.config_params["cest_pool"]:
                    del self.config_params["cest_pool"][pool_name]
                
                # Pool aus der Dropdown-Liste entfernen
                current_index = self.cest_pool_selection.currentIndex()
                self.cest_pool_selection.removeItem(current_index)
        else:
            QMessageBox.information(self, "Information", "Sie müssen mindestens einen CEST-Pool behalten.")

    def _save_config(self):
        """Speichert die aktuelle Konfiguration in eine YAML-Datei"""
        sim_lib_path = str(Path(__file__).resolve().parent.parent.parent / "sim_lib")
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Konfiguration speichern",
            sim_lib_path,
            "YAML Files (*.yaml)"
        )
        
        if filename:
            import yaml
            with open(filename, 'w') as f:
                yaml.dump(self.config_params, f, default_flow_style=False)
            
            QMessageBox.information(self, "Erfolgreich", f"Konfiguration wurde unter {os.path.basename(filename)} gespeichert.")

    def _reset_config(self):
        """Setzt die Konfiguration auf Standardwerte zurück"""
        reply = QMessageBox.question(self, 'Zurücksetzen', 
                                   "Sind Sie sicher, dass Sie alle Parameter auf die Standardwerte zurücksetzen möchten?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self._load_default_config()
            
            # GUI-Elemente aktualisieren
            # Wasserpool
            self.water_f.setValue(self.config_params["water_pool"]["f"])
            self.water_t1.setValue(self.config_params["water_pool"]["t1"])
            self.water_t2.setValue(self.config_params["water_pool"]["t2"])
            
            # CEST-Pool zurücksetzen
            self.cest_pool_selection.clear()
            self.cest_pool_selection.addItem("amide")
            self._update_cest_pool_display("amide")
            
            # Scanner-Einstellungen
            self.scanner_b0.setValue(self.config_params["b0"])
            self.scanner_gamma.setValue(self.config_params["gamma"])
            self.scanner_b0_inhom.setValue(self.config_params["b0_inhom"])
            self.scanner_rel_b1.setValue(self.config_params["rel_b1"])
            
            # Erweiterte Einstellungen
            self.advanced_verbose.setChecked(self.config_params["verbose"])
            self.advanced_reset.setChecked(self.config_params["reset_init_mag"])
            self.advanced_scale.setValue(self.config_params["scale"])
            self.advanced_samples.setValue(self.config_params["max_pulse_samples"])
            
            QMessageBox.information(self, "Zurückgesetzt", "Alle Parameter wurden auf die Standardwerte zurückgesetzt.")

