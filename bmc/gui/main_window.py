import sys
import os
from pathlib import Path
import yaml
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QSpinBox,
    QDoubleSpinBox, QProgressBar, QTabWidget, QGroupBox, QStyleFactory,
    QFrame
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPalette, QColor, QFont, QIcon
import matplotlib
matplotlib.use('QtAgg')  # Verwende QtAgg für PyQt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import torch

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
        layout.setContentsMargins(0, 15, 0, 0)  # Platz für den Titel oben
        layout.setSpacing(5)  # Reduzierter Abstand zwischen Titel und Box

        # Titel-Label ohne Rahmen
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 0px 5px;
                background-color: #2b2b2b;
                border: none;
            }
        """)
        title_container = QWidget()
        title_container.setStyleSheet("background: transparent; border: none;")
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(10, 0, 0, 0)
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        
        # Box mit nur äußerem Rahmen
        self.box = QWidget()
        self.box.setStyleSheet(f"""
            QWidget {{
                background-color: #333333;
                border: 2px solid #444444;
                border-radius: 5px;
            }}
            
            QPushButton {{
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 3px;
                font-weight: bold;
                min-width: 120px;
            }}
            
            QPushButton:hover {{
                background-color: #1565c0;
            }}
            
            QPushButton:pressed {{
                background-color: #0a3d91;
            }}
            
            QPushButton:disabled {{
                background-color: #666666;
                color: #999999;
            }}
            
            QLabel {{
                color: #ffffff;
                padding: 2px;
                border: none;
            }}
            
            QProgressBar {{
                border: none;
                background-color: #2b2b2b;
                min-height: 30px;
                text-align: center;
                color: white;
                font-weight: bold;
                font-size: 13px;
                margin: 0px;
            }}
            
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                stop:0 #1565C0, stop:1 #42a5f5);
                border-radius: 0px;
            }}
            
            QSpinBox, QDoubleSpinBox {{
                background-color: #424242;
                color: white;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px 25px 2px 5px;
                min-width: 80px;
            }}
            
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                height: 12px;
                border: none;
                background-color: #555555;
                border-top-right-radius: 3px;
            }}
            
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 20px;
                height: 12px;
                border: none;
                background-color: #555555;
                border-bottom-right-radius: 3px;
            }}
            
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: #666666;
            }}
            
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
                background-color: #444444;
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
        self.box_layout.setContentsMargins(15, 15, 15, 15)

        # Füge Titel und Box zum Layout hinzu
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
        # Remove global stylesheet
        # self.setStyleSheet(STYLE_SHEET)
        
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

    def _setup_controls(self, layout):
        # File Selection Group
        file_group = TitledGroupBox("Dateiauswahl")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(10)
        
        # Sequenz-Datei laden
        seq_group = QWidget()
        seq_layout = QHBoxLayout(seq_group)
        seq_layout.setSpacing(10)
        self.seq_label = QLabel("Keine Sequenz geladen")
        self.seq_label.setMinimumWidth(200)
        self.load_seq_btn = QPushButton("Sequenz laden")
        self.load_seq_btn.setFixedWidth(150)
        self.load_seq_btn.clicked.connect(self._load_sequence)
        seq_layout.addWidget(self.seq_label)
        seq_layout.addWidget(self.load_seq_btn)
        file_layout.addWidget(seq_group)
        
        # Konfigurations-Datei laden
        config_group = QWidget()
        config_layout = QHBoxLayout(config_group)
        config_layout.setSpacing(10)
        self.config_label = QLabel("Keine Konfiguration geladen")
        self.config_label.setMinimumWidth(200)
        self.load_config_btn = QPushButton("Config laden")
        self.load_config_btn.setFixedWidth(150)
        self.load_config_btn.clicked.connect(self._load_config)
        config_layout.addWidget(self.config_label)
        config_layout.addWidget(self.load_config_btn)
        file_layout.addWidget(config_group)
        
        file_group.addLayout(file_layout)
        layout.addWidget(file_group)
        
        # Parameter Group
        param_group = TitledGroupBox("Parameter")
        param_layout = QVBoxLayout()
        param_layout.setSpacing(15)
        
        # ADC Zeit
        adc_layout = QHBoxLayout()
        adc_layout.setSpacing(10)
        adc_label = QLabel("ADC Zeit (ms):")
        adc_label.setFixedWidth(150)
        self.adc_time = QDoubleSpinBox()
        self.adc_time.setRange(0.1, 100.0)
        self.adc_time.setValue(5.0)
        self.adc_time.setSingleStep(0.1)
        adc_layout.addWidget(adc_label)
        adc_layout.addWidget(self.adc_time)
        param_layout.addLayout(adc_layout)
        
        # Isochromaten
        iso_layout = QHBoxLayout()
        iso_layout.setSpacing(10)
        iso_label = QLabel("Isochromaten:")
        iso_label.setFixedWidth(150)
        self.n_iso = QSpinBox()
        self.n_iso.setRange(10, 1000)
        self.n_iso.setValue(100)
        iso_layout.addWidget(iso_label)
        iso_layout.addWidget(self.n_iso)
        param_layout.addLayout(iso_layout)

        # Backlog
        backlog_layout = QHBoxLayout()
        backlog_layout.setSpacing(10)
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
        
        # Simulation Control Group
        control_group = TitledGroupBox("Steuerung")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(15)
        
        # Start Button
        self.start_btn = QPushButton("Simulation starten")
        self.start_btn.clicked.connect(self._run_simulation)
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        
        # Progress Widget mit Status
        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setSpacing(8)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        # Progress Bar
        self.progress = AnimatedProgressBar()
        progress_layout.addWidget(self.progress)
        
        # Status Label direkt unter der Progress Bar
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                padding: 5px;
                font-weight: bold;
                margin-top: 0px;
            }
        """)
        progress_layout.addWidget(self.status_label)
        
        control_layout.addWidget(progress_widget)
        control_group.addLayout(control_layout)
        layout.addWidget(control_group)
        
        # Export Group
        export_group = TitledGroupBox("Export")
        export_layout = QVBoxLayout()
        self.save_btn = QPushButton("Ergebnisse speichern")
        self.save_btn.clicked.connect(self._save_results)
        self.save_btn.setEnabled(False)
        export_layout.addWidget(self.save_btn)
        export_group.addLayout(export_layout)
        layout.addWidget(export_group)
        
        # Füge Stretch hinzu
        layout.addStretch()
        
    def _setup_plots(self, layout):
        # Erstelle Tabs für verschiedene Plots
        tabs = QTabWidget()
        
        # Tab für Magnetisierungs-Plot
        mag_widget = QWidget()
        mag_layout = QVBoxLayout(mag_widget)
        
        # Matplotlib Figure für Magnetisierung
        self.mag_figure = Figure(figsize=(8, 6))
        self.mag_canvas = FigureCanvas(self.mag_figure)
        mag_layout.addWidget(self.mag_canvas)
        
        tabs.addTab(mag_widget, "Magnetisierung")
        
        # Tab für Phasen-Plot
        phase_widget = QWidget()
        phase_layout = QVBoxLayout(phase_widget)
        
        # Matplotlib Figure für Phase
        self.phase_figure = Figure(figsize=(8, 6))
        self.phase_canvas = FigureCanvas(self.phase_figure)
        phase_layout.addWidget(self.phase_canvas)
        
        tabs.addTab(phase_widget, "Phase")
        
        # Tab für z-Magnetisierung
        mz_widget = QWidget()
        mz_layout = QVBoxLayout(mz_widget)
        
        # Matplotlib Figure für z-Magnetisierung
        self.mz_figure = Figure(figsize=(8, 6))
        self.mz_canvas = FigureCanvas(self.mz_figure)
        mz_layout.addWidget(self.mz_canvas)
        
        tabs.addTab(mz_widget, "Z-Magnetisierung")
        
        layout.addWidget(tabs)
        
    def _load_sequence(self):
        seq_lib_path = str(Path(__file__).resolve().parent.parent.parent / "seq_lib")
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Sequenz-Datei laden",
            seq_lib_path,
            "Sequenz Files (*.seq)"
        )
        if filename:
            self.current_seq = filename
            self.seq_label.setText(os.path.basename(filename))
            self._check_start_enabled()
            
    def _load_config(self):
        sim_lib_path = str(Path(__file__).resolve().parent.parent.parent / "sim_lib")
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Konfigurations-Datei laden",
            sim_lib_path,
            "YAML Files (*.yaml)"
        )
        if filename:
            self.current_config = filename
            self.config_label.setText(os.path.basename(filename))
            self._check_start_enabled()
            
    def _check_start_enabled(self):
        self.start_btn.setEnabled(
            self.current_seq is not None and 
            self.current_config is not None
        )
            
    def _run_simulation(self):
        if not self.current_seq or not self.current_config:
            return
            
        # Parameter vorbereiten
        sim_params = load_params(self.current_config)
        
        # Z-Positionen vorbereiten
        n_iso = self.n_iso.value()
        low = -1e-3
        high = 1e-3
        z_pos = np.linspace(low, high, n_iso)
        z_pos = torch.tensor(z_pos)
        z_pos = torch.cat((z_pos, torch.tensor([0.0])))
        
        # GUI-Status aktualisieren
        self.start_btn.setEnabled(False)
        self.status_label.setText("Simulation läuft...")
        self.status_label.setStyleSheet("color: #42a5f5; font-weight: bold; padding: 5px;")
        
        try:
            # Simulation initialisieren
            self.sim_engine = BMCSim(
                adc_time=self.adc_time.value() / 1000.0,
                params=sim_params,
                seq_file=self.current_seq,
                z_positions=z_pos,
                n_backlog=self.n_backlog.value(),
                verbose=False,  # Deaktiviere tqdm, da wir unseren eigenen Fortschrittsbalken verwenden
                webhook=False
            )
            
            # Progress Bar vorbereiten
            total_events = len(self.sim_engine.seq.block_events)
            self.progress.setRange(0, total_events)
            self.progress.setValue(0)
            
            # Custom Fortschritts-Tracking
            current_adc = 1
            mag = torch.tensor(
                self.sim_engine.m_init[np.newaxis, np.newaxis, :, np.newaxis], 
                dtype=torch.float64,
                device=GLOBAL_DEVICE
            )
            
            # Simulation ausführen mit Fortschrittsanzeige
            for i, block_event in enumerate(self.sim_engine.seq.block_events, start=1):
                block = self.sim_engine.seq.get_block(block_event)
                counter = np.abs(total_events - i)
                current_adc, mag = self.sim_engine.run_adc(block, current_adc, mag, counter)
                self.progress.update_progress(i, total_events)
            
            # Aktualisiere die Anzeige
            self._plot_results()
            
            # GUI-Status aktualisieren
            self.status_label.setText("Simulation erfolgreich abgeschlossen")
            self.status_label.setStyleSheet("color: #4caf50; font-weight: bold; padding: 5px;")
            self.save_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            
        except Exception as e:
            self.status_label.setText(f"Fehler: {str(e)}")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold; padding: 5px;")
            self.start_btn.setEnabled(True)
        
    def _plot_results(self):
        if not self.sim_engine:
            return
            
        # Hole Magnetisierungsdaten
        t, m_z, m_z_total, m_trans, m_trans_total = self.sim_engine.get_mag()
        
        # Konvertiere zu NumPy Arrays und stelle sicher, dass die Dimensionen passen
        t_np = t.cpu().numpy()
        m_z_total_np = m_z_total.cpu().numpy()
        m_trans_total_np = m_trans_total.cpu().numpy()
        m_trans_np = m_trans.cpu().numpy()

        # Stelle sicher, dass die Arrays die gleiche Länge haben
        min_len = min(len(t_np), len(m_z_total_np), len(m_trans_total_np))
        t_np = t_np[:min_len]
        m_z_total_np = m_z_total_np[:min_len]
        m_trans_total_np = m_trans_total_np[:min_len]
        
        # Plot Magnetisierung
        self.mag_figure.clear()
        ax = self.mag_figure.add_subplot(111)
        ax.plot(t_np, abs(m_trans_total_np), 'b-', label='|Mxy|')
        ax.set_xlabel('Zeit (s)')
        ax.set_ylabel('Magnetisierung')
        ax.grid(True)
        ax.legend()
        self.mag_canvas.draw()
        
        # Plot Phase
        self.phase_figure.clear()
        ax = self.phase_figure.add_subplot(111)
        ax.plot(t_np, np.angle(m_trans_np[0, :min_len]), 'r-', label='Phase')
        ax.set_xlabel('Zeit (s)')
        ax.set_ylabel('Phase (rad)')
        ax.grid(True)
        ax.legend()
        self.phase_canvas.draw()
        
        # Plot Z-Magnetisierung
        self.mz_figure.clear()
        ax = self.mz_figure.add_subplot(111)
        ax.plot(t_np, m_z_total_np, 'g-', label='Mz')
        ax.set_xlabel('Zeit (s)')
        ax.set_ylabel('Z-Magnetisierung')
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
            "Ergebnisse speichern",
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

