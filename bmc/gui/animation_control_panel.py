from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QSpinBox, QDoubleSpinBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QLocale
from pathlib import Path

from bmc.gui.components import TitledGroupBox, AnimatedProgressBar
from bmc.gui.animated_toggle import AnimatedToggle


class AnimationControlPanel(QWidget):
    """A class representing the animation control panel in the BMC Simulator GUI."""
    
    # Define signals for communicating with the main window
    startAnimationClicked = pyqtSignal(dict)
    saveVideoClicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.seq_path = None
        # Setze das Locale für QDoubleSpinBox-Widgets auf Englisch
        self.locale = QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface for the animation control panel."""
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
        self._setup_file_info(control_panel_layout)
        self._setup_animation_parameters(control_panel_layout)
        self._setup_quality_settings(control_panel_layout)
        self._setup_animation_control(control_panel_layout)
        self._setup_export(control_panel_layout)
        
        # Add stretch at the end
        control_panel_layout.addStretch()
        
    def _setup_file_info(self, layout):
        """Set up the file info section."""
        file_group = TitledGroupBox("Current Sequence")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(12)
        
        # Sequence file info
        seq_group = QWidget()
        seq_layout = QHBoxLayout(seq_group)
        seq_layout.setSpacing(12)
        self.seq_label = QLabel("No simulation data available")
        self.seq_label.setMinimumWidth(200)
        seq_layout.addWidget(self.seq_label)
        file_layout.addWidget(seq_group)
        
        file_group.addLayout(file_layout)
        layout.addWidget(file_group)
    
    def _setup_animation_parameters(self, layout):
        """Set up the animation parameters."""
        param_group = TitledGroupBox("Animation Parameters")
        param_layout = QVBoxLayout()
        param_layout.setSpacing(16)
        
        # Run Time
        runtime_layout = QHBoxLayout()
        runtime_layout.setSpacing(12)
        runtime_label = QLabel("Animation Speed:")
        runtime_label.setFixedWidth(150)
        self.runtime = QDoubleSpinBox()
        self.runtime.setLocale(self.locale)  # Setze Englisches Locale
        self.runtime.setRange(0.01, 2.0)
        self.runtime.setValue(0.1)  # Default value
        self.runtime.setSingleStep(0.05)
        self.runtime.setDecimals(2)
        runtime_layout.addWidget(runtime_label)
        runtime_layout.addWidget(self.runtime)
        param_layout.addLayout(runtime_layout)
        
        # Step Size
        step_layout = QHBoxLayout()
        step_layout.setSpacing(12)
        step_label = QLabel("Frame Step Size:")
        step_label.setFixedWidth(150)
        self.step_size = QSpinBox()
        self.step_size.setRange(1, 100)
        self.step_size.setValue(1)  # Default value
        self.step_size.setSingleStep(1)
        step_layout.addWidget(step_label)
        step_layout.addWidget(self.step_size)
        param_layout.addLayout(step_layout)

        # Toggle für Track Path
        track_path_layout = QHBoxLayout()
        track_path_layout.setSpacing(40)
        track_path_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        track_path_label = QLabel("Track Magnetization Path:")
        track_path_label.setFixedWidth(170)
        
        self.track_path_toggle = AnimatedToggle(
            checked_color="#2962FF", 
            pulse_checked_color="#4400B0EE"
        )
        self.track_path_toggle.setFixedSize(self.track_path_toggle.sizeHint())
        self.track_path_toggle.setChecked(True)  # Default on
        
        track_path_layout.addWidget(track_path_label)
        track_path_layout.addWidget(self.track_path_toggle)
        track_path_layout.addStretch()
        param_layout.addLayout(track_path_layout)

        # Toggle for Timing Display
        timing_layout = QHBoxLayout()
        timing_layout.setSpacing(40)
        timing_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        timing_label = QLabel("Show Time Display:")
        timing_label.setFixedWidth(170)
        
        self.timing_toggle = AnimatedToggle(
            checked_color="#2962FF", 
            pulse_checked_color="#4400B0EE"
        )
        self.timing_toggle.setFixedSize(self.timing_toggle.sizeHint())
        self.timing_toggle.setChecked(False)  # Default off
        
        timing_layout.addWidget(timing_label)
        timing_layout.addWidget(self.timing_toggle)
        timing_layout.addStretch()
        param_layout.addLayout(timing_layout)

        # Toggle for Total Magnetization
        total_mag_layout = QHBoxLayout()
        total_mag_layout.setSpacing(40)
        total_mag_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        total_mag_label = QLabel("Show Total Magnetization:")
        total_mag_label.setFixedWidth(170)
        
        self.total_mag_toggle = AnimatedToggle(
            checked_color="#2962FF", 
            pulse_checked_color="#4400B0EE"
        )
        self.total_mag_toggle.setFixedSize(self.total_mag_toggle.sizeHint())
        self.total_mag_toggle.setChecked(False)  # Default off
        
        total_mag_layout.addWidget(total_mag_label)
        total_mag_layout.addWidget(self.total_mag_toggle)
        total_mag_layout.addStretch()
        param_layout.addLayout(total_mag_layout)

        # Toggle for CEST Pool Animation
        animate_cest_layout = QHBoxLayout()
        animate_cest_layout.setSpacing(40)
        animate_cest_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        animate_cest_label = QLabel("Animate CEST Pool:")
        animate_cest_label.setFixedWidth(170)
        
        self.animate_cest_toggle = AnimatedToggle(
            checked_color="#2962FF", 
            pulse_checked_color="#4400B0EE"
        )
        self.animate_cest_toggle.setFixedSize(self.animate_cest_toggle.sizeHint())
        self.animate_cest_toggle.setChecked(True)  # Default on
        
        animate_cest_layout.addWidget(animate_cest_label)
        animate_cest_layout.addWidget(self.animate_cest_toggle)
        animate_cest_layout.addStretch()
        param_layout.addLayout(animate_cest_layout)
        
        param_group.addLayout(param_layout)
        layout.addWidget(param_group)
    
    def _setup_quality_settings(self, layout):
        """Set up quality settings."""
        quality_group = TitledGroupBox("Video Quality")
        quality_layout = QVBoxLayout()
        quality_layout.setSpacing(16)
        
        # Quality dropdown
        quality_box_layout = QHBoxLayout()
        quality_box_layout.setSpacing(12)
        quality_label = QLabel("Quality:")
        quality_label.setFixedWidth(150)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Low", "Medium", "High"])
        self.quality_combo.setCurrentIndex(0)  # Default: Low
        quality_box_layout.addWidget(quality_label)
        quality_box_layout.addWidget(self.quality_combo)
        quality_layout.addLayout(quality_box_layout)
        
        quality_group.addLayout(quality_layout)
        layout.addWidget(quality_group)
    
    def _setup_animation_control(self, layout):
        """Set up the animation control group."""
        control_group = TitledGroupBox("Control")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(24)
        
        # Start Button
        self.start_btn = QPushButton("Start Animation")
        self.start_btn.clicked.connect(self._on_start_animation)
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
        self.save_btn = QPushButton("Save Video")
        self.save_btn.clicked.connect(self._on_save_video)
        self.save_btn.setEnabled(False)
        export_layout.addWidget(self.save_btn)
        export_group.addLayout(export_layout)
        layout.addWidget(export_group)
    
    def _on_start_animation(self):
        """Handle animation start button click."""
        # Collect all animation parameters
        animation_params = {
            'run_time': self.runtime.value(),
            'step': self.step_size.value(),
            'track_path': self.track_path_toggle.isChecked(),
            'timing': self.timing_toggle.isChecked(),
            'total_mag': self.total_mag_toggle.isChecked(),
            'animate_cest': self.animate_cest_toggle.isChecked(),
        }
        
        # Add quality setting
        quality_mapping = {
            0: '-ql',  # Low quality
            1: '-qm',  # Medium quality
            2: '-qh'   # High quality
        }
        animation_params['quality'] = quality_mapping.get(self.quality_combo.currentIndex(), '-ql')
        animation_params['write'] = '--write_to_movie'  # Always write to movie
        
        self.startAnimationClicked.emit(animation_params)
    
    def _on_save_video(self):
        """Handle save video button click."""
        self.saveVideoClicked.emit()
    
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
        """Enable or disable control elements during animation."""
        self.start_btn.setEnabled(enabled and self.seq_path is not None)
    
    def enable_save_button(self, enabled=True):
        """Enable or disable the save button."""
        self.save_btn.setEnabled(enabled)
    
    def set_sequence_info(self, seq_path=None):
        """Set the sequence info label."""
        if seq_path:
            self.seq_path = seq_path
            self.seq_label.setText(Path(seq_path).name)
            self.start_btn.setEnabled(True)
        else:
            self.seq_label.setText("No simulation data available")
            self.start_btn.setEnabled(False)