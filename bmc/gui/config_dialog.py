from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QMessageBox, QInputDialog, QFileDialog, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from pathlib import Path
import yaml
import os

class IconButton(QPushButton):
    """Ein einfacher Button mit Symbol (+/-) und optimiertem Styling"""
    def __init__(self, text, tooltip, color, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(72, 40)  # Quadratische Buttons
        self.setMaximumSize(72, 40)  # Verhindert Überdehnung
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)  # Fixierte Größenpolitik
        self.setToolTip(tooltip)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {color};
                border: 1px solid #444444;
                border-radius: 6px;  /* Angepasster Radius für kleinere Buttons */
                font-size: 30px;  /* Kleinere Schrift */
                font-weight: bold;
                padding: 0px;
                margin: 0px;
                text-align: center;
                padding-bottom: 2px; /* Kleinere Korrektur der vertikalen Position */
                width: 72px;
                height: 40px;
                min-width: 72px;
                min-height: 40px;
                max-width: 72px;
                max-height: 40px;
                padding-bottom: 4px;
            }}
            QPushButton:hover {{
                background-color: rgba({', '.join(str(int(c)) for c in self._hex_to_rgb(color))}, 0.1);
                border: 1px solid {color};
            }}
            QPushButton:pressed {{
                background-color: rgba({', '.join(str(int(c)) for c in self._hex_to_rgb(color))}, 0.2);
                border: 1px solid {color};
            }}
        """)
    
    def _hex_to_rgb(self, hex_color):
        # Konvertiere Hex-Farbe zu RGB
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

class ConfigDialog(QDialog):
    def __init__(self, config_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parameter Configuration")
        self.setModal(True)  # Makes the dialog modal
        
        # Get absolute paths for the arrow images
        current_dir = Path(__file__).resolve().parent
        up_arrow_path = current_dir / 'images' / 'upload.png'
        down_arrow_path = current_dir / 'images' / 'download.png'
        
        # Convert to URL format that Qt understands
        up_arrow_url = f"{up_arrow_path.as_posix()}"
        down_arrow_url = f"{down_arrow_path.as_posix()}"
        
        # Store the configuration parameters
        self.config_params = config_params.copy()
        
        # Set the style for input fields including the arrow images
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #1E1E1E;
            }}
            
            QDoubleSpinBox, QSpinBox {{
                background-color: #2A2A2A;
                color: #E0E0E0;
                border: 1px solid #383838;
                border-radius: 6px;
                padding: 5px 8px;
                padding-right: 20px;  /* Mehr Platz für die größeren Buttons */
                min-width: 100px;
                max-width: 400px;
                min-height: 32px;
                font-size: 13px;
            }}
            
            QDoubleSpinBox:hover, QSpinBox:hover {{
                border: 1px solid #424242;
                background-color: #2D2D2D;
            }}
            
            QDoubleSpinBox:focus, QSpinBox:focus {{
                border: 1px solid #2962FF;
                background-color: #2D2D2D;
            }}
            
            QDoubleSpinBox::up-button, QSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 25px;  /* Größere Buttons */
                height: 21px;
                border: none;
                background-color: transparent;
                margin: 1px 1px 0px 0px;
                border-top-right-radius: 5px;  /* Angepasster Radius */
            }}
            
            QDoubleSpinBox::down-button, QSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 25px;  /* Größere Buttons */
                height: 21px;
                border: none;
                background-color: transparent;
                margin: 0px 1px 1px 0px;
                border-bottom-right-radius: 5px;  /* Angepasster Radius */
            }}
            
            QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
            QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
                background-color: rgba(41, 98, 255, 0.1);
            }}
            
            QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
                image: url("{up_arrow_url}");
                width: 10px;
                height: 10px;
            }}
            
            QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
                image: url("{down_arrow_url}");
                width: 10px;
                height: 10px;
            }}
            
            QDoubleSpinBox::up-arrow:disabled, QSpinBox::up-arrow:disabled,
            QDoubleSpinBox::down-arrow:disabled, QSpinBox::down-arrow:disabled {{
                image: none;
            }}
            
            QLabel {{
                color: #E0E0E0;
                font-size: 13px;
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
                min-height: 32px;
            }}
            
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #2979FF, stop:1 #448AFF);
            }}
            
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #2962FF, stop:1 #2962FF);
            }}
            
            QTabWidget::pane {{
                border: none;
                background-color: rgba(42, 42, 42, 0.4);
                border-radius: 4px;
                margin-top: 2px;
            }}
            
            QTabWidget {{
                background-color: transparent;
            }}
            
            QTabBar {{
                background-color: transparent;
                border-bottom: none;
                border-radius: 0px;
            }}
            
            QTabBar::tab {{
                background-color: transparent;
                color: #888888;
                padding: 12px 24px;
                margin-right: 4px;
                margin-bottom: -1px;
                font-size: 13px;
                font-weight: 600;
                min-width: 120px;
                border: none;
                border-radius: 0px;
                border-top-left-radius: 0px;
                border-top-right-radius: 0px;
                border-bottom: 2px solid transparent;
            }}
            
            QTabBar::tab:hover:!selected {{
                color: #E0E0E0;
                background-color: #383838;
                border-bottom: 2px solid #404040;
            }}
            
            QTabBar::tab:selected {{
                color: #2962FF;
                background-color: #2D2D2D;
                border-bottom: 2px solid #2962FF;
            }}
            
            QTabBar::tab:selected:hover {{
                color: #2962FF;
                background-color: #383838;
                border-bottom: 2px solid #2962FF;
            }}
            
            QWidget#qt_tabwidget_stackedwidget {{
                background-color: rgba(42, 42, 42, 0.4);
                border-radius: 4px;
                padding: 2px;
            }}
            
            QCheckBox {{
                color: #E0E0E0;
                font-size: 13px;
                spacing: 8px;
            }}
            
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid #383838;
                border-radius: 4px;
                background-color: #2A2A2A;
            }}
            
            QCheckBox::indicator:hover {{
                border: 2px solid #424242;
                background-color: #2D2D2D;
            }}
            
            QCheckBox::indicator:checked {{
                background-color: #2962FF;
                border: 2px solid #2962FF;
                image: url("path/to/checkmark.png");
            }}
            
            QComboBox {{
                background-color: #2A2A2A;
                color: #E0E0E0;
                border: 1px solid #383838;
                border-radius: 6px;
                padding: 5px 8px;
                padding-right: 20px;
                min-width: 160px;
                min-height: 32px;
                font-size: 13px;
            }}
            
            QComboBox:hover {{
                border: 1px solid #424242;
                background-color: #2D2D2D;
            }}
            
            QComboBox:focus {{
                border: 1px solid #2962FF;
                background-color: #2D2D2D;
            }}
            
            QComboBox::drop-down {{
                border: none;
                width: 25px;
            }}
            
            QComboBox::down-arrow {{
                image: url("{down_arrow_url}");
                width: 10px;
                height: 10px;
            }}
            
            QComboBox::down-arrow:disabled {{
                image: none;
            }}
            
            QComboBox QAbstractItemView {{
                background-color: #2A2A2A;
                color: #E0E0E0;
                border: 1px solid #383838;
                selection-background-color: #2962FF;
                selection-color: white;
                outline: none;
                padding: 4px;
            }}
            
            QComboBox QAbstractItemView::item {{
                min-height: 24px;
                padding: 4px;
            }}
            
            QComboBox QAbstractItemView::item:hover {{
                background-color: rgba(41, 98, 255, 0.1);
            }}
            
            QComboBox QAbstractItemView::item:selected {{
                background-color: #2962FF;
            }}
        """)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Create tab widget for different parameter groups
        self.tab_widget = QTabWidget()
        
        # Add tabs
        self.tab_widget.addTab(self._create_water_pool_tab(), "Water Pool")
        self.tab_widget.addTab(self._create_cest_pool_tab(), "CEST Pool")
        self.tab_widget.addTab(self._create_scanner_tab(), "Scanner")
        self.tab_widget.addTab(self._create_advanced_tab(), "Advanced")
        
        layout.addWidget(self.tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)  # Flexibler Platz vor den Buttons
        
        # Save config button
        save_btn = QPushButton("Save Config")
        save_btn.clicked.connect(self._save_config)
        button_layout.addWidget(save_btn)
        button_layout.addSpacing(8)  # Fixer Abstand zwischen den Buttons
        
        # Reset button
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_config)
        button_layout.addWidget(reset_btn)
        button_layout.addSpacing(8)  # Fixer Abstand zwischen den Buttons
        
        # OK button
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        button_layout.addSpacing(8)  # Fixer Abstand zwischen den Buttons
        
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch(1)  # Flexibler Platz nach den Buttons
        
        layout.addLayout(button_layout)
        

    def _create_water_pool_tab(self):
        """Creates the tab for water pool parameters"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)  # Füge Außenabstände hinzu
        
        # Factors
        f_layout = QHBoxLayout()
        f_layout.setSpacing(40)
        f_label = QLabel("Relative Pool Size (f):")
        f_label.setFixedWidth(180)
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
        t1_layout.setSpacing(40)
        t1_label = QLabel("T1 [s]:")
        t1_label.setFixedWidth(180)
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
        t2_layout.setSpacing(40)
        t2_label = QLabel("T2 [s]:")
        t2_label.setFixedWidth(180)
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
        """Creates the tab for CEST pool parameters"""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(24, 24, 24, 24)
        
        # Pool selection layout with label, dropdown and +/- buttons
        pool_selection_layout = QHBoxLayout()
        pool_selection_layout.setSpacing(52)
        
        # Label
        pool_label = QLabel("CEST Pool:")
        pool_label.setFixedWidth(180)
        pool_label.setStyleSheet("padding-left: 8px;")
        # pool_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        pool_selection_layout.addWidget(pool_label)
        
        # Container for ComboBox and buttons
        combo_buttons_widget = QWidget()
        combo_buttons_container = QHBoxLayout(combo_buttons_widget)
        combo_buttons_container.setSpacing(8)
        combo_buttons_container.setContentsMargins(0, 0, 0, 0)
        # combo_buttons_container.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # ComboBox
        self.cest_pool_selection = QComboBox()
        # self.cest_pool_selection.setMinimumWidth(160)
        self.cest_pool_selection.addItem("amide")
        self.cest_pool_selection.currentTextChanged.connect(self._update_cest_pool_display)
        combo_buttons_container.addWidget(self.cest_pool_selection)
        
        # Buttons container to force layout
        button_container = QWidget()
        button_container.setFixedWidth(180)  # Genug Platz für beide Buttons mit Abstand
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(8)
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add Pool Button (+)
        add_pool_btn = IconButton("+", "Add new pool", "#4CAF50")
        add_pool_btn.clicked.connect(self._add_cest_pool)
        
        # Remove Pool Button (-)
        remove_pool_btn = IconButton("-", "Remove selected pool", "#F44336")
        remove_pool_btn.clicked.connect(self._remove_cest_pool)
        
        button_layout.addWidget(add_pool_btn)
        button_layout.addWidget(remove_pool_btn)
        combo_buttons_container.addWidget(button_container)
        
        # Add ComboBox and buttons to main layout
        pool_selection_layout.addWidget(combo_buttons_widget)
        pool_selection_layout.addStretch()
        
        main_layout.addLayout(pool_selection_layout)
        
        # Container for pool parameters
        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)
        param_layout.setSpacing(16)
        
        # Parameter fields
        # f - relative pool size
        f_layout = QHBoxLayout()
        f_layout.setSpacing(40)
        f_label = QLabel("Relative Pool Size (f):")
        f_label.setFixedWidth(180)
        self.cest_f = QDoubleSpinBox()
        self.cest_f.setRange(0.00001, 0.1)
        self.cest_f.setDecimals(6)
        self.cest_f.setSingleStep(0.00001)
        self.cest_f.valueChanged.connect(lambda val: self._update_cest_param("f", val))
        f_layout.addWidget(f_label)
        f_layout.addWidget(self.cest_f)
        param_layout.addLayout(f_layout)
        
        # T1
        t1_layout = QHBoxLayout()
        t1_layout.setSpacing(40)
        t1_label = QLabel("T1 [s]:")
        t1_label.setFixedWidth(180)
        self.cest_t1 = QDoubleSpinBox()
        self.cest_t1.setRange(0.1, 10.0)
        self.cest_t1.setSingleStep(0.1)
        self.cest_t1.valueChanged.connect(lambda val: self._update_cest_param("t1", val))
        t1_layout.addWidget(t1_label)
        t1_layout.addWidget(self.cest_t1)
        param_layout.addLayout(t1_layout)
        
        # T2
        t2_layout = QHBoxLayout()
        t2_layout.setSpacing(40)
        t2_label = QLabel("T2 [s]:")
        t2_label.setFixedWidth(180)
        self.cest_t2 = QDoubleSpinBox()
        self.cest_t2.setRange(0.001, 5.0)
        self.cest_t2.setSingleStep(0.01)
        self.cest_t2.setDecimals(3)
        self.cest_t2.valueChanged.connect(lambda val: self._update_cest_param("t2", val))
        t2_layout.addWidget(t2_label)
        t2_layout.addWidget(self.cest_t2)
        param_layout.addLayout(t2_layout)
        
        # k - exchange rate
        k_layout = QHBoxLayout()
        k_layout.setSpacing(40)
        k_label = QLabel("Exchange Rate k [Hz]:")
        k_label.setFixedWidth(180)
        self.cest_k = QDoubleSpinBox()
        self.cest_k.setRange(0, 1000)
        self.cest_k.setSingleStep(10)
        self.cest_k.valueChanged.connect(lambda val: self._update_cest_param("k", val))
        k_layout.addWidget(k_label)
        k_layout.addWidget(self.cest_k)
        param_layout.addLayout(k_layout)
        
        # dw - chemical shift
        dw_layout = QHBoxLayout()
        dw_layout.setSpacing(40)
        dw_label = QLabel("Chemical Shift dw [ppm]:")
        dw_label.setFixedWidth(180)
        self.cest_dw = QDoubleSpinBox()
        self.cest_dw.setRange(-10, 10)
        self.cest_dw.setSingleStep(0.5)
        self.cest_dw.setDecimals(2)
        self.cest_dw.valueChanged.connect(lambda val: self._update_cest_param("dw", val))
        dw_layout.addWidget(dw_label)
        dw_layout.addWidget(self.cest_dw)
        param_layout.addLayout(dw_layout)
        
        main_layout.addWidget(param_widget)
        main_layout.addStretch()
        
        # Set initial parameters
        self._update_cest_pool_display("amide")
        
        return tab

    def _create_scanner_tab(self):
        """Creates the tab for scanner settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # B0 - Field strength
        b0_layout = QHBoxLayout()
        b0_layout.setSpacing(40)
        b0_label = QLabel("B0 Field Strength [T]:")
        b0_label.setFixedWidth(180)
        self.scanner_b0 = QDoubleSpinBox()
        self.scanner_b0.setRange(1.0, 20.0)
        self.scanner_b0.setValue(self.config_params["b0"])
        self.scanner_b0.setSingleStep(0.1)
        self.scanner_b0.valueChanged.connect(lambda val: self._update_param("b0", None, val))
        b0_layout.addWidget(b0_label)
        b0_layout.addWidget(self.scanner_b0)
        layout.addLayout(b0_layout)
        
        # gamma - gyromagnetic ratio
        gamma_layout = QHBoxLayout()
        gamma_layout.setSpacing(40)
        gamma_label = QLabel("Gyromagnetic Ratio [rad/uT]:")
        gamma_label.setFixedWidth(180)
        self.scanner_gamma = QDoubleSpinBox()
        self.scanner_gamma.setRange(100.0, 300.0)
        self.scanner_gamma.setValue(self.config_params["gamma"])
        self.scanner_gamma.setSingleStep(0.1)
        self.scanner_gamma.setDecimals(4)
        self.scanner_gamma.valueChanged.connect(lambda val: self._update_param("gamma", None, val))
        gamma_layout.addWidget(gamma_label)
        gamma_layout.addWidget(self.scanner_gamma)
        layout.addLayout(gamma_layout)
        
        # b0_inhom - field inhomogeneity
        b0_inhom_layout = QHBoxLayout()
        b0_inhom_layout.setSpacing(40)
        b0_inhom_label = QLabel("B0 Inhomogeneity [ppm]:")
        b0_inhom_label.setFixedWidth(180)
        self.scanner_b0_inhom = QDoubleSpinBox()
        self.scanner_b0_inhom.setRange(0.0, 1.0)
        self.scanner_b0_inhom.setValue(self.config_params["b0_inhom"])
        self.scanner_b0_inhom.setSingleStep(0.01)
        self.scanner_b0_inhom.setDecimals(3)
        self.scanner_b0_inhom.valueChanged.connect(lambda val: self._update_param("b0_inhom", None, val))
        b0_inhom_layout.addWidget(b0_inhom_label)
        b0_inhom_layout.addWidget(self.scanner_b0_inhom)
        layout.addLayout(b0_inhom_layout)
        
        # rel_b1 - relative amplitude inhomogeneity
        rel_b1_layout = QHBoxLayout()
        rel_b1_layout.setSpacing(40)
        rel_b1_label = QLabel("Relative B1 Inhomogeneity:")
        rel_b1_label.setFixedWidth(180)
        self.scanner_rel_b1 = QDoubleSpinBox()
        self.scanner_rel_b1.setRange(0.1, 2.0)
        self.scanner_rel_b1.setValue(self.config_params["rel_b1"])
        self.scanner_rel_b1.setSingleStep(0.1)
        self.scanner_rel_b1.valueChanged.connect(lambda val: self._update_param("rel_b1", None, val))
        rel_b1_layout.addWidget(rel_b1_label)
        rel_b1_layout.addWidget(self.scanner_rel_b1)
        layout.addLayout(rel_b1_layout)
        
        layout.addStretch()
        return tab

    def _create_advanced_tab(self):
        """Creates the tab for advanced settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # verbose - detailed output
        verbose_layout = QHBoxLayout()
        verbose_layout.setSpacing(40)
        verbose_label = QLabel("Verbose Output:")
        verbose_label.setFixedWidth(180)
        self.advanced_verbose = QCheckBox()
        self.advanced_verbose.setChecked(self.config_params["verbose"])
        self.advanced_verbose.stateChanged.connect(lambda state: self._update_param("verbose", None, state == Qt.CheckState.Checked))
        verbose_layout.addWidget(verbose_label)
        verbose_layout.addWidget(self.advanced_verbose)
        layout.addLayout(verbose_layout)
        
        # reset_init_mag - reset magnetization
        reset_layout = QHBoxLayout()
        reset_layout.setSpacing(40)
        reset_label = QLabel("Reset Magnetization:")
        reset_label.setFixedWidth(180)
        self.advanced_reset = QCheckBox()
        self.advanced_reset.setChecked(self.config_params["reset_init_mag"])
        self.advanced_reset.stateChanged.connect(lambda state: self._update_param("reset_init_mag", None, state == Qt.CheckState.Checked))
        reset_layout.addWidget(reset_label)
        reset_layout.addWidget(self.advanced_reset)
        layout.addLayout(reset_layout)
        
        # scale - relative magnetization
        scale_layout = QHBoxLayout()
        scale_layout.setSpacing(40)
        scale_label = QLabel("Relative Magnetization:")
        scale_label.setFixedWidth(180)
        self.advanced_scale = QDoubleSpinBox()
        self.advanced_scale.setRange(0.1, 10.0)
        self.advanced_scale.setValue(self.config_params["scale"])
        self.advanced_scale.setSingleStep(0.1)
        self.advanced_scale.valueChanged.connect(lambda val: self._update_param("scale", None, val))
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.advanced_scale)
        layout.addLayout(scale_layout)
        
        # max_pulse_samples - maximum number of samples
        samples_layout = QHBoxLayout()
        samples_layout.setSpacing(40)
        samples_label = QLabel("Max. Pulse Samples:")
        samples_label.setFixedWidth(180)
        self.advanced_samples = QSpinBox()
        self.advanced_samples.setRange(100, 5000)
        self.advanced_samples.setValue(self.config_params["max_pulse_samples"])
        self.advanced_samples.setSingleStep(100)
        self.advanced_samples.valueChanged.connect(lambda val: self._update_param("max_pulse_samples", None, val))
        samples_layout.addWidget(samples_label)
        samples_layout.addWidget(self.advanced_samples)
        layout.addLayout(samples_layout)
        
        layout.addStretch()
        return tab

    def _update_param(self, key, param_key=None, value=None):
        """Updates a parameter in the configuration"""
        if param_key is not None:
            # For nested parameters like water_pool
            if key not in self.config_params:
                self.config_params[key] = {}
            self.config_params[key][param_key] = value
        else:
            # For direct parameters like b0, gamma etc.
            self.config_params[key] = value

    def _update_cest_param(self, param_key, value):
        """Updates a CEST pool parameter"""
        pool_name = self.cest_pool_selection.currentText()
        if "cest_pool" not in self.config_params:
            self.config_params["cest_pool"] = {}
        if pool_name not in self.config_params["cest_pool"]:
            self.config_params["cest_pool"][pool_name] = {}
        
        self.config_params["cest_pool"][pool_name][param_key] = value

    def _update_cest_pool_display(self, pool_name):
        """Updates the display of CEST pool parameters"""
        if "cest_pool" in self.config_params and pool_name in self.config_params["cest_pool"]:
            pool = self.config_params["cest_pool"][pool_name]
            self.cest_f.setValue(pool.get("f", 0.00064865))
            self.cest_t1.setValue(pool.get("t1", 1.3))
            self.cest_t2.setValue(pool.get("t2", 0.1))
            self.cest_k.setValue(pool.get("k", 50))
            self.cest_dw.setValue(pool.get("dw", 8))
        else:
            # Set default values
            self.cest_f.setValue(0.00064865)
            self.cest_t1.setValue(1.3)
            self.cest_t2.setValue(0.1)
            self.cest_k.setValue(50)
            self.cest_dw.setValue(8)
            
            # Add pool to configuration
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
        """Adds a new CEST pool"""
        pool_name, ok = QInputDialog.getText(self, "New CEST Pool", "Name of the new pool:")
        if ok and pool_name:
            if "cest_pool" not in self.config_params:
                self.config_params["cest_pool"] = {}
                
            if pool_name not in self.config_params["cest_pool"]:
                # Default pool parameters
                self.config_params["cest_pool"][pool_name] = {
                    "f": 0.00064865,
                    "t1": 1.3,
                    "t2": 0.1,
                    "k": 50,
                    "dw": 5
                }
                
                # Add pool to dropdown list
                self.cest_pool_selection.addItem(pool_name)
                self.cest_pool_selection.setCurrentText(pool_name)
            else:
                QMessageBox.warning(self, "Duplicate Pool", f"A pool with the name '{pool_name}' already exists.")

    def _remove_cest_pool(self):
        """Removes the currently selected CEST pool"""
        pool_name = self.cest_pool_selection.currentText()
        if pool_name and self.cest_pool_selection.count() > 1:
            reply = QMessageBox.question(self, 'Remove Pool', 
                                       f"Are you sure you want to remove the pool '{pool_name}'?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                        QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                # Remove pool from configuration
                if "cest_pool" in self.config_params and pool_name in self.config_params["cest_pool"]:
                    del self.config_params["cest_pool"][pool_name]
                
                # Remove pool from dropdown list
                current_index = self.cest_pool_selection.currentIndex()
                self.cest_pool_selection.removeItem(current_index)
        else:
            QMessageBox.information(self, "Information", "You must keep at least one CEST pool.")

    def _save_config(self):
        """Saves the current configuration"""
        sim_lib_path = str(Path(__file__).resolve().parent.parent.parent / "sim_lib")
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Config File",
            sim_lib_path,
            "YAML Files (*.yaml)"
        )
        if filename:
            with open(filename, 'w') as f:
                yaml.dump(self.config_params, f, default_flow_style=False)
            
            QMessageBox.information(self, "Success", f"Configuration has been saved to {os.path.basename(filename)}.")

    def _reset_config(self):
        """Resets the configuration to default values"""
        reply = QMessageBox.question(self, 'Reset', 
                                   "Are you sure you want to reset all parameters to their default values?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            # Set default values
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
            
            # Update GUI elements
            self._reset_gui_elements()
            
            QMessageBox.information(self, "Reset Complete", "All parameters have been reset to their default values.")

    def _reset_gui_elements(self):
        """Updates all GUI elements with the current values from config_params"""
        # Water pool
        self.water_f.setValue(self.config_params["water_pool"]["f"])
        self.water_t1.setValue(self.config_params["water_pool"]["t1"])
        self.water_t2.setValue(self.config_params["water_pool"]["t2"])
        
        # Reset CEST pool
        self.cest_pool_selection.clear()
        for pool_name in self.config_params["cest_pool"].keys():
            self.cest_pool_selection.addItem(pool_name)
        self.cest_pool_selection.setCurrentText(next(iter(self.config_params["cest_pool"].keys())))
        self._update_cest_pool_display(self.cest_pool_selection.currentText())
        
        # Scanner settings
        self.scanner_b0.setValue(self.config_params["b0"])
        self.scanner_gamma.setValue(self.config_params["gamma"])
        self.scanner_b0_inhom.setValue(self.config_params["b0_inhom"])
        self.scanner_rel_b1.setValue(self.config_params["rel_b1"])
        
        # Advanced settings
        self.advanced_verbose.setChecked(self.config_params["verbose"])
        self.advanced_reset.setChecked(self.config_params["reset_init_mag"])
        self.advanced_scale.setValue(self.config_params["scale"])
        self.advanced_samples.setValue(self.config_params["max_pulse_samples"])

    def get_config(self):
        """Returns the current configuration"""
        return self.config_params