from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt, QLocale
from PyQt6.QtGui import QIcon
from pathlib import Path
import sys
import os
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve
from PyQt6.QtWidgets import QApplication


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
        # Füge einen kleinen Abstand (5px) nach links hinzu, behalte den Rest bei 0
        layout.setContentsMargins(5, 22, 0, 0)
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
        # Entferne den linken Margin (16px) im title_layout
        title_layout.setContentsMargins(0, 0, 0, 0)
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
            
            /* Modernized styling without box-shadow */
            QWidget {{
                border: none;  /* Entferne den Standard-Rahmen */
                background-color: #2A2A2A;
                border-radius: 8px;
                border: 2px solid #303030;  /* Dunklerer Rand statt Schatten */
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
        self.box_layout.setContentsMargins(12, 12, 12, 12)  # Balanced padding
        self.box_layout.setSpacing(10)

        layout.addWidget(title_container)
        layout.addWidget(self.box)
        
        # Entferne die Zentrierung der Box, die eine Hauptursache für den Abstand ist
        # Stattdessen links ausrichten (AlignLeft)
        layout.setAlignment(self.box, Qt.AlignmentFlag.AlignLeft)
        
        # Setze das Locale für alle QDoubleSpinBox-Widgets auf Englisch
        self.locale = QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)
        
    # Überschreibe die resizeEvent-Methode, um die dynamische Breite anzupassen
    def resizeEvent(self, event):
        # Berechne die Box-Breite als relativen Wert zur Container-Breite
        # Lasse einen kleinen Rand (5px) zu den Seiten
        container_width = self.width()
        box_width = container_width - 20  # 5px Abstand nach rechts
        self.box.setFixedWidth(box_width)
        
        # Rufe die übergeordnete Methode auf
        super().resizeEvent(event)

    def addWidget(self, widget):
        self.box_layout.addWidget(widget)

    def addLayout(self, layout):
        self.box_layout.addLayout(layout)