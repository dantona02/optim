from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

class AboutPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()
    
    def setupUI(self):
        """Setup the user interface for the about page"""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add about information
        title = QLabel("BMC Simulator")
        title.setStyleSheet("""
            QLabel {
                color: #E0E0E0;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 20px;
            }
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        version = QLabel("Version 1.0.0")
        version.setStyleSheet("""
            QLabel {
                color: #BBBBBB;
                font-size: 16px;
                margin-bottom: 40px;
            }
        """)
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        description = QLabel(
            "BMC Simulator is a tool for simulating Magnetic Resonance Imaging (MRI) "
            "sequences with special focus on CEST (Chemical Exchange Saturation Transfer) "
            "effects. It provides a modern interface for sequence development, "
            "optimization, and analysis."
        )
        description.setStyleSheet("""
            QLabel {
                color: #E0E0E0;
                font-size: 14px;
                line-height: 1.6;
                margin-bottom: 20px;
                max-width: 600px;
            }
        """)
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add widgets to layout
        layout.addWidget(title)
        layout.addWidget(version)
        layout.addWidget(description)