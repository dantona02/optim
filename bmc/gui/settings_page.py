from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QPainter, QPen, QColor

class LoadingSpinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(52, 52)
        self._angle = 0
        self._dash_length = 1
        
        # Rotation animation - slightly faster
        self.rotation_animation = QPropertyAnimation(self, b"angle")
        self.rotation_animation.setDuration(1500)  # Schneller rotieren
        self.rotation_animation.setStartValue(0)
        self.rotation_animation.setEndValue(360)
        self.rotation_animation.setLoopCount(-1)
        
        # Dash length animation - smoother
        self.dash_animation = QPropertyAnimation(self, b"dashLength")
        self.dash_animation.setDuration(1500)
        self.dash_animation.setStartValue(1)
        self.dash_animation.setEndValue(90)  # Kürzere Dash-Länge für besseren Effekt
        self.dash_animation.setLoopCount(-1)
        self.dash_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)  # Sanftere Beschleunigung
        
        # Start animations
        self.rotation_animation.start()
        self.dash_animation.start()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Setup pen with dynamic dash pattern
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(QColor("#2962FF"))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)  # Runde Enden für weicheren Look
        pen.setDashPattern([self._dash_length, 150 - self._dash_length])  # Angepasstes Pattern
        painter.setPen(pen)
        
        # Apply rotation
        painter.translate(26, 26)
        painter.rotate(self._angle)
        painter.translate(-26, -26)
        
        # Draw circle
        painter.drawEllipse(4, 4, 44, 44)  # Etwas kleiner für bessere Proportionen
    
    @pyqtProperty(float)
    def angle(self):
        return self._angle
    
    @angle.setter
    def angle(self, value):
        self._angle = value
        self.update()
    
    @pyqtProperty(float)
    def dashLength(self):
        return self._dash_length
    
    @dashLength.setter
    def dashLength(self, value):
        self._dash_length = value
        self.update()

class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create and add loading spinner
        self.spinner = LoadingSpinner()
        layout.addWidget(self.spinner)