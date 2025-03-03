from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QHBoxLayout
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtProperty, QTimer, QParallelAnimationGroup, QPointF, QRectF
from PyQt6.QtGui import QPainter, QLinearGradient, QColor, QPen
from pathlib import Path

class LoadingBar(QFrame):
    def __init__(self, parent=None, delay=0):
        super().__init__(parent)
        self.setFixedSize(14, 40)  # Erhöhe die Höhe für mehr Platz zum Wachsen
        self._current_height = 32
        self._current_opacity = 0.75
        self._y_offset = 4  # Start in der Mitte
        self.delay = delay
        self.setupAnimation()
        
    def setupAnimation(self):
        # Height animation - größerer Unterschied zwischen min und max
        self.height_anim = QPropertyAnimation(self, b"barHeight")
        self.height_anim.setDuration(800)
        self.height_anim.setLoopCount(-1)
        self.height_anim.setKeyValues([
            (0.0, 32),    # Normalhöhe
            (0.4, 40),    # Maximale Höhe
            (0.8, 32),    # Zurück zur Normalhöhe
            (1.0, 32)     # Bleibe bei Normalhöhe
        ])
        self.height_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Y-Position animation für symmetrisches Wachsen
        self.y_anim = QPropertyAnimation(self, b"yOffset")
        self.y_anim.setDuration(800)
        self.y_anim.setLoopCount(-1)
        self.y_anim.setKeyValues([
            (0.0, 4),     # Startposition
            (0.4, 0),     # Position bei maximaler Höhe (zentriert)
            (0.8, 4),     # Zurück zur Startposition
            (1.0, 4)      # Bleibe bei Startposition
        ])
        self.y_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Opacity animation - stärkerer Kontrast
        self.opacity_anim = QPropertyAnimation(self, b"opacity")
        self.opacity_anim.setDuration(800)
        self.opacity_anim.setLoopCount(-1)
        self.opacity_anim.setKeyValues([
            (0.0, 0.65),    # Start mit niedrigerer Deckkraft
            (0.4, 1.0),     # Volle Deckkraft bei maximaler Höhe
            (0.8, 0.65),    # Zurück zu niedrigerer Deckkraft
            (1.0, 0.65)     # Bleibe bei niedrigerer Deckkraft
        ])
        self.opacity_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Group animations
        self.animation_group = QParallelAnimationGroup()
        self.animation_group.addAnimation(self.height_anim)
        self.animation_group.addAnimation(self.y_anim)
        self.animation_group.addAnimation(self.opacity_anim)
        
        # Start with delay if specified
        if self.delay > 0:
            QTimer.singleShot(self.delay, self.animation_group.start)
        else:
            self.animation_group.start()
            
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Create gradient
        gradient = QLinearGradient(QPointF(0, 0), QPointF(0, self._current_height))
        gradient.setColorAt(0.0, QColor(7, 111, 229, int(255 * self._current_opacity)))    # #076fe5
        gradient.setColorAt(0.5, QColor(36, 107, 253, int(255 * self._current_opacity)))   # #246bfd
        gradient.setColorAt(1.0, QColor(7, 111, 229, int(255 * self._current_opacity)))    # #076fe5
        
        # Draw the bar with rounded corners
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        rect = QRectF(0, self._y_offset, self.width(), self._current_height)
        painter.drawRoundedRect(rect, 2, 2)
        
    @pyqtProperty(float)
    def barHeight(self):
        return self._current_height
        
    @barHeight.setter
    def barHeight(self, height):
        self._current_height = height
        self.update()
        
    @pyqtProperty(float)
    def yOffset(self):
        return self._y_offset
        
    @yOffset.setter
    def yOffset(self, offset):
        self._y_offset = offset
        self.update()
        
    @pyqtProperty(float)
    def opacity(self):
        return self._current_opacity
        
    @opacity.setter
    def opacity(self, value):
        self._current_opacity = value
        self.update()

class LoadingAnimation(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("loader")
        self.setFixedSize(100, 50)  # Reduzierte Höhe für bessere Proportionen
        self.setupUI()
        
    def setupUI(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)  # Kleinerer Abstand zwischen den Balken
        
        # Create three loading bars with progressive delays
        self.bars = [
            LoadingBar(self, delay=0),    # First bar
            LoadingBar(self, delay=160),  # Second bar delayed by 160ms
            LoadingBar(self, delay=320)   # Third bar delayed by 320ms
        ]
        
        # Center the bars in the layout
        layout.addStretch()
        for bar in self.bars:
            layout.addWidget(bar, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch()

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
        
        # Create loading animation
        loading_animation = LoadingAnimation(self)
        
        # Add widgets to layout
        layout.addWidget(title)
        layout.addWidget(version)
        layout.addWidget(description)
        layout.addWidget(loading_animation, 0, Qt.AlignmentFlag.AlignCenter)