from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFrame, QSizePolicy, QScrollArea, QButtonGroup, QHBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QIcon, QPainter, QColor, QLinearGradient, QPalette, QPen, QFont, QFontMetrics
from pathlib import Path
import os


class HamburgerIcon(QWidget):
    """Custom hamburger menu icon widget with rotation animation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 48)  # Vergrößert für bessere Erkennbarkeit
        self._rotation = 0
        
        # Setup animation
        self.animation = QPropertyAnimation(self, b"rotation", self)
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate center point of the lines
        center_x = 30  # Mittelpunkt der Linien, angepasst an neue Größe
        center_y = 24  # Mittelpunkt zwischen oberster und unterster Linie
        
        # Rotate around the exact center of the three lines
        painter.translate(center_x, center_y)
        painter.rotate(self._rotation)
        painter.translate(-center_x, -center_y)
        
        # Draw the three lines
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor('#E0E0E0'))
        painter.drawRect(18, 15, 24, 2)  # Top line vergrößert
        painter.drawRect(18, 24, 24, 2)  # Middle line vergrößert
        painter.drawRect(18, 33, 24, 2)  # Bottom line vergrößert
    
    def animate_rotation(self, collapsed):
        """Animate the rotation based on collapsed state"""
        start = self._rotation
        end = 0 if collapsed else 180
        
        self.animation.setStartValue(start)
        self.animation.setEndValue(end)
        self.animation.start()
    
    @property
    def rotation(self):
        return self._rotation
    
    @rotation.setter
    def rotation(self, value):
        self._rotation = value
        self.update()  # Widget neu zeichnen

    # Property für die Animation
    rotation = pyqtProperty(float, rotation.fget, rotation.fset)


class NavigationButton(QPushButton):
    """Custom navigation button that can show icon only or icon with text."""
    
    def __init__(self, icon_path, text, parent=None):
        super().__init__(parent)
        self._text = text
        self._icon_path = icon_path
        self.collapsed = True
        self.setMinimumHeight(48)  # Höhere Buttons für bessere Erkennbarkeit
        self.setCheckable(True)
        
        # Create a custom layout for the button content
        self._init_layout()
        
        # Style for the button
        self._default_style = """
            QPushButton {
                background-color: transparent;
                color: #E0E0E0;
                padding: 0px;
                border: none;
                border-radius: 0px;
                font-weight: 500;
                font-size: 14px;
                text-align: left;
            }
            
            QPushButton:hover {
                background-color: #383838;
            }
            
            QPushButton:checked {
                background-color: #2D2D2D;
                border-left: 3px solid #2962FF;
                padding-left: 0px;
            }
            
            QLabel {
                color: #E0E0E0;
                font-size: 14px;
                font-weight: 500;
                background-color: transparent;
            }
        """
        super().setStyleSheet(self._default_style)
        self.updateAppearance()
    
    def _init_layout(self):
        """Initialize the custom layout for the button"""
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        
        # Container for the icon (fixed width of 60px)
        self._icon_container = QWidget()
        self._icon_container.setFixedWidth(60)  # Breiter für größere Icons
        self._icon_container.setStyleSheet("""
            QWidget {
                background-color: transparent;
            }
        """)
        icon_layout = QHBoxLayout(self._icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create icon label
        self._icon_label = QLabel()
        self._icon_label.setFixedSize(QSize(26, 26))  # Größere Icons
        icon = QIcon(self._icon_path)
        pixmap = icon.pixmap(QSize(26, 26))  # Passend zur neuen Größe
        self._icon_label.setPixmap(pixmap)
        icon_layout.addWidget(self._icon_label, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Container for the text
        self._text_container = QWidget()
        self._text_container.setStyleSheet("""
            QWidget {
                background-color: transparent;
            }
        """)
        text_layout = QHBoxLayout(self._text_container)
        text_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create text label
        self._text_label = QLabel(self._text)
        text_layout.addWidget(self._text_label, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Add containers to main layout
        self._layout.addWidget(self._icon_container)
        self._layout.addWidget(self._text_container)
    
    def updateAppearance(self):
        """Update the button appearance based on collapsed state"""
        if self.collapsed:
            self._text_container.hide()
            self.setFixedWidth(60)  # Breitere Buttons im eingeklappten Zustand
            self.setToolTip(self._text)
        else:
            self._text_container.show()
            self.setFixedWidth(200)  # Breitere Buttons im ausgeklappten Zustand
            self.setToolTip("")
        
        # Update margins based on checked state
        if self.isChecked():
            self._icon_container.setContentsMargins(3, 0, 0, 0)
            self._updateStyle(True)
        else:
            self._icon_container.setContentsMargins(0, 0, 0, 0)
            self._updateStyle(False)
    
    def _updateStyle(self, is_checked):
        """Update the style without causing recursion"""
        if is_checked:
            style = self._default_style + """
                QPushButton {
                    background-color: #2D2D2D;
                    border-left: 3px solid #2962FF;
                }
            """
        else:
            style = self._default_style
        super().setStyleSheet(style)
    
    def setChecked(self, checked):
        """Override setChecked to update the visual style"""
        super().setChecked(checked)
        self.updateAppearance()


class SideNavigation(QWidget):
    """Sidebar navigation component that can be collapsed and expanded."""
    
    navigationChanged = pyqtSignal(str)  # Signal emitted when a navigation item is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.collapsed = True
        self.current_button = None
        self._updating = False  # Flag to prevent recursion
        self.setupUI()
    
    def setupUI(self):
        """Setup the user interface for the sidebar."""
        self.setObjectName("sideNavigation")
        self.setFixedWidth(60)  # Initial collapsed width erhöht
        
        # Get the current directory path
        current_dir = Path(__file__).resolve().parent
        images_dir = current_dir / 'images'
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        
        # Header with hamburger menu button
        header_frame = QFrame()
        header_frame.setFixedHeight(48)  # Höher für bessere Bedienbarkeit
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #212121;
                border: none;
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Create and add the animated hamburger icon
        self.hamburger_icon = HamburgerIcon()
        self.hamburger_btn = QPushButton()
        self.hamburger_btn.setFixedSize(60, 48)  # Größerer Button
        self.hamburger_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                padding: 0px;
            }
            
            QPushButton:hover {
                background-color: #383838;
            }
            
            QPushButton:pressed {
                background-color: #404040;
            }
        """)
        self.hamburger_btn.clicked.connect(self.toggleCollapse)
        
        # Position the icon directly in the button without additional layout
        self.hamburger_icon.setParent(self.hamburger_btn)
        
        header_layout.addWidget(self.hamburger_btn)
        layout.addWidget(header_frame)
        
        # Create a button group to manage exclusive selection
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        
        # Scroll area for navigation items with improved style
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #222222;
                border: none;
            }
            QScrollBar:vertical {
                border: none;
                background: #2A2A2A;
                width: 6px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #4D4D4D;
                min-height: 20px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background: #666666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
        """)
        
        # Container for navigation items
        nav_container = QWidget()
        nav_container.setStyleSheet("""
            background-color: #222222;
        """)
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 4, 0, 4)  # Größere Abstände oben und unten
        nav_layout.setSpacing(2)  # Etwas mehr Abstand zwischen Buttons
        
        # Add navigation buttons
        self.nav_buttons = {}
        
        # Simulation button
        main_btn = NavigationButton(str(images_dir / 'wave.png'), "Simulation")
        self.button_group.addButton(main_btn)
        main_btn.clicked.connect(lambda: self._on_button_clicked("simulation"))
        nav_layout.addWidget(main_btn)
        self.nav_buttons["simulation"] = main_btn
        
        # PulseSeq button
        pulseseq_btn = NavigationButton(str(images_dir / 'pulseq_logo.png'), "PulSeq")
        self.button_group.addButton(pulseseq_btn)
        pulseseq_btn.clicked.connect(lambda: self._on_button_clicked("pulseseq"))
        nav_layout.addWidget(pulseseq_btn)
        self.nav_buttons["pulseseq"] = pulseseq_btn
        
        # Animation button
        animation_btn = NavigationButton(str(images_dir / 'play.png'), "Animation")
        self.button_group.addButton(animation_btn)
        animation_btn.clicked.connect(lambda: self._on_button_clicked("animation"))
        nav_layout.addWidget(animation_btn)
        self.nav_buttons["animation"] = animation_btn
        
        # Settings button
        settings_btn = NavigationButton(str(images_dir / 'settings.png'), "Settings")
        self.button_group.addButton(settings_btn)
        settings_btn.clicked.connect(lambda: self._on_button_clicked("settings"))
        nav_layout.addWidget(settings_btn)
        self.nav_buttons["settings"] = settings_btn
        
        # Add stretch before the about button to push it to bottom
        nav_layout.addStretch()
        
        # About button at the bottom
        about_btn = NavigationButton(str(images_dir / 'information.png'), "About")
        self.button_group.addButton(about_btn)
        about_btn.clicked.connect(lambda: self._on_button_clicked("about"))
        nav_layout.addWidget(about_btn)
        self.nav_buttons["about"] = about_btn
        
        scroll_area.setWidget(nav_container)
        layout.addWidget(scroll_area)
        
        # Erstellen eines dedizierten rechten Border-Frames
        right_border = QFrame(self)
        right_border.setFixedWidth(1)
        right_border.setStyleSheet("background-color: #444444;")  # Dunklere Graufarbe
        right_border.setGeometry(self.width()-1, 0, 1, self.height())
        
        # Stellen Sie sicher, dass sich der Border bei Größenänderung anpasst
        self.resizeEvent = lambda e: right_border.setGeometry(self.width()-1, 0, 1, self.height())
        
        # Set the sidebar's style with border on the right side (als Backup)
        self.setStyleSheet("""
            #sideNavigation {
                background-color: #222222;
                border-right: 1px solid #444444;
            }
        """)
        
        # Set simulation as default selected
        self._on_button_clicked("simulation")
    
    def _on_button_clicked(self, page_name):
        """Handle button clicks and emit navigation signal"""
        if self._updating or page_name not in self.nav_buttons:
            return
            
        self._updating = True
        try:
            for name, button in self.nav_buttons.items():
                if name != page_name:
                    button.setChecked(False)
                    button._updateStyle(False)
            
            # Aktiviere den ausgewählten Button
            self.current_button = self.nav_buttons[page_name]
            self.current_button.setChecked(True)
            self.current_button._updateStyle(True)
            
            # Emit navigation signal
            self.navigationChanged.emit(page_name)
        finally:
            self._updating = False
    
    def toggleCollapse(self):
        """Toggle between collapsed and expanded state."""
        self.collapsed = not self.collapsed
        
        # Animate hamburger icon rotation
        self.hamburger_icon.animate_rotation(self.collapsed)
        
        # Update navigation buttons
        for button in self.nav_buttons.values():
            button.collapsed = self.collapsed
            button.updateAppearance()
        
        # Animate width change
        self.animation = QPropertyAnimation(self, b"minimumWidth")
        self.animation.setDuration(300)  # Match rotation duration
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        
        if self.collapsed:
            self.animation.setStartValue(200)  # Breitere ausgeklappte Ansicht
            self.animation.setEndValue(60)    # Breitere eingeklappte Ansicht
        else:
            self.animation.setStartValue(60)   # Breitere eingeklappte Ansicht
            self.animation.setEndValue(200)   # Breitere ausgeklappte Ansicht
        
        self.animation.start()
    
    def setCurrentPage(self, page_name):
        """Set the current selected page in the navigation."""
        if not self._updating and page_name in self.nav_buttons:
            self._on_button_clicked(page_name)
    
    @pyqtProperty(int)
    def minimumWidth(self):
        return super().minimumWidth()
    
    @minimumWidth.setter
    def minimumWidth(self, width):
        self.setFixedWidth(width)