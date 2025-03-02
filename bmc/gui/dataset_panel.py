from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QPushButton,
    QToolButton, QLabel, QScrollArea, QCheckBox, QWidget,
    QGraphicsOpacityEffect
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QEasingCurve, QPoint, QPropertyAnimation
from PyQt6.QtGui import QIcon, QTransform
from pathlib import Path

class DatasetControlPanel(QFrame):
    """A panel for controlling dataset visibility"""
    # Signal when a dataset is deleted
    dataset_deleted = pyqtSignal(str)  # Emits the name of the deleted dataset

    def __init__(self, parent=None):
        super().__init__(parent)
        self.datasets = {}  # {name: (checkbox, delete_button, data)}
        self.is_expanded = False
        self.normal_width = 200
        self.collapsed_width = 24
        self.setup_ui()
        
        # Style
        self.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border: none;
            }
            QFrame#contentFrame {
                background-color: #2D2D2D;
                border: 1px solid #404040;
                border-radius: 4px;
            }
            QFrame#collapsedFrame {
                background-color: transparent;
                border: none;
                border-radius: 0;
            }
            QCheckBox {
                color: #E0E0E0;
                padding: 4px 8px;
                margin: 2px;
            }
            QCheckBox:hover {
                background-color: rgba(56, 56, 56, 0.6);
                border-radius: 3px;
            }
            QCheckBox:checked {
                color: #2962FF;
            }
            QPushButton {
                background-color: #2D2D2D;
                border: 1px solid #404040;
                border-radius: 3px;
                color: #E0E0E0;
                padding: 4px 12px;
                margin: 4px;
            }
            QPushButton:hover {
                background-color: rgba(56, 56, 56, 0.6);
                border: 1px solid #2962FF;
                color: #2962FF;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 2px;
                margin: 2px;
            }
            QToolButton#toggleButton {
                background-color: transparent;
                border: none;
                border-radius: 3px;
                padding: 4px;
            }
            QToolButton#toggleButton:hover {
                background-color: rgba(56, 56, 56, 0.6);
            }
            QToolButton#deleteButton:hover {
                background-color: rgba(224, 32, 32, 0.2);
                border-radius: 3px;
            }
            QLabel {
                color: #E0E0E0;
                font-weight: bold;
                padding: 4px;
                border: none;
                background: transparent;
            }
            QLabel#headerLabel {
                border: none;
                background: transparent;
                margin-top: 4px;
                margin-bottom: 4px;
            }
        """)
    
    def setup_ui(self):
        """Set up the UI components"""
        # Main layout
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Toggle button container (always visible)
        self.toggle_container = QFrame(self)
        self.toggle_container.setObjectName("collapsedFrame")
        self.toggle_container.setFixedWidth(self.collapsed_width)
        toggle_layout = QVBoxLayout(self.toggle_container)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(0)
        
        # Toggle button
        self.toggle_button = QToolButton()
        self.toggle_button.setObjectName("toggleButton")
        self.toggle_button.setIcon(QIcon(str(Path(__file__).resolve().parent / 'images' / 'chevron-left.svg')))
        self.toggle_button.setFixedSize(24, 24)
        self.toggle_button.clicked.connect(self.toggle_panel)
        toggle_layout.addWidget(self.toggle_button)
        toggle_layout.addStretch()
        
        # Content container (expandable)
        self.content_container = QFrame(self)
        self.content_container.setObjectName("contentFrame")
        content_layout = QVBoxLayout(self.content_container)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(4)
        
        # Header
        header = QLabel("Datasets")
        header.setObjectName("headerLabel")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(header)
        
        # Clear button
        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_all_datasets)
        content_layout.addWidget(self.clear_button)
        
        # Scroll area for datasets
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #2D2D2D;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #404040;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        # Container for checkboxes
        self.checkbox_container = QWidget()
        self.checkbox_layout = QVBoxLayout(self.checkbox_container)
        self.checkbox_layout.setContentsMargins(4, 4, 4, 4)
        self.checkbox_layout.setSpacing(2)
        self.checkbox_layout.addStretch()
        
        scroll.setWidget(self.checkbox_container)
        content_layout.addWidget(scroll)
        
        # Add containers to main layout
        self.main_layout.addWidget(self.toggle_container)
        self.main_layout.addWidget(self.content_container)
        
        # Initialize in collapsed state
        self.content_container.setFixedWidth(0)
        self.setFixedWidth(self.collapsed_width)
        
        # Create animation
        self.animation = QPropertyAnimation(self, b"minimumWidth")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        self.content_animation = QPropertyAnimation(self.content_container, b"minimumWidth")
        self.content_animation.setDuration(200)
        self.content_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    
    def toggle_panel(self):
        """Toggle the panel between expanded and collapsed states"""
        if self.animation.state() == QPropertyAnimation.State.Running:
            return
            
        if self.is_expanded:
            # Collapse
            self.animation.setStartValue(self.normal_width + self.collapsed_width)
            self.animation.setEndValue(self.collapsed_width)
            self.content_animation.setStartValue(self.normal_width)
            self.content_animation.setEndValue(0)
            # Rotate icon back
            transform = QTransform()
            self.toggle_button.setIcon(QIcon(str(Path(__file__).resolve().parent / 'images' / 'chevron-left.svg')))
        else:
            # Expand
            self.animation.setStartValue(self.collapsed_width)
            self.animation.setEndValue(self.normal_width + self.collapsed_width)
            self.content_animation.setStartValue(0)
            self.content_animation.setEndValue(self.normal_width)
            # Rotate icon 180 degrees
            transform = QTransform()
            transform.rotate(180)
            icon = QIcon(str(Path(__file__).resolve().parent / 'images' / 'chevron-left.svg'))
            pixmap = icon.pixmap(QSize(24, 24))
            rotated_icon = QIcon(pixmap.transformed(transform))
            self.toggle_button.setIcon(rotated_icon)
        
        self.animation.start()
        self.content_animation.start()
        self.is_expanded = not self.is_expanded
    
    def create_dataset_row(self, name):
        """Create a row with checkbox and delete button"""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        
        # Checkbox
        checkbox = QCheckBox(name)
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(self.update_plots)
        
        # Delete button
        delete_button = QToolButton()
        delete_button.setObjectName("deleteButton")
        delete_button.setIcon(QIcon(str(Path(__file__).resolve().parent / 'images' / 'trash.svg')))
        delete_button.setFixedSize(20, 20)
        delete_button.clicked.connect(lambda: self.delete_dataset(name))
        
        # Add widgets to row
        row_layout.addWidget(checkbox, stretch=1)
        row_layout.addWidget(delete_button)
        
        return row_widget, checkbox, delete_button
    
    def add_dataset(self, name, data):
        """Add a new dataset to the panel"""
        if name not in self.datasets:
            row_widget, checkbox, delete_button = self.create_dataset_row(name)
            
            # Insert before the stretch
            self.checkbox_layout.insertWidget(self.checkbox_layout.count() - 1, row_widget)
            self.datasets[name] = (checkbox, delete_button, data)
    
    def delete_dataset(self, name):
        """Delete a specific dataset"""
        if name in self.datasets:
            checkbox, delete_button, _ = self.datasets[name]
            # Get the parent widget (row_widget)
            row_widget = checkbox.parent()
            # Remove from layout and delete
            row_widget.setParent(None)
            row_widget.deleteLater()
            del self.datasets[name]
            # Emit signal that dataset was deleted to update counter
            self.dataset_deleted.emit(name)
            self.update_plots()
    
    def clear_all_datasets(self):
        """Remove all datasets"""
        dataset_names = list(self.datasets.keys())
        for name, (checkbox, _, _) in list(self.datasets.items()):
            # Get the parent widget (row_widget)
            row_widget = checkbox.parent()
            # Remove from layout and delete
            row_widget.setParent(None)
            row_widget.deleteLater()
        self.datasets.clear()
        # Emit signal that all datasets were deleted
        for name in dataset_names:
            self.dataset_deleted.emit(name)
        self.update_plots()
    
    def get_selected_datasets(self):
        """Return a list of selected datasets"""
        return [(name, data) for name, (checkbox, _, data) in self.datasets.items() 
                if checkbox.isChecked()]
    
    def update_plots(self):
        """Trigger plot update in the parent panel"""
        if hasattr(self.parent(), 'update_plots'):
            self.parent().update_plots()