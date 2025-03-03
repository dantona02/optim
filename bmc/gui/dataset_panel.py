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
    dataset_deleted = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.datasets = {}
        self.is_expanded = False
        self.normal_width = 220
        self.collapsed_width = 20
        self.parent_plot_panel = parent  # Store reference to plot panel
        self.setup_ui()
        
        # Style with dynamic color for checkboxes
        self.setStyleSheet("""
            /* Main Frame */
            QFrame {
                background-color: transparent;
                border: none;
            }
            
            /* Content Frame */
            QFrame#contentFrame {
                background-color: transparent;
                border: 1px solid #404040;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
            }
            
            /* Toggle Button Frame */
            QFrame#toggleFrame {
                background-color: #2D2D2D;
                border-top: 1px solid #404040;
                border-bottom: 1px solid #404040;
                border-left: 1px solid #404040;
                border-right: none;
                border-top-left-radius: 4px;
                border-bottom-left-radius: 4px;
                padding: 0;
                margin: 0;
            }
            
            /* Toggle Button */
            QToolButton#toggleButton {
                background-color: transparent;
                border: none;
                border-radius: 0;
                padding: 0;
                margin: 0;
                margin-left: 2px; /* Zentriert das Icon besser */
                height: 28px;
            }
            QToolButton#toggleButton:hover {
                background-color: rgba(41, 98, 255, 0.1);
            }
            QToolButton#toggleButton:pressed {
                background-color: rgba(41, 98, 255, 0.2);
            }
            
            /* Header */
            QLabel#headerLabel {
                color: #888888;
                font-size: 11px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 1px;
                padding: 12px 8px 8px 8px;
                border: none;
                background: transparent;
            }
            
            /* Clear All Button */
            QPushButton#clearButton {
                background-color: transparent;
                border: 1px solid #404040;
                border-radius: 3px;
                color: #888888;
                padding: 6px 12px;
                margin: 8px;
                margin-top: 0px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton#clearButton:hover {
                background-color: rgba(41, 98, 255, 0.1);
                border: 1px solid #2962FF;
                color: #2962FF;
            }
            QPushButton#clearButton:pressed {
                background-color: rgba(41, 98, 255, 0.2);
            }
            
            /* Dataset Row Container */
            QWidget#datasetRow {
                background-color: transparent;
                border-radius: 3px;
                margin: 0px;
                padding: 0px;
            }
            
            QWidget#datasetRow:hover {
                background-color: rgba(56, 56, 56, 0.5);
            }
            
            /* Dataset Checkboxes */
            QCheckBox {
                padding: 6px 8px;
                margin: 1px 0px;
                border-radius: 3px;
                font-size: 12px;
                background-color: transparent;
            }
            
            QCheckBox:unchecked {
                color: #E0E0E0;
            }
            
            /* Delete Button */
            QToolButton#deleteButton {
                background-color: transparent;
                border: none;
                border-radius: 3px;
                padding: 3px;
                margin-right: 2px;
                min-width: 28px;
                max-width: 28px;
                min-height: 28px;
                max-height: 28px;
                opacity: 0.7;
            }
            
            QToolButton#deleteButton:hover {
                background-color: rgba(224, 32, 32, 0.1);
                opacity: 1;
            }
            
            QToolButton#deleteButton:pressed {
                background-color: rgba(224, 32, 32, 0.2);
            }
            
            /* Scroll Area */
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
            
            /* Scrollbar */
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 4px;  /* Schmaler */
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(64, 64, 64, 0.5);  /* Semi-transparent */
                min-height: 20px;
                border-radius: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(77, 77, 77, 0.7);  /* Etwas dunkler beim Hover */
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0;  /* Verstecke die Buttons komplett */
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
            }
            
            /* Dataset Container */
            QWidget#datasetContainer {
                background-color: transparent;
                border: none;
            }
        """)
    
    def setup_ui(self):
        """Set up the UI components"""
        # Main layout
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)  # Kein Abstand zwischen Toggle und Content
        
        # Toggle button container
        self.toggle_container = QFrame(self)
        self.toggle_container.setObjectName("toggleFrame")
        self.toggle_container.setFixedWidth(self.collapsed_width)
        toggle_layout = QVBoxLayout(self.toggle_container)
        toggle_layout.setContentsMargins(0, 0, 0, 0)  # Keine Innenabstände
        toggle_layout.setSpacing(0)
        
        # Toggle button
        self.toggle_button = QToolButton()
        self.toggle_button.setObjectName("toggleButton")
        self.toggle_button.setIcon(QIcon(str(Path(__file__).resolve().parent / 'images' / 'chevron-left.svg')))
        self.toggle_button.setIconSize(QSize(16, 16))
        self.toggle_button.setFixedSize(20, 28)
        self.toggle_button.clicked.connect(self.toggle_panel)
        toggle_layout.addWidget(self.toggle_button, 0, Qt.AlignmentFlag.AlignTop)  # Align top
        toggle_layout.addStretch()
        
        # Content container with refined styling
        self.content_container = QFrame(self)
        self.content_container.setObjectName("contentFrame")
        content_layout = QVBoxLayout(self.content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Header with new styling
        header = QLabel("DATASETS")
        header.setObjectName("headerLabel")
        header.setAlignment(Qt.AlignmentFlag.AlignLeft)
        content_layout.addWidget(header)
        
        # Scroll area with refined appearance
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Dataset container mit angepasstem Styling
        self.checkbox_container = QWidget()
        self.checkbox_container.setObjectName("datasetContainer")
        self.checkbox_layout = QVBoxLayout(self.checkbox_container)
        self.checkbox_layout.setContentsMargins(6, 0, 6, 8)
        self.checkbox_layout.setSpacing(4)  # Optimaler Abstand zwischen den Reihen
        self.checkbox_layout.addStretch()
        
        scroll.setWidget(self.checkbox_container)
        content_layout.addWidget(scroll)
        
        # Clear button with new styling
        self.clear_button = QPushButton("Clear All")
        self.clear_button.setObjectName("clearButton")
        self.clear_button.clicked.connect(self.clear_all_datasets)
        content_layout.addWidget(self.clear_button)
        
        # Add containers to main layout
        self.main_layout.addWidget(self.toggle_container)
        self.main_layout.addWidget(self.content_container)
        
        # Initialize in collapsed state
        self.content_container.setFixedWidth(0)
        self.setFixedWidth(self.collapsed_width)
        
        # Animations
        self.animation = QPropertyAnimation(self, b"minimumWidth")
        self.animation.setDuration(150)  # Schnellere Animation
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        self.content_animation = QPropertyAnimation(self.content_container, b"minimumWidth")
        self.content_animation.setDuration(150)  # Schnellere Animation
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
        # Create main row widget
        row_widget = QWidget()
        row_widget.setObjectName("datasetRow")
        
        # Create row layout with better spacing and alignment
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(2, 2, 2, 2)
        row_layout.setSpacing(6)
        
        # Create checkbox with color from plot panel
        checkbox = QCheckBox(name)
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(lambda state, cb=checkbox: self.on_checkbox_state_changed(state, cb))
        
        # Get color from parent plot panel and apply to checkbox
        if self.parent_plot_panel and self.parent_plot_panel.get_dataset_color(name):
            color = self.parent_plot_panel.get_dataset_color(name)
            checkbox.setStyleSheet(f"""
                QCheckBox {{
                    color: {color};
                }}
                QCheckBox:checked {{
                    color: {color};
                }}
                QCheckBox:unchecked {{
                    color: #888888;
                }}
            """)
        
        # Add checkbox to row
        row_layout.addWidget(checkbox)
        
        # Create delete button with improved visibility
        delete_button = QToolButton()
        delete_button.setIcon(QIcon(str(Path(__file__).resolve().parent / "images" / "trash.svg")))
        delete_button.setObjectName("deleteButton")
        delete_button.setIconSize(QSize(14, 14))
        delete_button.clicked.connect(lambda: self.remove_dataset(name))
        
        # Add delete button to row
        row_layout.addWidget(delete_button)
        row_layout.setAlignment(delete_button, Qt.AlignmentFlag.AlignRight)
        
        # Return tuple of components
        return (row_widget, checkbox, delete_button)

    def on_checkbox_state_changed(self, state, checkbox):
        """Handle checkbox state changes"""
        self.update_plots()
    
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
    
    def remove_dataset(self, name):
        """Remove a dataset from the panel"""
        self.delete_dataset(name)
    
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
        
        # Reset color properties in parent plot panel
        if self.parent_plot_panel and hasattr(self.parent_plot_panel, 'reset_colors'):
            self.parent_plot_panel.reset_colors()
        
        # Find main window to reset simulation counters
        main_window = None
        parent = self.parent_plot_panel
        while parent is not None:
            if parent.__class__.__name__ == 'BMCSimulatorGUI':
                main_window = parent
                break
            parent = parent.parent()
        
        # Reset all simulation counters if main window is found
        if main_window and hasattr(main_window, 'reset_all_simulation_counters'):
            main_window.reset_all_simulation_counters()
        
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