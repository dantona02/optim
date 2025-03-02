from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QDoubleSpinBox, QFrame, QPushButton
)
from PyQt6.QtCore import QLocale, Qt, QSize
from PyQt6.QtGui import QIcon
from pathlib import Path
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg for PyQt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from .plot_popup_dialog import PlotPopupDialog

class CustomNavigationToolbar(NavigationToolbar):
    """Custom Navigation Toolbar with modified appearance"""
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        # Verkleinere die Toolbar-Icons
        for child in self.children():
            if isinstance(child, QWidget):  # Nur QWidget-Objekte bearbeiten
                child.setMaximumHeight(24)
                child.setMaximumWidth(24)
                if isinstance(child, QPushButton):
                    child.setIconSize(QSize(16, 16))  # Kleinere Icons
        
        # Füge den Popup-Button hinzu
        self.popup_button = self._create_popup_button()
        
        # Style der Toolbar
        self.setStyleSheet("""
            QToolBar {
                background-color: transparent;
                border: none;
                spacing: 2px;
                max-height: 28px;
            }
            QToolButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 2px;
                color: #888888;
            }
            QToolButton:hover {
                background-color: rgba(56, 56, 56, 0.6);
                border: 1px solid #404040;
                color: #E0E0E0;
            }
            QToolButton:pressed {
                background-color: rgba(45, 45, 45, 0.8);
                border: 1px solid #2962FF;
                color: #2962FF;
            }
            QToolButton[popupMode="1"] {
                padding-right: 15px;
            }
            QToolButton::menu-button {
                border: none;
            }
            QToolButton::menu-button:hover {
                background-color: rgba(56, 56, 56, 0.6);
                border: 1px solid #404040;
            }
        """)
    
    def _create_popup_button(self):
        """Erstellt einen Popup-Button für die Toolbar"""
        from PyQt6.QtWidgets import QToolButton
        from PyQt6.QtGui import QAction
        
        # Pfad zum Icon ermitteln
        icon_path = Path(__file__).resolve().parent / 'images' / 'popup.svg'
        
        # Erstelle die Aktion für den Button
        popup_action = QAction(QIcon(str(icon_path)), "Open in Popup", self)
        popup_action.setToolTip("Open plot in separate window")
        popup_action.triggered.connect(self._open_in_popup)
        
        # Füge die Aktion nach dem Save-Button ein
        # Finde zunächst den Save-Button Index
        save_button_index = None
        for i, action in enumerate(self.actions()):
            if action.text() == "Save":
                save_button_index = i
                break
        
        # Füge den neuen Button direkt nach dem Save-Button hinzu
        if save_button_index is not None:
            self.insertAction(self.actions()[save_button_index + 1], popup_action)
        else:
            self.addAction(popup_action)
        
        return popup_action
    
    def _open_in_popup(self):
        """Öffnet den aktuellen Plot in einem separaten Popup-Fenster"""
        figure = self.canvas.figure
        title = ""
        
        # Ermittle den Titel des aktuellen Tabs
        parent_widget = self.parent()
        while parent_widget:
            if isinstance(parent_widget, QTabWidget):
                break
            elif hasattr(parent_widget, 'parent'):
                parent_widget = parent_widget.parent()
            else:
                parent_widget = None
        
        # Falls der Parent ein TabWidget ist, hole den Tab-Namen als Titel
        if isinstance(parent_widget, QTabWidget):
            # Ermittle, welcher Tab das aktuelle Widget enthält
            for i in range(parent_widget.count()):
                if self.parent() == parent_widget.widget(i) or parent_widget.widget(i).isAncestorOf(self.parent()):
                    title = parent_widget.tabText(i)
                    break
        
        # Erstelle das Popup-Fenster
        dialog = PlotPopupDialog(parent=self.parent(), title=title)
        dialog.set_figure(figure)
        dialog.show()

class PlotPanel(QWidget):
    """A class representing the plot panel in the BMC Simulator GUI."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Setze das Locale für alle QDoubleSpinBox-Widgets auf Englisch
        self.locale = QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)
        self.setup_ui()
        
        # Set the style for the plot panel and tabs
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
            }
            
            QTabWidget::pane {
                border: none;
                background-color: rgba(42, 42, 42, 0.4);
                border-radius: 4px;
                margin-top: 2px;
            }
            
            QTabWidget {
                background-color: transparent;
            }
            
            QTabBar {
                background-color: transparent;
                border-bottom: none;
                border-radius: 0px;
            }
            
            QTabBar::tab {
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
            }
            
            QTabBar::tab:hover:!selected {
                color: #E0E0E0;
                background-color: #383838;
                border-bottom: 2px solid #404040;
            }
            
            QTabBar::tab:selected {
                color: #2962FF;
                background-color: #2D2D2D;
                border-bottom: 2px solid #2962FF;
            }
            
            QTabBar::tab:selected:hover {
                color: #2962FF;
                background-color: #383838;
                border-bottom: 2px solid #2962FF;
            }
            
            QWidget#qt_tabwidget_stackedwidget {
                background-color: rgba(42, 42, 42, 0.4);
                border-radius: 4px;
                padding: 2px;
            }
        """)
        
    def setup_ui(self):
        """Set up the user interface for the plot panel."""
        # Main layout for the plot panel
        plot_layout = QVBoxLayout(self)
        plot_layout.setContentsMargins(24, 24, 24, 24)
        plot_layout.setSpacing(16)
        
        # Create tabs for different plots
        self.tabs = QTabWidget()
        
        # Tab for Magnetization Plot
        mag_widget = QWidget()
        mag_layout = QVBoxLayout(mag_widget)
        mag_layout.setContentsMargins(8, 8, 8, 8)  # Reduzierte Abstände zum Rahmen
        self.mag_figure = Figure(figsize=(12, 8))  # Größerer Plot
        self.mag_canvas = FigureCanvas(self.mag_figure)
        self.mag_toolbar = CustomNavigationToolbar(self.mag_canvas, mag_widget)
        mag_layout.addWidget(self.mag_canvas)
        mag_layout.addWidget(self.mag_toolbar)  # Toolbar unter dem Plot
        self.tabs.addTab(mag_widget, "Magnetization")
        
        # Tab for Phase Plot
        phase_widget = QWidget()
        phase_layout = QVBoxLayout(phase_widget)
        phase_layout.setContentsMargins(8, 8, 8, 8)  # Reduzierte Abstände zum Rahmen
        self.phase_figure = Figure(figsize=(12, 8))  # Größerer Plot
        self.phase_canvas = FigureCanvas(self.phase_figure)
        self.phase_toolbar = CustomNavigationToolbar(self.phase_canvas, phase_widget)
        phase_layout.addWidget(self.phase_canvas)
        phase_layout.addWidget(self.phase_toolbar)  # Toolbar unter dem Plot
        self.tabs.addTab(phase_widget, "Phase")
        
        # Tab for Z-Magnetization
        mz_widget = QWidget()
        mz_layout = QVBoxLayout(mz_widget)
        mz_layout.setContentsMargins(8, 8, 8, 8)  # Reduzierte Abstände zum Rahmen
        self.mz_figure = Figure(figsize=(12, 8))  # Größerer Plot
        self.mz_canvas = FigureCanvas(self.mz_figure)
        self.mz_toolbar = CustomNavigationToolbar(self.mz_canvas, mz_widget)
        mz_layout.addWidget(self.mz_canvas)
        mz_layout.addWidget(self.mz_toolbar)  # Toolbar unter dem Plot
        self.tabs.addTab(mz_widget, "Z-Magnetization")
        
        plot_layout.addWidget(self.tabs)
    
    def plot_results(self, sim_engine):
        """Plot the simulation results."""
        if not sim_engine:
            return
            
        # Get time slices and magnetization data with get_exact()
        time_slices, magnetization_slices = sim_engine.get_exact()
        
        # Get total magnetization data for background display
        t, m_z, m_z_total, m_trans, m_trans_total = sim_engine.get_mag()
        
        # Convert to NumPy arrays
        t_np = t.cpu().numpy()
        m_z_total_np = m_z_total.cpu().numpy()
        m_trans_total_np = m_trans_total.cpu().numpy()
        m_trans_np = m_trans.cpu().numpy()
        
        # Plot Magnetization
        self.mag_figure.clear()
        ax = self.mag_figure.add_subplot(111)
        # Plot complete signal in gray
        ax.plot(t_np, abs(m_trans_total_np), '--o', markersize=2, linewidth=1, color='gray', label=r'$|M_{xy}|$')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Magnetization')
        ax.grid(True)
        ax.legend()
        self.mag_canvas.draw()
        
        # Plot Phase
        self.phase_figure.clear()
        ax = self.phase_figure.add_subplot(111)
        # Plot complete signal in gray
        ax.plot(t_np, np.angle(m_trans_np[0, :]), 'gray', alpha=0.3, label='Full Signal')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Phase (rad)')
        ax.grid(True)
        ax.legend()
        self.phase_canvas.draw()
        
        # Plot Z-Magnetization
        self.mz_figure.clear()
        ax = self.mz_figure.add_subplot(111)
        # Plot complete signal in gray
        ax.plot(t_np, m_z_total_np, 'gray', alpha=0.3, label='Full Signal')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Z-Magnetization')
        ax.grid(True)
        ax.legend()
        self.mz_canvas.draw()