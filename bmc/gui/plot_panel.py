from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel,
    QPushButton, QToolButton
)
from PyQt6.QtCore import QLocale, Qt, QSize
from PyQt6.QtGui import QIcon, QAction
from pathlib import Path
import re
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg for PyQt6
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from .plot_popup_dialog import PlotPopupDialog
from .dataset_panel import DatasetControlPanel

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
        
        # Speichere den letzten ausgewählten Button
        self.active_tool = None
        
        # Verbinde alle Action-Aktivierungen mit Highlighting
        for action in self.actions():
            action.triggered.connect(lambda checked, act=action: self._highlight_action(act))
        
        # Anpassen der Koordinatenanzeige
        self._customize_coordinates_display()
        
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
            QLabel#coordinates {
                color: #E0E0E0;
                background-color: rgba(42, 42, 42, 0.6);
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 12px;
                margin-left: 5px;
                border: 1px solid #404040;
            }
        """)
        
        # Setze das Aussehen der ausgewählten Buttons
        self._update_action_styles()

    def _normalize_number_string(self, value_str):
        """Normalisiert eine Zahl in String-Form für die Konvertierung zu float"""
        # Ersetze alle möglichen Unicode-Minuszeichen mit normalem Bindestrich
        value_str = value_str.replace('\u2212', '-').replace('−', '-')
        
        # Normalisiere wissenschaftliche Notation (e−05 -> e-05)
        value_str = re.sub(r'e[\u2212−-](\d+)', r'e-\1', value_str)
        
        # Entferne eventuelle Leerzeichen
        value_str = value_str.strip()
        
        return value_str
    
    def _customize_coordinates_display(self):
        """Passt die Koordinatenanzeige an"""
        # Entferne zuerst alle existierenden Koordinaten-Labels
        for widget in self.children():
            if isinstance(widget, QLabel) and hasattr(widget, 'text'):
                widget.setParent(None)
        
        # Erstelle ein neues Label mit verbessertem Styling
        self.coordinates_label = QLabel("")
        self.coordinates_label.setObjectName("coordinates")
        self.coordinates_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.coordinates_label.setSizePolicy(
            QToolButton().sizePolicy().horizontalPolicy(),
            QToolButton().sizePolicy().verticalPolicy()
        )
        
        # Finde den Index des Popup-Buttons
        popup_index = -1
        for i, action in enumerate(self.actions()):
            if action.text() == "Open in Popup":
                popup_index = i
                break
        
        if popup_index >= 0:
            # Entferne temporär alle Actions nach dem Popup-Button
            remaining_actions = self.actions()[popup_index + 1:]
            for action in remaining_actions:
                self.removeAction(action)
            
            # Füge das Label nach dem Popup-Button ein
            self.addWidget(self.coordinates_label)
            
            # Füge die restlichen Actions wieder hinzu
            for action in remaining_actions:
                self.addAction(action)
        else:
            # Fallback: Füge am Ende hinzu
            self.addWidget(self.coordinates_label)
        
        # Überschreibe die originale set_message Methode
        self.original_set_message = self.set_message
        self.set_message = self._custom_set_message
    
    def _custom_set_message(self, s):
        """Angepasste version der set_message Methode für die Koordinatenanzeige"""
        if not hasattr(self, 'coordinates_label'):
            self.original_set_message(s)
            return
            
        if s:
            # Für Koordinaten-Anzeige im Format (x,y) = (0.123423, 0.432534)
            if '(x, y)' in s and '=' in s:
                try:
                    # Extrahiere die Werte im Format "(x,y) = (0.123423, 0.432534)"
                    right_part = s.split('=')[1].strip()  # "(0.123423, 0.432534)"
                    
                    # Entferne Klammern und teile die Werte
                    values_part = right_part.strip('()')  # "0.123423, 0.432534"
                    values = values_part.split(',')
                    
                    if len(values) >= 2:
                        # Normalisiere die Werte für die Konvertierung
                        x_str = self._normalize_number_string(values[0].strip())
                        y_str = self._normalize_number_string(values[1].strip())
                        
                        x_value = float(x_str)
                        y_value = float(y_str)
                        
                        # Formatiere als t und M mit mehr signifikanten Stellen (.8f)
                        formatted_msg = f"t = {x_value:.8f}   M = {y_value:.8f}"
                        self.coordinates_label.setText(formatted_msg)
                        return
                except Exception as e:
                    print(f"Error parsing coordinates: {e}, raw input: '{values_part}'")
                    pass  # Falls die Extraktion fehlschlägt, verwende den Originaltext
            
            # Standard-Anzeige für alle anderen Nachrichten
            self.coordinates_label.setText(s)
        else:
            self.coordinates_label.setText("")
    
    def _highlight_action(self, action):
        """Hebt den ausgewählten Button hervor und entfernt Hervorhebung von anderen"""
        # Bei einigen Aktionen (wie Speichern) soll kein dauerhaftes Highlighting erfolgen
        non_toggle_actions = ["Save", "Home", "Open in Popup"]
        
        if action.text() in non_toggle_actions:
            # Für nicht-umschaltbare Aktionen keinen aktiven Status setzen
            return
            
        # Setze den neuen aktiven Button
        if self.active_tool == action:
            # Wenn der gleiche Button erneut geklickt wird, deaktivieren
            self.active_tool = None
        else:
            self.active_tool = action
        
        # Aktualisiere Styles für alle Actions
        self._update_action_styles()
    
    def _update_action_styles(self):
        """Aktualisiert die Styles aller Action-Buttons basierend auf dem aktiven Status"""
        for action in self.actions():
            # Finde den Button für diese Action
            for widget in self.children():
                if isinstance(widget, QToolButton) and widget.defaultAction() == action:
                    if action == self.active_tool:
                        # Hervorhebung des aktiven Buttons mit blauer Umrandung
                        widget.setStyleSheet("""
                            QToolButton {
                                background-color: rgba(41, 98, 255, 0.2);
                                border: 1px solid #2962FF;
                                border-radius: 3px;
                                color: #2962FF;
                            }
                        """)
                    else:
                        # Zurücksetzen auf normalen Stil
                        widget.setStyleSheet("")
    
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
        figure = self.canvas.figure  # Use self.canvas instead of canvas
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
        self.dataset_colors = {}  # Maps dataset names to colors
        self.color_props = {}  # Store matplotlib properties
        self.locale = QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)
        self.setup_ui()
        
        # Initialize figures with default color cycle
        self._setup_figure_colors(self.mag_figure)
        self._setup_figure_colors(self.phase_figure)
        self._setup_figure_colors(self.mz_figure)
        
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
        
        # Horizontal layout to hold plots and dataset control
        h_layout = QHBoxLayout()
        h_layout.setSpacing(0)  # Reduziere den Abstand für das Slide-out Panel
        
        # Create tabs for different plots
        self.tabs = QTabWidget()
        
        # Tab for Magnetization Plot
        mag_widget = QWidget()
        mag_layout = QVBoxLayout(mag_widget)
        mag_layout.setContentsMargins(8, 8, 8, 8)
        self.mag_figure = Figure(figsize=(12, 8))
        self.mag_canvas = FigureCanvas(self.mag_figure)
        self.mag_toolbar = CustomNavigationToolbar(self.mag_canvas, mag_widget)
        mag_layout.addWidget(self.mag_canvas)
        mag_layout.addWidget(self.mag_toolbar)
        self.tabs.addTab(mag_widget, "Magnetization")
        
        # Tab for Phase Plot
        phase_widget = QWidget()
        phase_layout = QVBoxLayout(phase_widget)
        phase_layout.setContentsMargins(8, 8, 8, 8)
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
        
        # Add tabs to horizontal layout
        h_layout.addWidget(self.tabs, stretch=1)
        
        # Create and add dataset control panel
        self.dataset_panel = DatasetControlPanel(self)
        h_layout.addWidget(self.dataset_panel)
        h_layout.setAlignment(self.dataset_panel, Qt.AlignmentFlag.AlignRight)
        
        # Add horizontal layout to main layout
        plot_layout.addLayout(h_layout)
    
    def _setup_figure_colors(self, figure):
        """Setup the color cycle for a figure and store the properties"""
        ax = figure.add_subplot(111)
        # Get the current color cycle from rcParams instead of the axes
        self.color_props = {'color': plt.rcParams['axes.prop_cycle'].by_key()['color']}
        figure.clear()  # Clear the temporary axis
        
    def _get_next_color(self, dataset_name):
        """Get next color from matplotlib's color cycle"""
        if dataset_name not in self.dataset_colors:
            colors = self.color_props['color']
            next_color = colors[len(self.dataset_colors) % len(colors)]
            self.dataset_colors[dataset_name] = next_color
        return self.dataset_colors[dataset_name]
    
    def _on_dataset_deleted(self, dataset_name):
        """Handle dataset deletion and reset counter if needed"""
        # Extract base name from dataset (remove _1, _2, etc.)
        base_name = dataset_name.split('_')[0]
        # Find the main window instance
        main_window = None
        parent = self.parent()
        while parent is not None:
            if parent.__class__.__name__ == 'BMCSimulatorGUI':
                main_window = parent
                break
            parent = parent.parent()
        
        # Reset the simulation counter if we found the main window
        if main_window is not None:
            main_window.reset_simulation_counter(base_name)
    
    def plot_results(self, sim_engine, dataset_name=None):
        """Plot the simulation results."""
        if not sim_engine:
            return
        
        # Get data
        time_slices, magnetization_slices = sim_engine.get_exact()
        t, m_z, m_z_total, m_trans, m_trans_total = sim_engine.get_mag()
        
        # Convert to NumPy arrays
        t_np = t.cpu().numpy()
        m_z_total_np = m_z_total.cpu().numpy()
        m_trans_total_np = m_trans_total.cpu().numpy()
        m_trans_np = m_trans.cpu().numpy()
        
        # Create dataset structure
        dataset = {
            't': t_np,
            'm_z': m_z_total_np,
            'm_trans': m_trans_total_np,
            'm_trans_phase': m_trans_np
        }
        
        # Add to dataset panel if name is provided
        if dataset_name:
            # Get color from matplotlib's color cycle
            dataset['color'] = self._get_next_color(dataset_name)
            self.dataset_panel.add_dataset(dataset_name, dataset)
        
        self.update_plots()
        
    def update_plots(self):
        """Update all plots with selected datasets"""
        selected_datasets = self.dataset_panel.get_selected_datasets()
        
        # Update Magnetization plot
        self.mag_figure.clear()
        ax = self.mag_figure.add_subplot(111)
        for name, data in selected_datasets:
            color = data['color']  # Use stored color for dataset
            ax.plot(data['t'], abs(data['m_trans']), '--', 
                   color=color, alpha=0.6, linewidth=1, label=f'{name}')
            ax.scatter(data['t'], abs(data['m_trans']), 
                      color=color, alpha=0.8, s=10, edgecolor=color, linewidth=1)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Magnetization')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        if selected_datasets:
            ax.legend(framealpha=0.9)
        self.mag_figure.set_tight_layout(True)
        self.mag_canvas.draw()
        
        # Update Phase plot
        self.phase_figure.clear()
        ax = self.phase_figure.add_subplot(111)
        for name, data in selected_datasets:
            color = data['color']  # Use stored color for dataset
            ax.plot(data['t'], np.angle(data['m_trans_phase'][0, :]), '--', 
                   color=color, alpha=0.6, linewidth=1, label=f'{name}')
            ax.scatter(data['t'], np.angle(data['m_trans_phase'][0, :]), 
                      color=color, alpha=0.8, s=10, edgecolor=color, linewidth=1)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Phase (rad)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        if selected_datasets:
            ax.legend(framealpha=0.9)
        self.phase_figure.set_tight_layout(True)
        self.phase_canvas.draw()
        
        # Update Z-Magnetization plot
        self.mz_figure.clear()
        ax = self.mz_figure.add_subplot(111)
        for name, data in selected_datasets:
            color = data['color']  # Use stored color for dataset
            ax.plot(data['t'], data['m_z'], '--', 
                   color=color, alpha=0.6, linewidth=1, label=f'{name}')
            ax.scatter(data['t'], data['m_z'], 
                      color=color, alpha=0.8, s=10, edgecolor=color, linewidth=1)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Z-Magnetization')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        if selected_datasets:
            ax.legend(framealpha=0.9)
        self.mz_figure.set_tight_layout(True)
        self.mz_canvas.draw()
    
    def get_dataset_color(self, dataset_name):
        """Get the color assigned to a dataset"""
        return self.dataset_colors.get(dataset_name)
    
    def reset_colors(self):
        """Reset all color-related properties to initial state"""
        self.dataset_colors.clear()  # Clear color assignments
        
        # Re-initialize the color properties from matplotlib
        self._setup_figure_colors(self.mag_figure)
        self._setup_figure_colors(self.phase_figure)
        self._setup_figure_colors(self.mz_figure)