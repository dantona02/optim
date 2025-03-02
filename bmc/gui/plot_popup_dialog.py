from PyQt6.QtWidgets import QDialog, QVBoxLayout, QWidget, QToolButton, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import re

class CustomPopupNavigationToolbar(NavigationToolbar):
    """Custom Navigation Toolbar mit modifiziertem Aussehen für das Popup-Dialog"""
    
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        
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
    
    def _customize_coordinates_display(self):
        """Passt die Koordinatenanzeige an"""
        # Suche das Label, das für die Koordinaten verwendet wird
        for widget in self.children():
            if isinstance(widget, QLabel) and hasattr(widget, 'text'):
                # Entferne das bestehende Label aus dem Layout
                widget.setParent(None)
                
                # Erstelle ein neues Label mit verbessertem Styling
                self.coordinates_label = QLabel("")
                self.coordinates_label.setObjectName("coordinates")
                self.coordinates_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.coordinates_label.setSizePolicy(
                    QToolButton().sizePolicy().horizontalPolicy(),
                    QToolButton().sizePolicy().verticalPolicy()
                )
                
                # Füge das Label direkt zur Toolbar hinzu, ohne einen Index zu verwenden
                self.addWidget(self.coordinates_label)
                
                # Überschreibe die originale set_message Methode
                self.original_set_message = self.set_message
                self.set_message = self._custom_set_message
                return
    
    def _normalize_number_string(self, value_str):
        """Normalisiert eine Zahl in String-Form für die Konvertierung zu float
        
        Behandelt spezielle Zeichen wie Unicode-Minuszeichen und wissenschaftliche Notation
        """
        # Ersetze alle möglichen Unicode-Minuszeichen mit normalem Bindestrich
        # \u2212 ist das Unicode-Minuszeichen (−)
        value_str = value_str.replace('\u2212', '-').replace('−', '-')
        
        # Normalisiere wissenschaftliche Notation (e−05 -> e-05)
        # Suche nach 'e' gefolgt von einem beliebigen Minuszeichen und Zahlen
        value_str = re.sub(r'e[\u2212−-](\d+)', r'e-\1', value_str)
        
        # Entferne eventuelle Leerzeichen
        value_str = value_str.strip()
        
        return value_str
    
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
                        
                        # Formatiere als t und M mit mehr signifikanten Stellen (.8f statt .6f)
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
        non_toggle_actions = ["Save", "Home"]
        
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

class PlotPopupDialog(QDialog):
    """Dialog zum Anzeigen eines Plots in einem separaten Fenster."""
    
    def __init__(self, parent=None, title="Plot View"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 600)  # Größere Standardgröße
        
        # Setze das Fenster auf modeless, damit es unabhängig vom Hauptfenster bedient werden kann
        self.setModal(False)
        
        # Setze das Styling
        self.setStyleSheet("""
            QDialog {
                background-color: #1E1E1E;
            }
        """)
        
        # Layout erstellen
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Figure und Canvas erstellen
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # Toolbar hinzufügen - jetzt mit der Custom-Klasse
        self.toolbar = CustomPopupNavigationToolbar(self.canvas, self)
        
        # Canvas und Toolbar zum Layout hinzufügen
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
    
    def set_figure(self, original_figure):
        """Kopiert den Inhalt einer vorhandenen Figur in diese Figur."""
        # Lösche den aktuellen Inhalt
        self.figure.clear()
        
        # Kopiere alle Achsen und Daten
        for i, ax_orig in enumerate(original_figure.get_axes()):
            # Erhalte die Position und Größe der Originalaxen
            pos = ax_orig.get_position()
            
            # Erstelle neue Axen mit der gleichen Position
            ax_new = self.figure.add_axes([pos.x0, pos.y0, pos.width, pos.height])
            
            # Kopiere alle Linien und ihre Eigenschaften
            for line in ax_orig.get_lines():
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                color = line.get_color()
                linestyle = line.get_linestyle()
                linewidth = line.get_linewidth()
                marker = line.get_marker()
                markersize = line.get_markersize()
                alpha = line.get_alpha()
                label = line.get_label()
                
                # Erstelle eine neue Linie mit den gleichen Eigenschaften
                new_line = ax_new.plot(x_data, y_data, color=color, linestyle=linestyle, 
                           linewidth=linewidth, marker=marker, markersize=markersize, 
                           label=label, alpha=alpha)[0]
                
                # Kopiere scatter plot Eigenschaften wenn vorhanden
                if isinstance(line, matplotlib.lines.Line2D) and line.get_marker() not in ['', 'None', None]:
                    new_line.set_markerfacecolor(line.get_markerfacecolor())
                    new_line.set_markeredgecolor(line.get_markeredgecolor())
                    new_line.set_markeredgewidth(line.get_markeredgewidth())
                    
            # Kopiere scatter plots separat
            for collection in ax_orig.collections:
                if isinstance(collection, matplotlib.collections.PathCollection):  # Scatter plot
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        ax_new.scatter(offsets[:, 0], offsets[:, 1],
                                     c=collection.get_facecolor(),
                                     s=collection.get_sizes(),
                                     alpha=collection.get_alpha(),
                                     edgecolor=collection.get_edgecolor())
            
            # Kopiere die Achsenbeschriftungen und Titel
            ax_new.set_xlabel(ax_orig.get_xlabel())
            ax_new.set_ylabel(ax_orig.get_ylabel())
            ax_new.set_title(ax_orig.get_title())
            
            # Kopiere die Legende, falls vorhanden
            if ax_orig.get_legend():
                ax_new.legend(framealpha=ax_orig.get_legend().get_frame().get_alpha())
            
            # Setze die Grid-Eigenschaften
            ax_new.grid(True, alpha=0.3, linestyle='--')
            
            # Kopiere die Achseneinstellungen
            ax_new.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            
            # Kopiere die Achsenbegrenzungen
            ax_new.set_xlim(ax_orig.get_xlim())
            ax_new.set_ylim(ax_orig.get_ylim())
            
            # Kopiere die Achsenspines (Ränder)
            for spine in ax_new.spines.values():
                spine.set_visible(True)
        
        # Aktualisiere das Layout
        self.figure.set_tight_layout(True)
        
        # Aktualisiere die Figur
        self.canvas.draw()