from PyQt6.QtWidgets import QDialog, QVBoxLayout
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

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
        
        # Toolbar hinzufügen
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("""
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
        """)
        
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
            
            # Kopiere alle Linien und ihre Daten
            for line in ax_orig.get_lines():
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                color = line.get_color()
                linestyle = line.get_linestyle()
                linewidth = line.get_linewidth()
                marker = line.get_marker()
                markersize = line.get_markersize()
                label = line.get_label()
                
                # Erstelle eine neue Linie mit den gleichen Eigenschaften
                ax_new.plot(x_data, y_data, color=color, linestyle=linestyle, 
                           linewidth=linewidth, marker=marker, markersize=markersize, 
                           label=label)
            
            # Kopiere die Achsenbeschriftungen und Titel
            ax_new.set_xlabel(ax_orig.get_xlabel())
            ax_new.set_ylabel(ax_orig.get_ylabel())
            ax_new.set_title(ax_orig.get_title())
            
            # Kopiere die Legende, falls vorhanden
            if ax_orig.get_legend():
                ax_new.legend()
            
            # Kopiere die Gitterlinien
            # In Matplotlib gibt es keine get_grid() Methode
            # Stattdessen prüfen wir die Sichtbarkeit der vorhandenen Grid-Linien
            ax_new.grid(ax_orig.xaxis._major_tick_kw.get('gridOn', False) or 
                       ax_orig.yaxis._major_tick_kw.get('gridOn', False))
            
            # Kopiere die Achsenbegrenzungen
            ax_new.set_xlim(ax_orig.get_xlim())
            ax_new.set_ylim(ax_orig.get_ylim())
        
        # Aktualisiere die Figur
        self.canvas.draw()