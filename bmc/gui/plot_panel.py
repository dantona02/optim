from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget
)
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg for PyQt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class PlotPanel(QWidget):
    """A class representing the plot panel in the BMC Simulator GUI."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface for the plot panel."""
        # Main layout for the plot panel
        plot_layout = QVBoxLayout(self)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tabs for different plots
        self.tabs = QTabWidget()
        
        # Tab for Magnetization Plot
        mag_widget = QWidget()
        mag_layout = QVBoxLayout(mag_widget)
        self.mag_figure = Figure(figsize=(8, 6))
        self.mag_canvas = FigureCanvas(self.mag_figure)
        mag_layout.addWidget(self.mag_canvas)
        self.tabs.addTab(mag_widget, "Magnetization")
        
        # Tab for Phase Plot
        phase_widget = QWidget()
        phase_layout = QVBoxLayout(phase_widget)
        self.phase_figure = Figure(figsize=(8, 6))
        self.phase_canvas = FigureCanvas(self.phase_figure)
        phase_layout.addWidget(self.phase_canvas)
        self.tabs.addTab(phase_widget, "Phase")
        
        # Tab for Z-Magnetization
        mz_widget = QWidget()
        mz_layout = QVBoxLayout(mz_widget)
        self.mz_figure = Figure(figsize=(8, 6))
        self.mz_canvas = FigureCanvas(self.mz_figure)
        mz_layout.addWidget(self.mz_canvas)
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