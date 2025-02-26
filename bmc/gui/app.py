#!/usr/bin/env python3
from PyQt6.QtWidgets import QApplication
import sys
from bmc.gui.main_window import BMCSimulatorGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BMCSimulatorGUI()
    window.show()
    sys.exit(app.exec())