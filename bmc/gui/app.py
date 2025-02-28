#!/usr/bin/env python3
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QLocale
import sys
import locale
from bmc.gui.main_window import BMCSimulatorGUI

if __name__ == "__main__":
    # Setze das Locale auf Englisch, um sicherzustellen, dass Punkt als Dezimaltrennzeichen verwendet wird
    locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')
    
    app = QApplication(sys.argv)
    
    # Setze auch das Qt-Locale auf Englisch mit Punkt als Dezimaltrennzeichen
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
    
    window = BMCSimulatorGUI()
    window.show()
    sys.exit(app.exec())