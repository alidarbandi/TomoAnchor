from __future__ import annotations

import os
import sys


QT_BACKEND = ""

try:
    os.environ.setdefault("QT_API", "pyside6")
    from PySide6.QtCore import QEvent, QObject, QThread, QTimer, Qt, Signal, Slot
    from PySide6.QtGui import QColor, QIcon, QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSlider,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextBrowser,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    QT_BACKEND = "PySide6"
except Exception:
    for module_name in list(sys.modules):
        if module_name == "PySide6" or module_name.startswith("PySide6."):
            sys.modules.pop(module_name, None)
    os.environ["QT_API"] = "pyqt5"
    from PyQt5.QtCore import QEvent, QObject, QThread, QTimer, Qt, pyqtSignal as Signal, pyqtSlot as Slot
    from PyQt5.QtGui import QColor, QIcon, QTextCursor
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSlider,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextBrowser,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    QT_BACKEND = "PyQt5"


def exec_app(app: QApplication) -> int:
    method = getattr(app, "exec", None) or getattr(app, "exec_")
    return int(method())
