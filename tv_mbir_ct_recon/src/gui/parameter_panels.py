from __future__ import annotations

from .qt_compat import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget


class PathPicker(QWidget):
    def __init__(self, mode: str = "folder") -> None:
        super().__init__()
        self.mode = mode
        self.edit = QLineEdit()
        self.button = QPushButton("Browse")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)
        self.button.clicked.connect(self._browse)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, text: str) -> None:
        self.edit.setText(text)

    def _browse(self) -> None:
        if self.mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, "Choose file", "", "All files (*)")
        elif self.mode == "save":
            path, _ = QFileDialog.getSaveFileName(self, "Choose file", "", "All files (*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Choose folder")
        if path:
            self.edit.setText(path)
