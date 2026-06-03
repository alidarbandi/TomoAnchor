from __future__ import annotations

from pathlib import Path


APP_BACKGROUND = "#273A59"
PANEL_BACKGROUND = "#1F2F49"
GROUP_BACKGROUND = "#2E4568"
INPUT_BACKGROUND = "#1F2F49"
TEXT_COLOR = "#F4F8FF"
MUTED_TEXT_COLOR = "#DDE8F7"
ACCENT_COLOR = "#78A6E6"
ACCENT_HOVER_COLOR = "#A8C7F7"
GRID_COLOR = "#8EAEE0"

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
APP_ICON_PATH = ASSETS_DIR / "app_icon.ico"
COMBO_DOWN_ARROW_PATH = (ASSETS_DIR / "combo_down_arrow.xpm").as_posix()
SPIN_UP_ARROW_PATH = (ASSETS_DIR / "spin_up_arrow.xpm").as_posix()
SPIN_DOWN_ARROW_PATH = (ASSETS_DIR / "spin_down_arrow.xpm").as_posix()
SPLITTER_VERTICAL_GRIP_PATH = (ASSETS_DIR / "splitter_vertical_grip.xpm").as_posix()
SCROLLBAR_VERTICAL_GRIP_PATH = (ASSETS_DIR / "scrollbar_vertical_grip.xpm").as_posix()
SCROLLBAR_HORIZONTAL_GRIP_PATH = (ASSETS_DIR / "scrollbar_horizontal_grip.xpm").as_posix()


APP_STYLE_SHEET = """
QWidget {
    background-color: #273A59;
    color: #F4F8FF;
    font-size: 10pt;
    selection-background-color: #78A6E6;
    selection-color: #102033;
}

QMainWindow,
QScrollArea,
QAbstractScrollArea,
QSplitter,
QTabWidget::pane {
    background-color: #273A59;
}

QGroupBox {
    background-color: #2E4568;
    border: 1px solid #5F78A0;
    border-radius: 6px;
    margin-top: 18px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: #FFFFFF;
    background-color: #273A59;
}

QLabel,
QCheckBox,
QRadioButton {
    background: transparent;
    color: #F4F8FF;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #A8C7F7;
    border-radius: 3px;
    background-color: #1F2F49;
}

QCheckBox::indicator:checked {
    background-color: #78A6E6;
    border-color: #D8E8FF;
}

QCheckBox::indicator:disabled {
    background-color: #33445F;
    border-color: #52657F;
}

QSplitter::handle {
    background-color: #78A6E6;
    border: 1px solid #D8E8FF;
}

QSplitter::handle:horizontal {
    image: url("__SPLITTER_VERTICAL_GRIP_PATH__");
    width: 12px;
    margin: 0 2px;
}

QSplitter::handle:hover {
    background-color: #A8C7F7;
    border-color: #FFFFFF;
}

QSplitter::handle:pressed {
    background-color: #3F5F8B;
}

QLineEdit,
QTextEdit,
QPlainTextEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QTableWidget {
    background-color: #1F2F49;
    color: #F8FBFF;
    border: 1px solid #6C86AE;
    border-radius: 4px;
    padding: 4px 6px;
}

QComboBox {
    padding-right: 34px;
}

QSpinBox,
QDoubleSpinBox {
    padding-right: 30px;
}

QLineEdit:focus,
QTextEdit:focus,
QPlainTextEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus {
    border: 1px solid #A8C7F7;
}

QLineEdit:read-only,
QSpinBox:read-only,
QDoubleSpinBox:read-only {
    background-color: #263954;
    color: #DDE8F7;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    background-color: #3F5F8B;
    border-left: 1px solid #8EAEE0;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
    width: 30px;
}

QComboBox::drop-down:hover {
    background-color: #4C70A2;
    border-left-color: #B8D1FA;
}

QComboBox::drop-down:pressed {
    background-color: #213451;
}

QComboBox::down-arrow {
    image: url("__COMBO_DOWN_ARROW_PATH__");
    width: 16px;
    height: 16px;
}

QComboBox::down-arrow:on {
    top: 1px;
}

QSpinBox::up-button,
QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    background-color: #3F5F8B;
    border-left: 1px solid #8EAEE0;
    border-bottom: 1px solid #8EAEE0;
    border-top-right-radius: 4px;
    width: 26px;
    height: 15px;
}

QSpinBox::down-button,
QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    background-color: #3F5F8B;
    border-left: 1px solid #8EAEE0;
    border-top: 1px solid #8EAEE0;
    border-bottom-right-radius: 4px;
    width: 26px;
    height: 15px;
}

QSpinBox::up-button:hover,
QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover {
    background-color: #4C70A2;
    border-color: #B8D1FA;
}

QSpinBox::up-button:pressed,
QSpinBox::down-button:pressed,
QDoubleSpinBox::up-button:pressed,
QDoubleSpinBox::down-button:pressed {
    background-color: #213451;
}

QSpinBox::up-button:disabled,
QSpinBox::down-button:disabled,
QDoubleSpinBox::up-button:disabled,
QDoubleSpinBox::down-button:disabled {
    background-color: #33445F;
    border-color: #52657F;
}

QSpinBox::up-arrow,
QDoubleSpinBox::up-arrow {
    image: url("__SPIN_UP_ARROW_PATH__");
    width: 16px;
    height: 10px;
}

QSpinBox::down-arrow,
QDoubleSpinBox::down-arrow {
    image: url("__SPIN_DOWN_ARROW_PATH__");
    width: 16px;
    height: 10px;
}

QComboBox QAbstractItemView {
    background-color: #1F2F49;
    color: #F8FBFF;
    selection-background-color: #78A6E6;
    selection-color: #102033;
    border: 1px solid #6C86AE;
    outline: 0;
}

QPushButton {
    background-color: #3F5F8B;
    color: #FFFFFF;
    border: 1px solid #8EAEE0;
    border-radius: 5px;
    padding: 7px 10px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #4C70A2;
    border-color: #B8D1FA;
}

QPushButton:pressed {
    background-color: #213451;
}

QPushButton:disabled,
QLineEdit:disabled,
QTextEdit:disabled,
QPlainTextEdit:disabled,
QComboBox:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled,
QCheckBox:disabled,
QLabel:disabled {
    background-color: #33445F;
    color: #B8C5D8;
    border-color: #52657F;
}

QTabBar::tab {
    background-color: #213451;
    color: #DDE8F7;
    border: 1px solid #5F78A0;
    border-bottom: none;
    padding: 8px 12px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #2E4568;
    color: #FFFFFF;
}

QTabBar::tab:hover {
    background-color: #354F75;
    color: #FFFFFF;
}

QHeaderView::section {
    background-color: #354F75;
    color: #FFFFFF;
    border: 1px solid #6C86AE;
    padding: 5px;
}

QTableWidget {
    gridline-color: #5F78A0;
    alternate-background-color: #263954;
}

QTableWidget::item {
    padding: 4px;
}

QTableWidget::item:selected {
    background-color: #78A6E6;
    color: #102033;
}

QScrollBar:vertical {
    background-color: #18263E;
    border-left: 1px solid #D8E8FF;
    width: 18px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #78A6E6;
    image: url("__SCROLLBAR_VERTICAL_GRIP_PATH__");
    border: 1px solid #FFFFFF;
    border-radius: 7px;
    min-height: 72px;
    margin: 3px 2px;
}

QScrollBar::handle:vertical:hover {
    background-color: #A8C7F7;
    border-color: #FFFFFF;
}

QScrollBar::handle:vertical:pressed {
    background-color: #3F5F8B;
}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background-color: #18263E;
}

QScrollBar:horizontal {
    background-color: #18263E;
    border-top: 1px solid #D8E8FF;
    height: 18px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background-color: #78A6E6;
    image: url("__SCROLLBAR_HORIZONTAL_GRIP_PATH__");
    border: 1px solid #FFFFFF;
    border-radius: 7px;
    min-width: 72px;
    margin: 2px 3px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #A8C7F7;
    border-color: #FFFFFF;
}

QScrollBar::handle:horizontal:pressed {
    background-color: #3F5F8B;
}

QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background-color: #18263E;
}

QScrollBar::add-line,
QScrollBar::sub-line {
    width: 0;
    height: 0;
}

QSlider::groove:horizontal {
    background-color: #1F2F49;
    border: 1px solid #6C86AE;
    border-radius: 4px;
    height: 8px;
}

QSlider::handle:horizontal {
    background-color: #78A6E6;
    border: 1px solid #D8E8FF;
    border-radius: 6px;
    width: 16px;
    margin: -5px 0;
}

QSlider::handle:horizontal:hover {
    background-color: #A8C7F7;
}

QProgressBar {
    background-color: #1F2F49;
    color: #F4F8FF;
    border: 1px solid #6C86AE;
    border-radius: 4px;
    padding: 2px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #78A6E6;
    border-radius: 3px;
}

QStatusBar {
    background-color: #1F2F49;
    color: #F4F8FF;
    border-top: 1px solid #5F78A0;
}

QMenu {
    background-color: #1F2F49;
    color: #F8FBFF;
    border: 1px solid #6C86AE;
}

QMenu::item {
    padding: 6px 22px;
}

QMenu::item:selected {
    background-color: #78A6E6;
    color: #102033;
}

QToolTip {
    background-color: #1F2F49;
    color: #F8FBFF;
    border: 1px solid #A8C7F7;
    padding: 4px;
}
""".replace("__COMBO_DOWN_ARROW_PATH__", COMBO_DOWN_ARROW_PATH).replace(
    "__SPIN_UP_ARROW_PATH__", SPIN_UP_ARROW_PATH
).replace("__SPIN_DOWN_ARROW_PATH__", SPIN_DOWN_ARROW_PATH).replace(
    "__SPLITTER_VERTICAL_GRIP_PATH__", SPLITTER_VERTICAL_GRIP_PATH
).replace("__SCROLLBAR_VERTICAL_GRIP_PATH__", SCROLLBAR_VERTICAL_GRIP_PATH).replace(
    "__SCROLLBAR_HORIZONTAL_GRIP_PATH__", SCROLLBAR_HORIZONTAL_GRIP_PATH
)


def style_canvas(canvas) -> None:
    canvas.setStyleSheet(f"background-color: {APP_BACKGROUND};")


def style_figure(figure) -> None:
    figure.patch.set_facecolor(APP_BACKGROUND)


def style_axis(axis, *, grid: bool = False) -> None:
    axis.set_facecolor(PANEL_BACKGROUND)
    axis.tick_params(colors=MUTED_TEXT_COLOR)
    axis.xaxis.label.set_color(TEXT_COLOR)
    axis.yaxis.label.set_color(TEXT_COLOR)
    axis.title.set_color(TEXT_COLOR)
    for spine in axis.spines.values():
        spine.set_color(GRID_COLOR)
    if grid:
        axis.grid(True, color=GRID_COLOR, alpha=0.24)
