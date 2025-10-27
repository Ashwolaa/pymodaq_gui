"""
Simple ConfigManager Menu Example

A minimal example showing ConfigManager menu usage without displaying
the settings tree in the main window (which can cause ownership issues).

This is the recommended pattern for using ConfigManager with menus.
"""

import sys
from pathlib import Path
from qtpy import QtWidgets, QtCore

from pymodaq_gui.managers.config_manager import ConfigManager


class SimpleConfigManager(ConfigManager):
    """Simple ConfigManager subclass"""

    title = "Demo"
    name = "demo"

    def make_config(self):
        """Define simple parameters"""
        return [
            {'title': 'Name:', 'name': 'name', 'type': 'str', 'value': 'Demo Config'},
            {'title': 'Value:', 'name': 'value', 'type': 'int', 'value': 42},
            {'title': 'Enabled:', 'name': 'enabled', 'type': 'bool', 'value': True},
        ]


def main():
    """Simple example application"""
    app = QtWidgets.QApplication(sys.argv)

    # Create main window
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("ConfigManager Simple Menu Example")
    window.resize(400, 200)

    # Create central widget
    central_widget = QtWidgets.QWidget()
    window.setCentralWidget(central_widget)
    layout = QtWidgets.QVBoxLayout(central_widget)

    # Add some info text
    info_label = QtWidgets.QLabel(
        "<h2>ConfigManager Menu Example</h2>"
        "<p>Use the <b>Demo Configs</b> menu to:</p>"
        "<ul>"
        "<li><b>New:</b> Create new configurations</li>"
        "<li><b>Edit Current:</b> Modify the loaded config</li>"
        "<li><b>Duplicate:</b> Copy and edit a config</li>"
        "<li><b>Load:</b> Quick load from submenu</li>"
        "<li><b>Delete:</b> Remove unwanted configs</li>"
        "<li><b>Open Config Directory:</b> Browse files</li>"
        "</ul>"
        "<p><i>All config dialogs are modal and stable!</i></p>"
    )
    info_label.setWordWrap(True)
    layout.addWidget(info_label)

    # Status display
    status_display = QtWidgets.QTextEdit()
    status_display.setReadOnly(True)
    status_display.setMaximumHeight(150)
    layout.addWidget(QtWidgets.QLabel("<b>Activity Log:</b>"))
    layout.addWidget(status_display)

    # Create temp directory for configs
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix="demo_configs_"))

    # Create ConfigManager WITHOUT showing settings tree in main window
    config_manager = SimpleConfigManager(config_path=temp_dir, msgbox=False)

    # Create menu bar and add config menu
    menubar = window.menuBar()

    # Add File menu
    file_menu = menubar.addMenu("File")
    file_menu.addAction("Exit", app.quit)

    # Example 1: Full menu with all actions (default)
    config_menu = config_manager.create_menu(menubar, "Configs full actions")

    # Example 2: Custom menu with only specific actions (commented out)
    config_menu = config_manager.create_menu(
        menubar,
        "Configs selected actions",
        actions=['new', 'edit', 'load', 'open_dir']  # No delete for safety
    )

    # Connect signals to update display
    def log_message(msg):
        status_display.append(msg)
        status_display.verticalScrollBar().setValue(
            status_display.verticalScrollBar().maximum()
        )

    def on_config_loaded(path):
        log_message(f"✓ Loaded: {path.stem}")

    def on_config_saved(path):
        log_message(f"✓ Saved: {path.stem}")

    def on_config_deleted(path):
        log_message(f"✗ Deleted: {path.stem}")

    config_manager.config_loaded.connect(on_config_loaded)
    config_manager.config_saved.connect(on_config_saved)
    config_manager.config_deleted.connect(on_config_deleted)

    # Show initial message
    log_message("Welcome! Use 'Demo Configs → New Demo...' to get started.")
    log_message(f"Config directory: {temp_dir}")

    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
