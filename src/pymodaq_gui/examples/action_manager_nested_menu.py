"""
Action Manager Nested Menu Example

Demonstrates the nested menu capabilities of ActionManager including:
- Creating submenus within submenus (multiple levels)
- Adding actions to nested submenus
- Getting action paths
- Using icons for both actions and submenus
"""

import sys
from qtpy import QtWidgets

from pymodaq_gui.managers.action_manager import ActionManager


class NestedMenuExample(QtWidgets.QMainWindow, ActionManager):
    """Example application demonstrating nested menus with ActionManager"""

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        ActionManager.__init__(self, menu=self.menuBar())

        self.setWindowTitle("ActionManager Nested Menu Example")
        self.resize(800, 600)

        # Create central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Add info label
        info_label = QtWidgets.QLabel(
            "<h2>ActionManager Nested Menu Example</h2>"
            "<p>This example demonstrates nested menu capabilities:</p>"
            "<ul>"
            "<li><b>Multiple menu levels</b> - Submenus within submenus</li>"
            "<li><b>Action paths</b> - See full menu path when clicking actions</li>"
            "<li><b>Icons</b> - Both menus and actions can have icons</li>"
            "<li><b>Flexible organization</b> - Group related actions logically</li>"
            "<li><b>Shared submenus</b> - 'Recent Files' appears in both File and Edit menus</li>"
            "<li><b>Shared actions</b> - 'Save' appears in File and Tools menus</li>"
            "</ul>"
            "<p>Try clicking any menu item to see its full path!</p>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Status display
        self.status_display = QtWidgets.QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setMaximumHeight(300)
        layout.addWidget(QtWidgets.QLabel("<b>Action Log:</b>"))
        layout.addWidget(self.status_display)

        # Setup all menus and actions
        self.setup_actions()

        self.log_message("Application started. Click any menu item to see its path.")

    def setup_actions(self):
        """Create nested menu structure with actions"""

        # ========== File Menu ==========
        file_menu = self.add_submenu('file', 'File', icon_name='Folder')

        # Add simple actions to File menu
        self.add_action('new', 'New', icon_name='NewFile',
                       submenu='file', shortcut='Ctrl+N')
        self.add_action('open', 'Open...', icon_name='Open',
                       submenu='file', shortcut='Ctrl+O')
        self.add_action('save', 'Save', icon_name='SaveAs',
                       submenu='file', shortcut='Ctrl+S')

        # Create "Recent Files" submenu within File menu
        recent_menu = self.add_submenu('recent', 'Recent Files',
                                      menu=file_menu, icon_name='Folder')

        # Add recent file actions
        for i in range(1, 4):
            self.add_action(f'recent_{i}', f'Project_{i}.py',
                          submenu=recent_menu)

        # Create "Export" submenu within File menu
        export_menu = self.add_submenu('export', 'Export',
                                      menu=file_menu, icon_name='SaveAs')

        # Add export format actions
        self.add_action('export_pdf', 'Export as PDF', submenu=export_menu)
        self.add_action('export_png', 'Export as PNG', submenu=export_menu)
        self.add_action('export_svg', 'Export as SVG', submenu=export_menu)

        # ========== Edit Menu ==========
        edit_menu = self.add_submenu('edit', 'Edit')

        self.add_action('undo', 'Undo', submenu='edit', shortcut='Ctrl+Z')
        self.add_action('redo', 'Redo', submenu='edit', shortcut='Ctrl+Y')
        self.add_action('cut', 'Cut', submenu='edit', shortcut='Ctrl+X')
        self.add_action('copy', 'Copy', submenu='edit', shortcut='Ctrl+C')
        self.add_action('paste', 'Paste', submenu='edit', shortcut='Ctrl+V')

        # ========== View Menu with Deep Nesting ==========
        view_menu = self.add_submenu('view', 'View')

        # Level 1: Panels submenu
        panels_menu = self.add_submenu('panels', 'Panels', menu=view_menu)

        # Level 2: Left Panel submenu
        left_panel_menu = self.add_submenu('left_panel', 'Left Panel',
                                          menu=panels_menu)
        self.add_action('left_files', 'File Explorer',
                       submenu=left_panel_menu, checkable=True, checked=True)
        self.add_action('left_search', 'Search',
                       submenu=left_panel_menu, checkable=True)
        self.add_action('left_git', 'Git',
                       submenu=left_panel_menu, checkable=True)

        # Level 2: Right Panel submenu
        right_panel_menu = self.add_submenu('right_panel', 'Right Panel',
                                           menu=panels_menu)
        self.add_action('right_outline', 'Outline',
                       submenu=right_panel_menu, checkable=True)
        self.add_action('right_terminal', 'Terminal',
                       submenu=right_panel_menu, checkable=True, checked=True)

        # Level 2: Bottom Panel submenu
        bottom_panel_menu = self.add_submenu('bottom_panel', 'Bottom Panel',
                                            menu=panels_menu)
        self.add_action('bottom_console', 'Console',
                       submenu=bottom_panel_menu, checkable=True, checked=True)
        self.add_action('bottom_problems', 'Problems',
                       submenu=bottom_panel_menu, checkable=True)
        self.add_action('bottom_output', 'Output',
                       submenu=bottom_panel_menu, checkable=True)

        # Add appearance submenu to View
        appearance_menu = self.add_submenu('appearance', 'Appearance',
                                          menu=view_menu)

        # Theme submenu (3 levels deep!)
        theme_menu = self.add_submenu('theme', 'Theme', menu=appearance_menu)
        self.add_action('theme_light', 'Light', submenu=theme_menu)
        self.add_action('theme_dark', 'Dark', submenu=theme_menu)
        self.add_action('theme_auto', 'Auto', submenu=theme_menu)

        # Font size submenu (3 levels deep!)
        font_menu = self.add_submenu('font', 'Font Size', menu=appearance_menu)
        self.add_action('font_small', 'Small', submenu=font_menu)
        self.add_action('font_medium', 'Medium', submenu=font_menu)
        self.add_action('font_large', 'Large', submenu=font_menu)

        # ========== Tools Menu ==========
        tools_menu = self.add_submenu('tools', 'Tools')

        # Add tools with nested settings
        self.add_action('tool_format', 'Format Document', submenu='tools')

        settings_menu = self.add_submenu('settings', 'Settings',
                                        menu=tools_menu, icon_name='Params')

        # General settings
        general_settings = self.add_submenu('general_settings',
                                           'General', menu=settings_menu)
        self.add_action('auto_save', 'Auto Save',
                       submenu=general_settings, checkable=True, checked=True)
        self.add_action('show_tooltips', 'Show Tooltips',
                       submenu=general_settings, checkable=True, checked=True)

        # Advanced settings
        advanced_settings = self.add_submenu('advanced_settings',
                                            'Advanced', menu=settings_menu)
        self.add_action('debug_mode', 'Debug Mode',
                       submenu=advanced_settings, checkable=True)
        self.add_action('verbose_logging', 'Verbose Logging',
                       submenu=advanced_settings, checkable=True)

        # ========== Help Menu ==========
        help_menu = self.add_submenu('help', 'Help')
        self.add_action('docs', 'Documentation', submenu='help')
        self.add_action('about', 'About', submenu='help')

        # ========== Demonstration of Shared Submenus ==========
        # You can add the same submenu to multiple parent menus!
        # Let's add the "Recent Files" submenu to the Edit menu as well
        edit_menu_obj = self.get_submenu('edit')
        recent_menu_obj = self.get_submenu('recent')
        edit_menu_obj.addMenu(recent_menu_obj)
        # Now "Recent Files" appears in both File and Edit menus,
        # but it's the same submenu instance - changes appear everywhere!

        # ========== Demonstration of Shared Actions ==========
        # The 'save' action can be added to multiple menus/toolbars
        tools_menu_obj = self.get_submenu('tools')
        self.affect_to('save', tools_menu_obj)
        # Now "Save" appears in File and Tools menus

        # Connect all actions to the same handler that shows the path
        for action_name in self.actions_names:
            self.connect_action(action_name, lambda checked=False, name=action_name:
                              self.on_action_triggered(name))

    def on_action_triggered(self, action_name):
        """Handle action triggered - display action info"""
        action = self.get_action(action_name)
        path = self.get_action_path(action_name)

        # Build status message
        message = f"<b>Action:</b> {action.text()}"
        if path:
            message += f"<br><b>Path:</b> {path}"
        else:
            message += "<br><i>(Not in any menu)</i>"

        message += f"<br><b>Internal Name:</b> {action_name}"

        if action.isCheckable():
            state = "Checked" if action.isChecked() else "Unchecked"
            message += f"<br><b>State:</b> {state}"

        if action.shortcut().toString():
            message += f"<br><b>Shortcut:</b> {action.shortcut().toString()}"

        self.log_message(message)

    def log_message(self, message):
        """Add message to status display"""
        self.status_display.append(message)
        self.status_display.append("<hr>")
        # Scroll to bottom
        scrollbar = self.status_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


def main():
    """Run the example application"""
    app = QtWidgets.QApplication(sys.argv)

    window = NestedMenuExample()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
