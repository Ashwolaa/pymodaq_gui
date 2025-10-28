from pathlib import Path
from typing import Optional, List

from qtpy import QtWidgets, QtCore
from qtpy.QtWidgets import QMessageBox, QDialogButtonBox, QDialog, QMenu, QAction, QMenuBar
from qtpy.QtCore import QObject, Signal

from pymodaq_utils.logger import set_logger, get_module_name

from pymodaq_gui.managers.parameter_manager import ParameterManager
from pymodaq_gui.managers.action_manager import ActionManager
from pymodaq_gui.parameter import ioxml, Parameter
from pymodaq_gui.utils import select_file
from pymodaq_gui.messenger import dialog as dialogbox
import qtawesome as qta

logger = set_logger(get_module_name(__file__))


class ConfigManager(ParameterManager, ActionManager, QObject):
    """
    Manager class for handling configuration files and parameters.

    Provides functionality to create, load, modify, and save configuration
    files in XML format with a graphical user interface.

    Attributes:
        title (str): Display title for the configuration manager
        name (str): Internal name for the configuration manager
        config_path (Path): Path to the directory containing config files

    Signals:
        config_loaded (Path): Emitted when a configuration file is loaded
        config_saved (Path): Emitted when a configuration file is saved
    """

    # Signals
    config_loaded = Signal(Path)
    config_saved = Signal(Path)
    config_deleted = Signal(Path)

    title = "Config"
    name = "config"

    def __init__(self, config_path: Path = "", msgbox=False):
        """
        Initialize the ConfigManager.

        Args:
            config_path (Path, optional): Path to the configuration directory. Defaults to ''.
            msgbox (bool, optional): If True, shows a dialog box on initialization asking
                whether to create a new config or modify an existing one. Defaults to False.
        """
        ParameterManager.__init__(self, settings_name=self.name)
        ActionManager.__init__(self)
        QtCore.QObject.__init__(self)
        self.config_path = config_path
        self.setup_actions()
        if msgbox:
            msgBox = QMessageBox()
            msgBox.setText(f"{self.title} Manager")
            msgBox.setInformativeText("What do you want to do?")
            cancel_button = msgBox.addButton(QMessageBox.StandardButton.Cancel)
            new_button = msgBox.addButton("New", QMessageBox.ButtonRole.ActionRole)
            modify_button = msgBox.addButton("Modify", QMessageBox.ButtonRole.AcceptRole)
            msgBox.setDefaultButton(QMessageBox.StandardButton.Cancel)
            ret = msgBox.exec()

            if msgBox.clickedButton() == new_button:
                self.set_new_config()

            elif msgBox.clickedButton() == modify_button:
                path = select_file(start_path=config_path, save=False, ext="xml")
                if path != "":
                    self.set_config_from_file(str(path))
            else:  # cancel
                pass

    def make_config(self):
        """
        Create additional configuration parameters.

        Method to be subclassed to add custom parameters specific to
        the configuration needs of derived classes.

        Returns:
            list: List of parameter dictionaries to be added to the configuration.
                Empty list in base implementation.
        """
        return []

    def validate_config(self) -> bool:
        """
        Validate the loaded configuration.

        Method to be subclassed to add custom validation logic specific to
        the configuration needs of derived classes. Called after loading
        a configuration from file.

        The validation should use the logger to report issues at appropriate levels:
        - logger.error(): Critical issues that prevent config from working (return False)
        - logger.warning(): Non-critical issues that might cause problems
        - logger.info(): Informational messages about config contents

        Returns:
            bool: True if configuration is valid (no errors), False otherwise.
                Always returns True in base implementation.

        Example:
            >>> def validate_config(self):
            >>>     # Check required parameters exist
            >>>     if not self.settings.hasChild('required_param'):
            >>>         logger.error("Missing required parameter 'required_param'")
            >>>         return False
            >>>
            >>>     # Check value ranges
            >>>     value = self.settings.child('some_param').value()
            >>>     if value < 0:
            >>>         logger.error(f"Parameter 'some_param' must be >= 0, got {value}")
            >>>         return False
            >>>
            >>>     # Non-critical warnings
            >>>     if value > 100:
            >>>         logger.warning(f"Parameter 'some_param' is unusually high: {value}")
            >>>
            >>>     logger.info("Configuration validation passed")
            >>>     return True
        """
        return True

    # ============ Action Management ============

    def setup_actions(self):
        """
        Setup standard configuration actions using ActionManager.

        Creates the following actions:
        - 'new': Create a new configuration
        - 'edit': Edit the currently loaded configuration
        - 'duplicate': Duplicate current config with a new name
        - 'delete': Delete a configuration file
        - 'refresh': Refresh the load menu
        - 'open_dir': Open the configuration directory

        Subclasses can override to add custom actions.
        """

        self.add_action(
            "new",
            f"New {self.title}...",
            icon_name=qta.icon("ei.file-new"),
            tip=f"Create a new {self.title} configuration",
        )
        self.add_action(
            "edit",
            f"Edit Current {self.title}...",
            icon_name=qta.icon("ei.file-edit"),
            tip="Edit the currently loaded configuration",
        )
        self.add_action(
            "duplicate",
            f"Duplicate {self.title}...",
            icon_name=qta.icon("fa5.copy"),
            tip="Duplicate configuration with a new name",
        )
        self.add_action(
            "delete",
            f"Delete {self.title}...",
            icon_name=qta.icon("mdi.delete"),
            tip="Delete a configuration file",
        )
        self.add_action(
            "refresh",
            "Refresh List",
            icon_name=qta.icon("ei.refresh"),
            tip="Refresh the configuration list",
        )
        self.add_action(
            "open_dir",
            "Open Config Directory",
            icon_name=qta.icon("mdi.folder-open"),
            tip="Open configuration directory in file explorer",
        )

        # Connect actions to methods
        self.connect_action('new', self._menu_new_config)
        self.connect_action('edit', self._menu_edit_current_config)
        self.connect_action('duplicate', self._menu_duplicate_config)
        self.connect_action('delete', self._menu_delete_config)
        self.connect_action('refresh', self._menu_refresh_list)
        self.connect_action('open_dir', self._menu_open_config_dir)

        self.config_saved.connect(self._populate_load_menu)
        self.config_deleted.connect(self._populate_load_menu)
    # ============ End Action Management ============

    def set_new_config(self, file: str = None, show=True):
        """
        Create a new configuration with default parameters.

        Opens a dialog allowing the user to set up a new configuration file
        with a filename and any additional parameters defined in make_config().

        Args:
            file (str, optional): Default filename for the new config.
                If None, uses "{title}_default". Defaults to None.
            show (bool, optional): If True, displays the configuration dialog
                after creating. Defaults to True.
        """
        if file is None:
            file = f"{self.title}_default"
        param = [
            {"title": "Filename:", "name": "filename", "type": "str", "value": file},
        ]
        additional_params = self.make_config()
        self.settings = Parameter.create(
            title=f"{self.title}",
            name=f"{self.name}",
            type="group",
            children=param + additional_params,
        )
        logger.info(f"Creating a new {self.name} file")
        if show:
            self.show_config()

    def set_config_from_file(self, file_path: Path, show=True):
        """
        Load an existing configuration from an XML file.

        Reads an XML configuration file and populates the settings tree
        with the parameters from the file.

        Args:
            file_path (Path): Path to the XML configuration file to load.
            show (bool, optional): If True, displays the configuration dialog
                after loading. Defaults to True.

        Returns:
            bool: True if configuration was loaded and validated successfully,
                False otherwise.

        Note:
            If file_path is not a Path object, it will be converted to one.
            Only XML files are supported.
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        if file_path.suffix == ".xml":
            children = ioxml.XML_file_to_parameter(file_path)
        else:
            logger.exception("file_path must be of xml type")
            return False

        self.settings = Parameter.create(
            title=f"{self.title}",
            name=f"{self.name}",
            type="group",
            children=children,
        )

        # Validate the loaded configuration
        is_valid = self.validate_config()
        if not is_valid:
            logger.error(f"Configuration validation failed for {file_path.name}")
            # Show validation error dialog
            QMessageBox.critical(
                None,
                "Invalid Configuration",
                f"The configuration file '{file_path.name}' failed validation.\n"
            )
            return False

        self.config_loaded.emit(file_path)
        if show:
            self.show_config()

        return True

    def show_config(self, widget=None, overwrite=False):
        """
        Display the configuration dialog for viewing and editing settings.

        Creates and shows a modal dialog containing the settings tree with
        Save and Cancel buttons. If the user clicks Save, the configuration
        is saved to file.

        Args:
            widget (QtWidgets.QWidget, optional): Additional widget to include
                in the dialog layout. Defaults to None.
            overwrite (bool, optional): If True, overwrites existing files without
                prompting. Defaults to False.
        Returns:
            bool: True if the file was successfully saved, False otherwise.
        """
        dialog = QDialog()
        vlayout = QtWidgets.QVBoxLayout()

        # Store original parent to restore later
        original_parent = self.settings_tree.parent()

        vlayout.addWidget(self.settings_tree)
        dialog.setLayout(vlayout)
        buttonBox = QDialogButtonBox(parent=dialog)

        buttonBox.addButton("Save", QDialogButtonBox.ButtonRole.AcceptRole)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        buttonBox.rejected.connect(dialog.reject)

        vlayout.addWidget(buttonBox)
        dialog.setWindowTitle("Fill in information about this managers")

        if widget is not None and isinstance(widget, QtWidgets.QWidget):
            vlayout.addWidget(widget)

        res = dialog.exec()

        # Restore settings_tree to original parent before dialog is destroyed
        if original_parent is not None:
            original_parent.layout().addWidget(self.settings_tree)
        else:
            # Remove from dialog layout to prevent deletion
            vlayout.removeWidget(self.settings_tree)
            self.settings_tree.setParent(None)

        if res == QDialog.DialogCode.Accepted:
            return self.save_config(overwrite)
        else:
            return False
    
    def save_config(self, overwrite=False):
        """
        Save the current configuration to an XML file.

        Saves the settings to an XML file in the config_path directory using
        the filename specified in the settings. If the file already exists and
        overwrite is False, prompts the user for confirmation before overwriting.

        Args:
            overwrite (bool, optional): If True, overwrites existing files without
                prompting. If False, asks for user confirmation before overwriting.
                Defaults to False.

        Returns:
            bool: True if the file was successfully saved, False otherwise.

        Note:
            The filename is retrieved from the 'filename' parameter in settings.
            The file is saved with a .xml extension in the config_path directory.
        """
        filename = self.settings.child("filename").value()       
        saved = False
        try:
            ioxml.parameter_to_xml_file(self.settings, self.config_path.joinpath(filename), overwrite=overwrite)
            saved = True
        except FileExistsError as currenterror:
            logger.warning(f"{currenterror} File {filename}.xml exists")
            user_agreed = dialogbox(
                title="Overwrite confirmation",
                message="File exist do you want to overwrite it ?",
            )
            if user_agreed:
                ioxml.parameter_to_xml_file(self.settings, self.config_path.joinpath(filename))
                logger.warning(f"File {filename}.xml overwriten at user request")
                saved = True
            else:
                logger.warning(f"File {filename}.xml wasn't saved at user request")

        if saved:
            self.config_saved.emit(self.config_path.joinpath(filename))
        return saved

    # ============ Menu Management Methods ============

    def create_menu(self, menubar: Optional[QMenuBar] = None, menu_title: Optional[str] = None,
                    actions: Optional[List[str]] = None) -> QMenu:
        """
        Create a menu with standard configuration actions using ActionManager's built-in menu.

        Available actions (default: all):
        - 'new': Create a new configuration from scratch
        - 'edit': Open dialog to modify the currently loaded configuration
        - 'duplicate': Create a copy of current config with a new name for editing
        - 'load': Submenu with quick load from available configurations
        - 'delete': Delete a configuration file
        - 'refresh': Refresh the Load submenu
        - 'open_dir': Open the configuration directory in file explorer

        Args:
            menubar (QMenuBar, optional): Menu bar to add the menu to. If None,
                uses ActionManager's internal menu. Defaults to None.
            menu_title (str, optional): Title for the menu. If None, uses
                "{self.title} Configs". Defaults to None.
            actions (List[str], optional): List of action names to include in the menu.
                If None, includes all actions: ['new', 'edit', 'duplicate', 'load',
                'delete', 'refresh', 'open_dir']. Defaults to None.

        Returns:
            QMenu: The menu object (ActionManager's self._menu or newly created)

        Examples:
            >>> # Full menu with all actions, added to menubar
            >>> menu = manager.create_menu(menubar)

            >>> # Minimal menu with only new and load
            >>> menu = manager.create_menu(menubar, actions=['new', 'load'])

            >>> # Standalone menu (uses ActionManager's internal menu)
            >>> menu = manager.create_menu()  # Returns self._menu
        """
        if menu_title is None:
            menu_title = f"{self.title} Configs"

        # Default to all actions if not specified
        if actions is None:
            actions = ['new', 'edit', 'duplicate', 'load', 'delete', 'refresh', 'open_dir']

        # Use ActionManager's menu or create new one for menubar
        if menubar is not None:
            # Create menu in the menubar
            self._menu = menubar.addMenu(menu_title)
        elif self._menu is None:
            # Create standalone menu
            self._menu = QMenu(menu_title)

        # Clear menu if it already has items
        self._menu.clear()
        self._menu.setTitle(menu_title)

        # Track if we need separators
        has_creation_actions = False
        has_file_actions = False

        # Creation/Edit actions group - use ActionManager's affect_to
        for action_name in ['new', 'edit', 'duplicate']:
            if action_name in actions:
                self.affect_to(action_name, self._menu)
                has_creation_actions = True

        # Separator before load menu
        if has_creation_actions and any(a in actions for a in ['load', 'delete', 'refresh', 'open_dir']):
            self._menu.addSeparator()

        # Load submenu - USE ActionManager's add_submenu for consistency
        if 'load' in actions:
            # Check if submenu already exists (in case create_menu is called multiple times)
            if self.has_submenu('load_submenu'):
                # Reuse existing submenu
                self._load_submenu = self.get_submenu('load_submenu')
                # Add to current menu
                self._menu.addMenu(self._load_submenu)
            else:
                # Create new submenu
                self._load_submenu = self.add_submenu(
                    'load_submenu',
                    f"Load {self.title}",
                    menu=self._menu,
                    icon_name=qta.icon('mdi.folder-open'),
                    auto_menu=False  # We explicitly pass the parent menu
                )
            self._populate_load_menu()
            has_file_actions = True

        # Separator before management actions
        if has_file_actions and any(a in actions for a in ['delete', 'refresh', 'open_dir']):
            self._menu.addSeparator()

        # File management actions
        for action_name in ['delete', 'refresh', 'open_dir']:
            if action_name in actions:
                self.affect_to(action_name, self._menu)

        return self._menu

    def _menu_new_config(self):
        """Menu action: Create a new configuration"""
        self.set_new_config(show=True)

    def _menu_edit_current_config(self):
        """Menu action: Edit the currently loaded configuration"""
        if not hasattr(self, 'settings') or self.settings is None:
            logger.warning("No configuration loaded to edit")
            QMessageBox.warning(
                None,
                "No Configuration Loaded",
                f"Please create a new {self.title} or load an existing one first."
            )
            return

        # Show the config dialog to edit current settings
        success = self.show_config()
        if success:
            logger.info(f"Configuration '{self.settings.child('filename').value()}' updated")

    def _menu_duplicate_config(self):
        """Menu action: Duplicate current configuration with a new name"""
        if not hasattr(self, 'settings') or self.settings is None:
            logger.warning("No configuration loaded to duplicate")
            QMessageBox.warning(
                None,
                "No Configuration Loaded",
                f"Please create a new {self.title} or load an existing one first."
            )
            return

        # Prompt for new filename
        current_filename = self.settings.child("filename").value()
        new_filename, ok = QtWidgets.QInputDialog.getText(
            None,
            f"Duplicate {self.title}",
            "Enter name for the duplicate:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            f"{current_filename}_copy"
        )

        if ok and new_filename:
            # Update filename in settings
            old_filename = current_filename
            self.settings.child("filename").setValue(new_filename)

            # Show dialog to edit the duplicate before saving
            success = self.show_config(overwrite=False)

            if success:
                logger.info(f"Configuration duplicated: {old_filename} -> {new_filename}")
            else:
                # Restore original filename if user cancelled
                self.settings.child("filename").setValue(old_filename)

    def _menu_delete_config(self):
        """Menu action: Delete a configuration file after confirmation"""
        # Get list of available configs
        config_files = self._get_config_files()

        if not config_files:
            QMessageBox.information(
                None,
                "No Configurations",
                f"No {self.title} configuration files found in {self.config_path}"
            )
            return

        # Show selection dialog
        file_names = [f.stem for f in config_files]
        file_name, ok = QtWidgets.QInputDialog.getItem(
            None,
            f"Delete {self.title}",
            "Select configuration to delete:",
            file_names,
            0,
            False
        )

        if not ok or not file_name:
            return

        # Confirm deletion
        confirm = dialogbox(
            title="Confirm Deletion",
            message=f"Are you sure you want to delete '{file_name}.xml'?\n\nThis action cannot be undone."
        )

        if confirm:            
            file_to_delete = self.config_path.joinpath(f"{file_name}.xml")
            try:
                file_to_delete.unlink()
                logger.info(f"Deleted configuration: {file_to_delete}")
                self.config_deleted.emit(file_to_delete)
                QMessageBox.information(
                    None,
                    "Deleted",
                    f"Configuration '{file_name}' has been deleted."
                )
            except Exception as e:
                logger.exception(f"Failed to delete {file_to_delete}: {e}")
                QMessageBox.critical(
                    None,
                    "Delete Failed",
                    f"Failed to delete configuration:\n{str(e)}"
                )

    def _menu_refresh_list(self):
        """Menu action: Refresh the Load submenu"""
        self._populate_load_menu()
        logger.info("Configuration list refreshed")

    def _menu_open_config_dir(self):
        """Menu action: Open configuration directory in file explorer"""
        if not self.config_path or not isinstance(self.config_path, Path):
            QMessageBox.warning(
                None,
                "No Config Directory",
                "Configuration directory path is not set."
            )
            return

        # Create directory if it doesn't exist
        if not self.config_path.exists():
            try:
                self.config_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created config directory: {self.config_path}")
            except Exception as e:
                logger.exception(f"Failed to create directory: {e}")
                QMessageBox.critical(
                    None,
                    "Directory Error",
                    f"Failed to create config directory:\n{str(e)}"
                )
                return

        # Open directory in system file explorer
        try:
            from qtpy.QtGui import QDesktopServices
            from qtpy.QtCore import QUrl

            url = QUrl.fromLocalFile(str(self.config_path))
            success = QDesktopServices.openUrl(url)

            if success:
                logger.info(f"Opened config directory: {self.config_path}")
            else:
                logger.warning(f"Failed to open directory: {self.config_path}")
                QMessageBox.warning(
                    None,
                    "Cannot Open Directory",
                    f"Failed to open directory in file explorer:\n{self.config_path}\n\n"
                    f"You can navigate to it manually."
                )
        except Exception as e:
            logger.exception(f"Error opening directory: {e}")
            QMessageBox.critical(
                None,
                "Error",
                f"Error opening directory:\n{str(e)}"
            )

    def _populate_load_menu(self):
        """Populate the Load submenu with available configuration files"""
        if not hasattr(self, '_load_submenu'):
            return

        # Clear existing items
        self._load_submenu.clear()

        # Get all config files
        config_files = self._get_config_files()

        if not config_files:
            # Add disabled "No configs" item
            no_configs_action = self._load_submenu.addAction("(No configurations found)")
            no_configs_action.setEnabled(False)
            return

        # Add action for each config file
        for config_file in sorted(config_files, key=lambda f: f.stem):
            action = self._load_submenu.addAction(config_file.stem)
            # Use lambda with default argument to capture config_file correctly
            action.triggered.connect(
                lambda checked=False, path=config_file: self._menu_load_config(path)
            )

    def _menu_load_config(self, file_path: Path):
        """
        Menu action: Load a specific configuration file

        Args:
            file_path (Path): Path to the configuration file to load
        """
        try:
            success = self.set_config_from_file(file_path, show=False)
            if success:
                logger.info(f"Loaded configuration: {file_path.stem}")
        except Exception as e:
            logger.exception(f"Failed to load {file_path}: {e}")
            QMessageBox.critical(
                None,
                "Load Failed",
                f"Failed to load configuration:\n{str(e)}"
            )

    def _get_config_files(self) -> List[Path]:
        """
        Get list of all XML configuration files in config_path

        Returns:
            List[Path]: List of Path objects for all .xml files in config_path
        """
        if not self.config_path or not isinstance(self.config_path, Path):
            return []

        if not self.config_path.exists():
            logger.warning(f"Config path does not exist: {self.config_path}")
            return []

        try:
            return [f for f in self.config_path.iterdir() if f.suffix == ".xml"]
        except Exception as e:
            logger.exception(f"Error reading config directory: {e}")
            return []
