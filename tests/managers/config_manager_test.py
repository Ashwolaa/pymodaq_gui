# -*- coding: utf-8 -*-
"""
Tests for ConfigManager

@author: Test Suite
"""

import pytest
from pathlib import Path
from qtpy import QtWidgets, QtCore
from qtpy.QtWidgets import QMessageBox, QDialog

from pymodaq_gui.managers.config_manager import ConfigManager
from pymodaq_gui.parameter import Parameter

@pytest.fixture
def init_qt(qtbot):
    """Basic Qt fixture for compatibility with other tests"""
    return qtbot


@pytest.fixture(autouse=True)
def mock_dialogs(monkeypatch):
    """Automatically mock all dialog boxes to prevent popups during tests"""
    # This prevents overwrite confirmation dialogs from appearing
    monkeypatch.setattr('pymodaq_gui.managers.config_manager.dialogbox', lambda *args, **kwargs: True)


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary directory for config files"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def config_manager(temp_config_dir, qtbot):
    """Create a basic ConfigManager instance for testing"""
    return ConfigManager(config_path=temp_config_dir, msgbox=False)


class TestConfigManager:
    """Test basic ConfigManager functionality"""

    def test_initialization(self, temp_config_dir, qtbot):
        """Test ConfigManager initialization"""
        manager = ConfigManager(config_path=temp_config_dir, msgbox=False)

        assert manager.config_path == temp_config_dir
        assert manager.title == "Config"
        assert manager.name == "config"
        assert hasattr(manager, 'config_loaded')
        assert hasattr(manager, 'config_saved')
        assert hasattr(manager, 'config_deleted')

    def test_initialization_with_path_conversion(self, qtbot):
        """Test that string paths are properly handled"""
        manager = ConfigManager(config_path="/tmp/test", msgbox=False)
        assert manager.config_path == "/tmp/test"

    def test_make_config_default(self, config_manager:ConfigManager):
        """Test that make_config returns empty list by default"""
        result = config_manager.make_config()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_validate_config_default(self, config_manager:ConfigManager):
        """Test that validate_config returns True by default"""
        result = config_manager.validate_config()
        assert result is True

    def test_actions_setup(self, config_manager:ConfigManager):
        """Test that standard actions are created"""
        assert config_manager.has_action('new')
        assert config_manager.has_action('edit')
        assert config_manager.has_action('duplicate')
        assert config_manager.has_action('refresh')
        assert config_manager.has_action('open_dir')


class TestConfigCreation:
    """Test configuration creation functionality"""

    def test_set_new_config_default(self, config_manager:ConfigManager):
        """Test creating a new config with default settings"""
        config_manager.set_new_config(show=False)

        assert config_manager.settings is not None
        assert config_manager.settings.name() == "config"
        assert config_manager.settings.child('filename') is not None
        assert config_manager.settings.child('filename').value() == "Config_default"

    def test_set_new_config_custom_filename(self, config_manager:ConfigManager):
        """Test creating a new config with custom filename"""
        config_manager.set_new_config(file="my_custom_config", show=False)

        assert config_manager.settings.child('filename').value() == "my_custom_config"

    def test_set_new_config_with_additional_params(self, temp_config_dir, qtbot):
        """Test creating a new config with additional parameters from make_config"""

        class CustomConfigManager(ConfigManager):
            def make_config(self):
                return [
                    {'title': 'Test Param', 'name': 'test_param', 'type': 'int', 'value': 42}
                ]

        manager = CustomConfigManager(config_path=temp_config_dir, msgbox=False)
        manager.set_new_config(show=False)

        assert manager.settings.child('test_param') is not None
        assert manager.settings.child('test_param').value() == 42


class TestConfigSaving:
    """Test configuration saving functionality"""

    def test_save_config_basic(self, config_manager:ConfigManager, temp_config_dir):
        """Test basic config saving"""
        config_manager.set_new_config(file="test_config", show=False)
        result = config_manager.save_config(overwrite=True)

        assert result is True
        saved_file = temp_config_dir / "test_config.xml"
        assert saved_file.exists()

    def test_save_config_signal_emission(self, config_manager:ConfigManager, temp_config_dir, qtbot):
        """Test that config_saved signal is emitted"""
        config_manager.set_new_config(file="test_signal", show=False)

        with qtbot.waitSignal(config_manager.config_saved, timeout=1000) as blocker:
            config_manager.save_config(overwrite=True)

        assert blocker.signal_triggered
        emitted_path = blocker.args[0]
        assert emitted_path.name == "test_signal.xml"

    def test_save_config_overwrite_existing(self, config_manager:ConfigManager, temp_config_dir):
        """Test overwriting an existing config file"""
        # Create and save initial config
        config_manager.set_new_config(file="overwrite_test", show=False)
        config_manager.save_config(overwrite=True)

        # Modify and save again with overwrite=True
        config_manager.settings.child('filename').setValue('overwrite_test')
        result = config_manager.save_config(overwrite=True)

        assert result is True


class TestConfigLoading:
    """Test configuration loading functionality"""

    def test_load_config_from_file(self, config_manager:ConfigManager, temp_config_dir):
        """Test loading a config from an XML file"""
        # First create and save a config
        config_manager.set_new_config(file="load_test", show=False)
        config_manager.save_config(overwrite=True)

        # Create a new manager and load the config
        new_manager = ConfigManager(config_path=temp_config_dir, msgbox=False)
        file_path = temp_config_dir / "load_test.xml"
        result = new_manager.set_config_from_file(file_path, show=False)

        assert result is True
        assert new_manager.settings is not None
        assert new_manager.settings.child('filename').value() == "load_test"

    def test_load_config_signal_emission(self, config_manager:ConfigManager, temp_config_dir, qtbot):
        """Test that config_loaded signal is emitted"""
        # Create and save a config
        config_manager.set_new_config(file="signal_test", show=False)
        config_manager.save_config(overwrite=True)

        # Create new manager and load with signal check
        new_manager = ConfigManager(config_path=temp_config_dir, msgbox=False)
        file_path = temp_config_dir / "signal_test.xml"

        with qtbot.waitSignal(new_manager.config_loaded, timeout=1000) as blocker:
            new_manager.set_config_from_file(file_path, show=False)

        assert blocker.signal_triggered

    def test_load_config_invalid_extension(self, config_manager:ConfigManager, temp_config_dir, caplog):
        """Test loading a file with invalid extension"""
        invalid_file = temp_config_dir / "invalid.txt"
        invalid_file.touch()

        result = config_manager.set_config_from_file(invalid_file, show=False)

        assert result is False

    def test_load_config_with_string_path(self, config_manager, temp_config_dir):
        """Test loading config with string path (auto-conversion to Path)"""
        config_manager.set_new_config(file="string_path_test", show=False)
        config_manager.save_config(overwrite=True)

        new_manager = ConfigManager(config_path=temp_config_dir, msgbox=False)
        file_path_str = str(temp_config_dir / "string_path_test.xml")
        result = new_manager.set_config_from_file(file_path_str, show=False)

        assert result is True


class TestConfigValidation:
    """Test configuration validation functionality"""

    def test_validation_success(self, temp_config_dir, qtbot):
        """Test successful validation"""

        class ValidatingConfigManager(ConfigManager):
            def validate_config(self):
                return True

        manager = ValidatingConfigManager(config_path=temp_config_dir, msgbox=False)
        manager.set_new_config(file="valid_config", show=False)
        manager.save_config(overwrite=True)

        file_path = temp_config_dir / "valid_config.xml"
        result = manager.set_config_from_file(file_path, show=False)

        assert result is True

    def test_validation_failure(self, temp_config_dir, monkeypatch, qtbot):
        """Test failed validation prevents loading"""

        class FailingConfigManager(ConfigManager):
            def validate_config(self):
                return False

        manager = FailingConfigManager(config_path=temp_config_dir, msgbox=False)
        manager.set_new_config(file="invalid_config", show=False)

        # Save with different manager to bypass validation
        base_manager = ConfigManager(config_path=temp_config_dir, msgbox=False)
        base_manager.settings = manager.settings
        base_manager.save_config(overwrite=True)

        # Try to load with validating manager
        file_path = temp_config_dir / "invalid_config.xml"

        # Mock QMessageBox to avoid GUI interaction
        def mock_critical(*args, **kwargs):
            pass
        monkeypatch.setattr(QMessageBox, 'critical', mock_critical)

        result = manager.set_config_from_file(file_path, show=False)
        assert result is False


class TestConfigMenus:
    """Test menu creation and management"""

    def test_create_menu_default(self, config_manager:ConfigManager):
        """Test creating menu with default settings"""
        menu = config_manager.create_menu()

        assert menu is not None
        assert menu.title() == "Config Configs"
        assert len(menu.actions()) > 0

    def test_create_menu_custom_title(self, config_manager:ConfigManager):
        """Test creating menu with custom title"""
        menu = config_manager.create_menu(menu_title="Custom Title")

        assert menu.title() == "Custom Title"

    def test_create_menu_with_menubar(self, config_manager:ConfigManager, qtbot):
        """Test creating menu attached to a menubar"""
        menubar = QtWidgets.QMenuBar()
        menu = config_manager.create_menu(menubar=menubar)

        # Menu should be added to menubar
        assert menu in menubar.findChildren(QtWidgets.QMenu)

    def test_create_menu_limited_actions(self, config_manager:ConfigManager):
        """Test creating menu with only specific actions"""
        menu = config_manager.create_menu(actions=['new', 'load'])

        # Should have actions but limited set
        assert menu is not None
        # We can't easily count actions because some are in submenus

    def test_populate_menus_empty_directory(self, config_manager:ConfigManager):
        """Test populating menus when no config files exist"""
        config_manager.create_menu()
        config_manager._populate_menus()

        # Should not raise an error even with no config files
        assert hasattr(config_manager, '_load_menu')

    def test_populate_menus_with_configs(self, config_manager:ConfigManager):
        """Test populating menus with existing config files"""
        # Create some config files
        config_manager.set_new_config(file="config1", show=False)
        config_manager.save_config(overwrite=True)
        config_manager.set_new_config(file="config2", show=False)
        config_manager.save_config(overwrite=True)

        # Create menu and populate
        config_manager.create_menu()
        config_files = config_manager._get_config_files()

        assert len(config_files) == 2


class TestConfigFileOperations:
    """Test file operations like delete, list, etc."""

    def test_get_config_files_empty(self, config_manager:ConfigManager):
        """Test getting config files from empty directory"""
        files = config_manager._get_config_files()
        assert files == []

    def test_get_config_files_with_configs(self, config_manager:ConfigManager, temp_config_dir):
        """Test getting list of config files"""
        # Create some configs
        config_manager.set_new_config(file="config1", show=False)
        config_manager.save_config(overwrite=True)
        config_manager.set_new_config(file="config2", show=False)
        config_manager.save_config(overwrite=True)

        # Create a non-xml file to ensure it's filtered
        (temp_config_dir / "notconfig.txt").touch()

        files = config_manager._get_config_files()

        assert len(files) == 2
        assert all(f.suffix == '.xml' for f in files)

    def test_get_config_files_invalid_path(self, caplog, qtbot):
        """Test getting config files with invalid path"""
        manager = ConfigManager(config_path=Path("/nonexistent/path"), msgbox=False)
        files = manager._get_config_files()

        assert files == []

    def test_delete_config_file(self, config_manager:ConfigManager, temp_config_dir, qtbot, monkeypatch):
        """Test deleting a config file"""
        # Create a config
        config_manager.set_new_config(file="to_delete", show=False)
        config_manager.save_config(overwrite=True)
        file_path = temp_config_dir / "to_delete.xml"
        assert file_path.exists()

        # Mock dialog to auto-confirm
        def mock_dialog(*args, **kwargs):
            return True
        monkeypatch.setattr('pymodaq_gui.managers.config_manager.dialogbox', mock_dialog)

        # Test deletion with signal
        with qtbot.waitSignal(config_manager.config_deleted, timeout=1000) as blocker:
            config_manager._menu_delete_config(file_path)

        assert blocker.signal_triggered
        assert not file_path.exists()


class TestConfigMenuActions:
    """Test menu action handlers"""

    def test_menu_new_config(self, config_manager:ConfigManager, monkeypatch):
        """Test new config menu action"""
        # Track if set_new_config was called
        called = []
        original_method = config_manager.set_new_config

        def mock_set_new_config(file=None, show=True):
            called.append(True)
            return original_method(file=file, show=False)

        monkeypatch.setattr(config_manager, 'set_new_config', mock_set_new_config)
        config_manager._menu_new_config()

        assert len(called) == 1

    def test_menu_edit_current_config_no_config(self, config_manager:ConfigManager, monkeypatch):
        """Test edit menu action with no config loaded"""
        # Mock QMessageBox to avoid GUI
        called = []
        def mock_warning(*args, **kwargs):
            called.append('warning')
        monkeypatch.setattr(QMessageBox, 'warning', mock_warning)

        # Remove settings if they exist
        if hasattr(config_manager, 'settings'):
            config_manager._settings = None

        config_manager._menu_edit_current_config()

        assert 'warning' in called

    def test_menu_edit_current_config_with_config(self, config_manager:ConfigManager, monkeypatch):
        """Test edit menu action with config loaded"""
        config_manager.set_new_config(file="edit_test", show=False)

        # Mock show_config to avoid GUI
        called = []
        def mock_show_config(widget=None, overwrite=False):
            called.append('show_config')
            return True
        monkeypatch.setattr(config_manager, 'show_config', mock_show_config)

        config_manager._menu_edit_current_config()

        assert 'show_config' in called

    def test_menu_duplicate_config_no_config(self, config_manager:ConfigManager, monkeypatch):
        """Test duplicate menu action with no config"""
        called = []
        def mock_warning(*args, **kwargs):
            called.append('warning')
        monkeypatch.setattr(QMessageBox, 'warning', mock_warning)

        config_manager._settings = None
        config_manager._menu_duplicate_config()

        assert 'warning' in called

    def test_menu_load_config(self, config_manager:ConfigManager, temp_config_dir):
        """Test load config menu action"""
        # Create and save a config
        config_manager.set_new_config(file="load_via_menu", show=False)
        config_manager.save_config(overwrite=True)

        # Create new manager and load via menu action
        new_manager = ConfigManager(config_path=temp_config_dir, msgbox=False)
        file_path = temp_config_dir / "load_via_menu.xml"

        new_manager._menu_load_config(file_path)

        assert new_manager.settings is not None
        assert new_manager.settings.child('filename').value() == "load_via_menu"

    def test_menu_refresh_list(self, config_manager:ConfigManager):
        """Test refresh list menu action"""
        # Create menu first
        config_manager.create_menu()

        # Should not raise error
        config_manager._menu_refresh_list()


class TestCustomConfigManager:
    """Test with custom ConfigManager subclasses"""

    def test_custom_title_and_name(self, temp_config_dir, qtbot):
        """Test custom title and name attributes"""

        class CustomConfigManager(ConfigManager):
            title = "MyCustom"
            name = "my_custom"

        manager = CustomConfigManager(config_path=temp_config_dir, msgbox=False)

        assert manager.title == "MyCustom"
        assert manager.name == "my_custom"

    def test_custom_make_config(self, temp_config_dir, qtbot):
        """Test custom make_config implementation"""

        class CustomConfigManager(ConfigManager):
            def make_config(self):
                return [
                    {'title': 'Host', 'name': 'host', 'type': 'str', 'value': 'localhost'},
                    {'title': 'Port', 'name': 'port', 'type': 'int', 'value': 8080},
                ]

        manager = CustomConfigManager(config_path=temp_config_dir, msgbox=False)
        manager.set_new_config(show=False)

        assert manager.settings.child('host') is not None
        assert manager.settings.child('port') is not None
        assert manager.settings.child('host').value() == 'localhost'
        assert manager.settings.child('port').value() == 8080

    def test_custom_validation(self, temp_config_dir, qtbot):
        """Test custom validate_config implementation"""

        class CustomConfigManager(ConfigManager):
            def make_config(self):
                return [
                    {'title': 'Value', 'name': 'value', 'type': 'int', 'value': 10},
                ]

            def validate_config(self):
                value = self.settings.child('value').value()
                return value >= 0

        manager = CustomConfigManager(config_path=temp_config_dir, msgbox=False)
        manager.set_new_config(show=False)

        # Test valid value
        assert manager.validate_config() is True

        # Test invalid value
        manager.settings.child('value').setValue(-5)
        assert manager.validate_config() is False


class TestSignals:
    """Test signal emissions"""

    def test_config_loaded_signal(self, config_manager:ConfigManager, temp_config_dir, qtbot):
        """Test config_loaded signal is emitted correctly"""
        config_manager.set_new_config(file="signal_load_test", show=False)
        config_manager.save_config(overwrite=True)

        new_manager = ConfigManager(config_path=temp_config_dir, msgbox=False)
        file_path = temp_config_dir / "signal_load_test.xml"

        with qtbot.waitSignal(new_manager.config_loaded, timeout=1000) as blocker:
            new_manager.set_config_from_file(file_path, show=False)

        assert blocker.args[0] == file_path

    def test_config_saved_signal(self, config_manager:ConfigManager, qtbot):
        """Test config_saved signal is emitted correctly"""
        config_manager.set_new_config(file="signal_save_test", show=False)

        with qtbot.waitSignal(config_manager.config_saved, timeout=1000) as blocker:
            config_manager.save_config(overwrite=True)

        assert blocker.args[0].stem == "signal_save_test"

    def test_config_deleted_signal(self, config_manager:ConfigManager, temp_config_dir, qtbot, monkeypatch):
        """Test config_deleted signal is emitted correctly"""
        # Create a config file
        config_manager.set_new_config(file="signal_delete_test", show=False)
        config_manager.save_config(overwrite=True)
        file_path = temp_config_dir / "signal_delete_test.xml"

        # Mock dialog to auto-confirm
        def mock_dialog(*args, **kwargs):
            return True
        monkeypatch.setattr('pymodaq_gui.managers.config_manager.dialogbox', mock_dialog)

        with qtbot.waitSignal(config_manager.config_deleted, timeout=1000) as blocker:
            config_manager._menu_delete_config(file_path)

        assert blocker.args[0] == file_path


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_config_path_none(self, qtbot):
        """Test behavior with None config_path"""
        manager = ConfigManager(config_path=None, msgbox=False)
        files = manager._get_config_files()
        assert files == []

    def test_config_path_nonexistent(self, qtbot):
        """Test behavior with non-existent config_path"""
        manager = ConfigManager(config_path=Path("/nonexistent/path"), msgbox=False)
        files = manager._get_config_files()
        assert files == []

    def test_save_config_returns_false_on_cancel(self, config_manager:ConfigManager, monkeypatch):
        """Test that save_config returns False when user cancels overwrite"""
        # Create existing file
        config_manager.set_new_config(file="existing", show=False)
        config_manager.save_config(overwrite=True)

        # Try to save again without overwrite flag
        # Mock dialog to return False (user cancels)
        def mock_dialog(*args, **kwargs):
            return False
        monkeypatch.setattr('pymodaq_gui.managers.config_manager.dialogbox', mock_dialog)

        result = config_manager.save_config(overwrite=False)
        assert result is False

    def test_multiple_saves_same_config(self, config_manager:ConfigManager):
        """Test saving the same config multiple times"""
        config_manager.set_new_config(file="multi_save", show=False)

        result1 = config_manager.save_config(overwrite=True)
        result2 = config_manager.save_config(overwrite=True)
        result3 = config_manager.save_config(overwrite=True)

        assert result1 is True
        assert result2 is True
        assert result3 is True
