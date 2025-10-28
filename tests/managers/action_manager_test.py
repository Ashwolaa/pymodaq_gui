# -*- coding: utf-8 -*-
"""
Created the 08/11/2024

@author: Constant Schouder
"""

from packaging.version import Version

import pytest
from qtpy import QtWidgets, QtGui, QtCore
from pymodaq_gui.managers.action_manager import ActionManager


version_qt = QtCore.qVersion()


@pytest.fixture
def ini_qt_widget(init_qt):
    qtbot = init_qt
    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    widget.show()
    yield qtbot, widget
    widget.close()


def is_icon_null(
    action_manager,
    action_name,
):
    action = action_manager.get_action(action_name)
    return action.icon().isNull()


def test_icon(qtbot):
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=QtWidgets.QMenu())

    action_manager.add_action(short_name="no_icon", name="my_no_icon", icon_name="")

    assert is_icon_null(action_manager, "no_icon")

    action_manager.add_action(
        short_name="icon_from_pymodaq", name="an_icon_from_pymodaq", icon_name="NewFile"
    )
    assert not is_icon_null(action_manager, "icon_from_pymodaq")

    if Version(version_qt) > Version("6.7"):
        action_manager.add_action(
            short_name="icon_from_Qt", name="an_icon_from_Qt", icon_name="WindowClose"
        )
        assert not is_icon_null(action_manager, "icon_from_Qt")

        icon = QtGui.QIcon.fromTheme(QtGui.QIcon.ThemeIcon.WindowClose)
        action_manager.add_action(
            short_name="icon", name="an_icon_from_Qt", icon_name=icon
        )
        assert not is_icon_null(action_manager, "icon")



def test_action_properties(qtbot):
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=QtWidgets.QMenu())

    action_manager.add_action(short_name="no_icon", name="my_no_icon", icon_name="")
    action_manager.add_action(
        short_name="icon_from_pymodaq", name="an_icon_from_pymodaq", icon_name="NewFile"
    )

    assert action_manager.get_action('no_icon') == action_manager._actions['no_icon']

    assert 'no_icon' in action_manager.actions_names
    assert 'icon_from_pymodaq' in action_manager.actions_names

    assert action_manager.get_action('no_icon') in action_manager.actions
    assert isinstance(action_manager.get_action('icon_from_pymodaq'), QtWidgets.QAction)


def test_submenu_creation(qtbot):
    """Test creating submenus"""
    menu = QtWidgets.QMenu()
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=menu)

    # Create a simple submenu
    submenu = action_manager.add_submenu('file_menu', 'File')

    assert action_manager.has_submenu('file_menu')
    assert action_manager.get_submenu('file_menu') == submenu
    assert 'file_menu' in action_manager.submenus_names
    assert submenu in action_manager.submenus
    assert isinstance(submenu, QtWidgets.QMenu)
    assert submenu.title() == 'File'


def test_submenu_with_icon(qtbot):
    """Test creating submenus with icons"""
    menu = QtWidgets.QMenu()
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=menu)

    # Create submenu with icon
    submenu = action_manager.add_submenu('edit_menu', 'Edit', icon_name='NewFile')

    assert not submenu.icon().isNull()


def test_nested_submenus(qtbot):
    """Test creating nested submenus (submenu within submenu)"""
    menu = QtWidgets.QMenu()
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=menu)

    # Create parent submenu
    parent_submenu = action_manager.add_submenu('parent', 'Parent Menu')

    # Create child submenu within parent
    child_submenu = action_manager.add_submenu('child', 'Child Menu', menu=parent_submenu)

    assert action_manager.has_submenu('parent')
    assert action_manager.has_submenu('child')

    # Check that child submenu is in parent's menu actions
    # Qt wraps submenus in QActions, so we check if any action's menu() returns our child
    parent_menus = [action.menu() for action in parent_submenu.actions() if action.menu() is not None]
    assert child_submenu in parent_menus


def test_add_action_to_submenu_by_name(qtbot):
    """Test adding actions to submenu using submenu name"""
    menu = QtWidgets.QMenu()
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=menu)

    # Create submenu
    action_manager.add_submenu('file_menu', 'File')

    # Add action to submenu using name
    action_manager.add_action('open', 'Open', icon_name='Open', submenu='file_menu')

    assert action_manager.has_action('open')
    submenu = action_manager.get_submenu('file_menu')
    action = action_manager.get_action('open')
    assert action in submenu.actions()


def test_add_action_to_submenu_by_object(qtbot):
    """Test adding actions to submenu using QMenu object"""
    menu = QtWidgets.QMenu()
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=menu)

    # Create submenu
    submenu = action_manager.add_submenu('edit_menu', 'Edit')

    # Add action to submenu using QMenu object
    action_manager.add_action('copy', 'Copy', submenu=submenu)

    assert action_manager.has_action('copy')
    action = action_manager.get_action('copy')
    assert action in submenu.actions()


def test_multiple_actions_in_submenu(qtbot):
    """Test adding multiple actions to the same submenu"""
    menu = QtWidgets.QMenu()
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=menu)

    # Create submenu
    action_manager.add_submenu('file_menu', 'File')

    # Add multiple actions
    action_manager.add_action('new', 'New', submenu='file_menu')
    action_manager.add_action('open', 'Open', submenu='file_menu')
    action_manager.add_action('save', 'Save', submenu='file_menu')

    submenu = action_manager.get_submenu('file_menu')
    assert len(submenu.actions()) == 3


def test_submenu_without_auto_menu(qtbot):
    """Test creating submenu without automatically adding it to parent menu"""
    menu = QtWidgets.QMenu()
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=menu)

    # Create submenu without auto-adding
    submenu = action_manager.add_submenu('floating', 'Floating Menu', auto_menu=False)

    assert action_manager.has_submenu('floating')
    # Submenu should not be in the main menu's actions
    menu_submenus = [action.menu() for action in menu.actions() if action.menu() is not None]
    assert submenu not in menu_submenus


def test_submenu_getter_errors(qtbot):
    """Test error handling for submenu getters"""
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=QtWidgets.QMenu())

    # Test getting non-existent submenu
    with pytest.raises(KeyError):
        action_manager.get_submenu('nonexistent')

    # Test has_submenu for non-existent
    assert not action_manager.has_submenu('nonexistent')


def test_add_action_invalid_submenu_type(qtbot):
    """Test error handling when passing invalid submenu type"""
    menu = QtWidgets.QMenu()
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=menu)

    # Try to add action with invalid submenu type
    with pytest.raises(TypeError):
        action_manager.add_action('test', 'Test Action', submenu=123)


def test_shared_submenu(qtbot):
    """Test that the same submenu can be added to multiple parent menus"""
    menu = QtWidgets.QMenu()
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=menu)

    # Create two parent menus
    file_menu = action_manager.add_submenu('file', 'File')
    view_menu = action_manager.add_submenu('view', 'View')

    # Create a shared submenu (add to file menu first)
    shared_submenu = action_manager.add_submenu('recent', 'Recent Files', menu=file_menu)

    # Add the same submenu to view menu
    view_menu.addMenu(shared_submenu)

    # Verify it appears in both menus
    file_submenus = [action.menu() for action in file_menu.actions() if action.menu() is not None]
    view_submenus = [action.menu() for action in view_menu.actions() if action.menu() is not None]

    assert shared_submenu in file_submenus
    assert shared_submenu in view_submenus

    # Add action to shared submenu
    action_manager.add_action('recent_1', 'Project1.py', submenu=shared_submenu)

    # Verify action appears in the shared submenu (accessible from both parent menus)
    assert len(shared_submenu.actions()) == 1
    assert action_manager.get_action('recent_1') in shared_submenu.actions()


def test_action_path_simple(qtbot):
    """Test getting path for action in simple submenu"""
    menu = QtWidgets.QMenu()
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=menu)

    # Create submenu and add action
    action_manager.add_submenu('file_menu', 'File')
    action_manager.add_action('open', 'Open', submenu='file_menu')

    path = action_manager.get_action_path('open')
    assert path == 'File'


def test_action_path_nested(qtbot):
    """Test getting path for action in nested submenus"""
    menu = QtWidgets.QMenu()
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=menu)

    # Create nested submenus
    file_menu = action_manager.add_submenu('file_menu', 'File')
    recent_menu = action_manager.add_submenu('recent_menu', 'Recent Files', menu=file_menu)

    # Add action to nested submenu
    action_manager.add_action('recent_1', 'Project1.py', submenu=recent_menu)

    path = action_manager.get_action_path('recent_1')
    assert path == 'File > Recent Files'


def test_action_path_deeply_nested(qtbot):
    """Test getting path for action in deeply nested submenus"""
    menu = QtWidgets.QMenu()
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=menu)

    # Create deeply nested submenus
    level1 = action_manager.add_submenu('level1', 'Level 1')
    level2 = action_manager.add_submenu('level2', 'Level 2', menu=level1)
    level3 = action_manager.add_submenu('level3', 'Level 3', menu=level2)

    # Add action to deepest level
    action_manager.add_action('deep_action', 'Deep Action', submenu=level3)

    path = action_manager.get_action_path('deep_action')
    assert path == 'Level 1 > Level 2 > Level 3'


def test_action_path_custom_separator(qtbot):
    """Test getting path with custom separator"""
    menu = QtWidgets.QMenu()
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=menu)

    # Create nested submenus
    file_menu = action_manager.add_submenu('file_menu', 'File')
    recent_menu = action_manager.add_submenu('recent_menu', 'Recent', menu=file_menu)
    action_manager.add_action('recent_1', 'Project1', submenu=recent_menu)

    path = action_manager.get_action_path('recent_1', separator=' / ')
    assert path == 'File / Recent'


def test_action_path_not_in_menu(qtbot):
    """Test getting path for action not in any menu"""
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=QtWidgets.QMenu())

    # Add action without submenu (only to toolbar or no menu)
    action_manager.add_action('toolbar_only', 'Toolbar Only', auto_menu=False)

    path = action_manager.get_action_path('toolbar_only')
    assert path == ''


def test_action_path_nonexistent_action(qtbot):
    """Test error handling for nonexistent action"""
    action_manager = ActionManager(toolbar=QtWidgets.QToolBar(), menu=QtWidgets.QMenu())

    with pytest.raises(KeyError):
        action_manager.get_action_path('nonexistent')