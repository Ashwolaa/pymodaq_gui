import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, Iterable as IterableType, Union

from multipledispatch import dispatch
from pymodaq_utils.warnings import deprecation_msg
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import QAction as QtQAction

here = Path(__file__).parent
icon_folder = here.parent.joinpath('QtDesigner_Ressources/Icon_Library/')
QtCore.QDir.addSearchPath('icons', str(icon_folder))

def create_icon(icon_name: Union[str, Path]):
    icon = QtGui.QIcon()
    if Path(icon_name).is_file(): # Test if icon is in path
        icon.addPixmap(QtGui.QPixmap(icon_name), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
    else:
        pixmap = QtGui.QPixmap(f"icons:{icon_name}.png") # Test if icon is in pymodaq's library
        if pixmap.isNull(): 
            if hasattr(QtGui.QIcon,'ThemeIcon') and hasattr(QtGui.QIcon.ThemeIcon, icon_name): # Test if icon is in Qt's library
                icon = QtGui.QIcon.fromTheme(getattr(QtGui.QIcon.ThemeIcon, icon_name))
            elif hasattr(QtWidgets.QStyle.StandardPixmap, icon_name):
                pixmapi = getattr(QtWidgets.QStyle.StandardPixmap, icon_name)
                icon = QtWidgets.QWidget().style().standardIcon(pixmapi)
        else:
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(pixmap), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
    return icon


class QAction(QtQAction):
    """
    QAction subclass to mimic signals as pushbuttons. Done to be sure of backcompatibility
    when I moved from pushbuttons to QAction
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def click(self):
        deprecation_msg("click for PyMoDAQ's QAction is deprecated, use *trigger*",
                        stacklevel=3)
        self.trigger()

    @property
    def clicked(self):
        deprecation_msg("clicked for PyMoDAQ's QAction is deprecated, use *trigger*",
                        stacklevel=3)
        return self.triggered

    def connect_to(self, slot):
        self.triggered.connect(slot)

    def set_icon(self, icon_name: str):
        self.setIcon(create_icon(icon_name))

    def __repr__(self):
        return f'QAction {self.text()}'


def addaction(name: str = '', icon_name: Union[str, Path, QtGui.QIcon]= '', tip='', checkable=False, checked=False,
              slot: Callable = None, toolbar: QtWidgets.QToolBar = None,
              menu: QtWidgets.QMenu = None, visible=True, shortcut=None,
              enabled=True):
    """Create a new action and add it eventually to a toolbar and a menu

    Parameters
    ----------
    name: str
        Displayed name if should be displayed (for instance in menus)
    icon_name: str / Path / QtGui.QIcon / enum name
        str/Path: the png file name/path to produce the icon
        QtGui.QIcon: the instance of a QIcon element
        ThemeIcon enum: the value of QtGui.QIcon.ThemeIcon (requires Qt>=6.7)
    tip: str
        a tooltip to be displayed when hovering above the action
    checkable: bool
        set the checkable state of the action
    checked: bool
        set the current state of the action
    slot: callable
        Method or function that will be called when the action is triggered
    toolbar: QToolBar
        a toolbar where action should be added.
    menu: QMenu
        a menu where action should be added.
    visible: bool
        display or not the action in the toolbar/menu
    shortcut: str
        a string defining a shortcut for this action
    enabled: bool
        set the enabled state
    """

    if icon_name is None or icon_name == '':
        action = QAction(name)
    elif isinstance(icon_name, QtGui.QIcon):
        action = QAction(icon_name, name, None)
    else:
        action = QAction(create_icon(icon_name), name, None)

    if slot is not None:
        action.connect_to(slot)
    action.setCheckable(checkable)
    if checkable:
        action.setChecked(checked)
    action.setToolTip(tip)
    if toolbar is not None:
        toolbar.addAction(action)
    if menu is not None:
        menu.addAction(action)
    if shortcut is not None:
        action.setShortcut(shortcut)
    action.setVisible(visible)
    action.setEnabled(enabled)
    return action


def addwidget(klass: Union[str, QtWidgets.QWidget, object], *args, tip='', toolbar: QtWidgets.QToolBar = None,
              visible=True,
              signal_str=None, slot: Callable=None, setters: dict = None, enabled=True, **kwargs):
    """Create and eventually add a widget to a toolbar

    Parameters
    ----------
    klass: str or QWidget or QWidget instance
        should be a custom widget class or the name of a standard widget of QWidgets
    args: list
     variable arguments passed as is to the widget constructor
    tip: str
        a tooltip to be displayed when hovering above the widget
    toolbar: QToolBar
        a toolbar where the widget should be added.
    visible: bool
        display or not the action in the toolbar/menu
    signal_str: str
        an attribute of type Signal of the widget
    slot: Callable
        a callable connected to the signal
    enabled: bool
        enable state of the widget
    kwargs: dict
        variable named arguments used as is in the widget constructor
    setters: dict
        method/value pair of the widget (for instance setMaximumWidth)
    Returns
    -------
    QtWidgets.QWidget
    """
    if setters is None:
        setters = {}
    if isinstance(klass, str):
        if hasattr(QtWidgets, klass):
            widget: QtWidgets.QWidget = getattr(QtWidgets, klass)(*args)
        else:
            return None
    elif isinstance(klass, QtWidgets.QWidget):
        widget = klass
    else:
        try:
            widget = klass(*args, **kwargs)
        except:
            return None

    if toolbar is not None:
        action: QtWidgets.QAction = toolbar.addWidget(widget)
        action.setVisible(visible)
        action.setToolTip(tip)
        widget.setVisible = action.setVisible #because visibility is only possible on the underlying QAction
    else:
        widget.setVisible(visible)
        widget.setToolTip(tip)

    if isinstance(signal_str, str) and slot is not None:
        if hasattr(widget, signal_str):
            getattr(widget, signal_str).connect(slot)

    for setter in setters:
        if hasattr(widget, setter):
            getattr(widget, setter)(setters[setter])
    widget.setEnabled(enabled)
    return widget


class ActionManager:
    """MixIn Class to be used by all UserInterface to manage their QActions and the action they are connected to

    Parameters
    ----------
    toolbar: QToolbar, optional
        The toolbar to use as default
    menu: QMenu, option
        The menu to use as default
    """
    def __init__(self, toolbar=None, menu=None):
        self._actions = dict([])
        self._submenus = dict([])
        self._toolbar = toolbar
        self._menu = menu

        #self.setup_actions()

    def setup_actions(self):
        """Method where to create actions to be subclassed. Mandatory

        Examples
        --------
        >>> self.add_action('Quit', 'close2', "Quit program")
        >>> self.add_action('Grab', 'camera', "Grab from camera", checkable=True)
        >>> self.add_action('Load', 'Open', "Load target file (.h5, .png, .jpg) or data from camera", checkable=False)
        >>> self.add_action('Save', 'SaveAs', "Save current data", checkable=False)

        See Also
        --------
        ActionManager.add_action
        """
        raise NotImplementedError(f'You have to define actions here in the following form:'
                                  f'{self.setup_actions.__doc__}')

    def add_action(self, short_name: str = '', name: str = '', icon_name: Union[str, Path, QtGui.QIcon] = '', tip='',
                   checkable=False,
                   checked=False, toolbar=None, menu=None, submenu: Union[str, QtWidgets.QMenu] = None,
                   visible=True, shortcut=None, auto_toolbar=True, auto_menu=True,
                   enabled=True):
        """Create a new action and add it to toolbar and menu

        Parameters
        ----------
        short_name: str
            the name as referenced in the dict self.actions
        name: str
            Displayed name if should be displayed in
        icon_name: str / Path / QtGui.QIcon / enum name
            str/Path: the png file name/path to produce the icon
            QtGui.QIcon: the instance of a QIcon element
            ThemeIcon enum: the value of QtGui.QIcon.ThemeIcon (requires Qt>=6.7)
        tip: str
            a tooltip to be displayed when hovering above the action
        checkable: bool
            set the checkable state of the action
        checked: bool
            set the current state of the action
        toolbar: QToolBar
            a toolbar where action should be added. Actions can also be added later see *affect_to*
        menu: QMenu
            a menu where action should be added. Actions can also be added later see *affect_to*
        submenu: str or QMenu
            If provided, the action will be added to this submenu instead of the main menu.
            Can be either a string (submenu name as registered) or a QMenu instance.
        visible: bool
            display or not the action in the toolbar/menu
        auto_toolbar: bool
            if True add this action to the defined toolbar
        auto_menu: bool
            if True add this action to the defined menu (or submenu if specified)
        enabled: bool
            set the enabled state of this action
        See Also
        --------
        affect_to, add_submenu, pymodaq.resources.QtDesigner_Ressources.Icon_Library,
        pymodaq.utils.managers.action_manager.add_action
        """
        if auto_toolbar:
            if toolbar is None:
                toolbar = self._toolbar
        if auto_menu:
            if menu is None:
                # If submenu is specified, resolve it to a QMenu object
                if submenu is not None:
                    if isinstance(submenu, str):
                        menu = self.get_submenu(submenu)
                    elif isinstance(submenu, QtWidgets.QMenu):
                        menu = submenu
                    else:
                        raise TypeError(f'submenu must be either a string or QMenu, got {type(submenu)}')
                else:
                    menu = self._menu
        self._actions[short_name] = addaction(name, icon_name, tip, checkable=checkable,
                                              checked=checked, toolbar=toolbar, menu=menu,
                                              visible=visible, shortcut=shortcut, enabled=enabled)

    def add_widget(self, short_name, klass: Union[str, QtWidgets.QWidget, object], *args, tip='',
                   toolbar: QtWidgets.QToolBar = None, visible=True, signal_str=None,
                   slot: Callable=None, enabled=True, **kwargs):
        """Create and add a widget to a toolbar

        Parameters
        ----------
        short_name: str
            the name as referenced in the dict self.actions
        klass: str or QWidget or QWidget instance
            should be a custom widget class or the name of a standard widget of QWidgets
        args: list
         variable arguments passed as is to the widget constructor
        tip: str
            a tooltip to be displayed when hovering above the widget
        toolbar: QToolBar
            a toolbar where the widget should be added.
        visible: bool
            display or not the action in the toolbar/menu
        signal_str: str
            an attribute of type Signal of the widget
        slot: Callable
            a callable connected to the signal
        enabled: bool
            enable state of the widget
        kwargs: dict
            variable named arguments passed as is to the widget constructor
        Returns
        -------
        QtWidgets.QWidget
        """
        if toolbar is None:
            toolbar = self._toolbar
        widget = addwidget(klass, *args, tip=tip, toolbar=toolbar, visible=visible, signal_str=signal_str,
                           slot=slot, enabled=enabled, **kwargs)
        if widget is not None:
            self._actions[short_name] = widget
        else:
            warnings.warn(UserWarning(f'Impossible to add the widget {short_name} and type {klass} to the toolbar'))

    def add_submenu(self, short_name: str, title: str, menu: QtWidgets.QMenu = None,
                    icon_name: Union[str, Path, QtGui.QIcon] = '', auto_menu=True) -> QtWidgets.QMenu:
        """Create and add a submenu to a parent menu

        Parameters
        ----------
        short_name: str
            the name as referenced in the dict self._submenus
        title: str
            Displayed title of the submenu
        menu: QMenu, optional
            a parent menu where this submenu should be added. If None, uses the default menu
        icon_name: str / Path / QtGui.QIcon / enum name, optional
            str/Path: the png file name/path to produce the icon
            QtGui.QIcon: the instance of a QIcon element
            ThemeIcon enum: the value of QtGui.QIcon.ThemeIcon (requires Qt>=6.7)
        auto_menu: bool
            if True add this submenu to the defined menu

        Returns
        -------
        QtWidgets.QMenu
            The created submenu

        See Also
        --------
        add_action, get_submenu
        """
        if auto_menu:
            if menu is None:
                menu = self._menu

        submenu = QtWidgets.QMenu(title)

        # Set icon if provided
        if icon_name and icon_name != '':
            if isinstance(icon_name, QtGui.QIcon):
                submenu.setIcon(icon_name)
            else:
                submenu.setIcon(create_icon(icon_name))

        # Add to parent menu if specified
        if menu is not None:
            menu.addMenu(submenu)

        # Store reference
        self._submenus[short_name] = submenu

        return submenu

    def set_toolbar(self, toolbar):
        """affect a toolbar to self

        Parameters
        ----------
        toolbar:
            QtWidgets.QToolBar
        """
        self._toolbar = toolbar

    def set_menu(self, menu):
        """affect a menu to self

        Parameters
        ----------
        menu:
            QtWidgets.QMenu
        """
        self._menu = menu

    def set_action_text(self, action_name: str, text: str):
        """Convenience method to set the displayed text on an action

        Parameters
        ----------
        action_name: str
            The action name as defined in setup_actions
        text: str
            The text to display
        """
        self.get_action(action_name).setText(text)

    @property
    def actions(self) -> list[QAction]:
        return list(self._actions.values())

    @property
    def actions_names(self) -> list[str]:
        return list(self._actions.keys())

    def get_action(self, name) -> QAction:
        """Getter of a given action

        Parameters
        ----------
        name: str
            The action name as defined in setup_actions

        Returns
        -------
        QAction
        """
        if self.has_action(name):
            return self._actions[name]
        else:
            raise KeyError(f'The action with name: {name} is not referenced'
                           f' in the view actions: {self._actions.keys()}')

    def get_action_path(self, action_name: str, separator: str = ' > ') -> str:
        """Get the menu path of an action (e.g., "File > Recent > Open")

        Parameters
        ----------
        action_name: str
            The action name as defined in setup_actions
        separator: str
            The separator to use between menu levels (default: ' > ')

        Returns
        -------
        str
            The full menu path of the action, or empty string if not in any menu

        Examples
        --------
        >>> action_manager.get_action_path('open')
        'File > Open'
        >>> action_manager.get_action_path('recent_1')
        'File > Recent Files > Project1'
        """
        if not self.has_action(action_name):
            raise KeyError(f'The action with name: {action_name} is not referenced'
                           f' in the view actions: {self._actions.keys()}')

        action = self._actions[action_name]
        path_parts = []

        # Find which menu(s) contain this action
        def find_action_in_menu(menu: QtWidgets.QMenu, action: QAction) -> list[str]:
            """Recursively search for action in menu hierarchy"""
            if menu is None:
                return []

            # Check if action is directly in this menu
            if action in menu.actions():
                return [menu.title()]

            # Check submenus
            for menu_action in menu.actions():
                if menu_action.menu() is not None:
                    submenu = menu_action.menu()
                    sub_path = find_action_in_menu(submenu, action)
                    if sub_path:
                        return [menu.title()] + sub_path

            return []

        # Search in all registered submenus
        for submenu in self._submenus.values():
            path = find_action_in_menu(submenu, action)
            if path:
                path_parts = path
                break

        # If not found in submenus, check main menu
        if not path_parts and self._menu is not None:
            path = find_action_in_menu(self._menu, action)
            if path:
                path_parts = path

        return separator.join(path_parts) if path_parts else ''

    def has_action(self, action_name) -> bool:
        """Check if an action has been defined
        Parameters
        ----------
        action_name: str
            The action name as defined in setup_actions

        Returns
        -------
        bool: True if the action exists, False otherwise
        """
        return action_name in self._actions

    def get_submenu(self, name: str) -> QtWidgets.QMenu:
        """Getter of a given submenu

        Parameters
        ----------
        name: str
            The submenu name as defined when calling add_submenu

        Returns
        -------
        QMenu
        """
        if self.has_submenu(name):
            return self._submenus[name]
        else:
            raise KeyError(f'The submenu with name: {name} is not referenced'
                           f' in the submenus: {self._submenus.keys()}')

    def has_submenu(self, submenu_name: str) -> bool:
        """Check if a submenu has been defined

        Parameters
        ----------
        submenu_name: str
            The submenu name as defined when calling add_submenu

        Returns
        -------
        bool: True if the submenu exists, False otherwise
        """
        return submenu_name in self._submenus

    @property
    def submenus(self) -> list[QtWidgets.QMenu]:
        """Get all submenus"""
        return list(self._submenus.values())

    @property
    def submenus_names(self) -> list[str]:
        """Get all submenu names"""
        return list(self._submenus.keys())

    @property
    def toolbar(self) -> QtWidgets.QToolBar:
        """Get the default toolbar"""
        return self._toolbar

    @property
    def menu(self) -> QtWidgets.QMenuBar:
        """Get the default menu"""
        return self._menu

    def affect_to(self, action_name, obj: Union[QtWidgets.QToolBar, QtWidgets.QMenu]):
        """Affect action to an object either a toolbar or a menu

        Parameters
        ----------
        action_name: str
            The action name as defined in setup_actions
        obj: QToolbar or QMenu
            The object where to add the action
        """
        if isinstance(obj, QtWidgets.QToolBar) or isinstance(obj, QtWidgets.QMenu):
            obj.addAction(self._actions[action_name])

    def connect_action(self, name, slot, connect=True, signal_name=''):
        """Connect (or disconnect) the action referenced by name to the given slot

        Parameters
        ----------
        name: str
            key of the action as referenced in the self._actions dict
        slot: method
            a method/function
        connect: bool
            if True connect the trigger signal of the action to the defined slot else disconnect it
        signal_name: str
            try to use it as a signal (for widgets added...) otherwise use the *triggered* signal
        """
        signal = 'triggered'
        if name in self._actions:
            if hasattr(self._actions[name], signal_name):
                signal = signal_name
            if connect:
                getattr(self._actions[name], signal).connect(slot)
            else:
                try:
                    getattr(self._actions[name], signal).disconnect()
                except (TypeError,) as e:
                    pass  # the action was not connected
        else:
            raise KeyError(f'The action with name: {name} is not referenced'
                           f' in the view actions: {self._actions.keys()}')

    @dispatch(str)
    def is_action_visible(self, action_name: str):
        """Check the visibility of a given action or the list of an action"""
        if action_name in self._actions:
            return self._actions[action_name].isVisible()
        else:
            raise KeyError(f'The action with name: {action_name} is not referenced'
                           f' in the actions list: {self._actions}')

    @dispatch(Iterable)
    def is_action_visible(self, actions_name: IterableType):
        """Check the visibility of a given action or the list of an action"""
        isvisible = False
        for action_name in actions_name:
            isvisible = isvisible and self.is_action_visible(action_name)
        return isvisible

    @dispatch(str)
    def is_action_checked(self, action_name: str):
        """Get the CheckState of a given action or a list of actions"""
        if action_name in self._actions:
            return self._actions[action_name].isChecked()
        else:
            raise KeyError(f'The action with name: {action_name} is not referenced'
                           f' in the actions list: {self._actions}')

    @dispatch(Iterable)
    def is_action_checked(self, actions_name: IterableType):
        """Get the CheckState of a given action or a list of actions"""
        ischecked = False
        for action_name in actions_name:
            ischecked = ischecked and self.is_action_checked(action_name)
        return ischecked

    @dispatch(str, bool)
    def set_action_visible(self, action_name: str, visible=True):
        """Set the visibility of a given action or a list of an action"""
        if action_name in self._actions:
            self._actions[action_name].setVisible(visible)
        else:
            raise KeyError(f'The action with name: {action_name} is not referenced'
                           f' in the actions list: {self._actions}')

    @dispatch(Iterable, bool)
    def set_action_visible(self, actions_name: IterableType, visible=True):
        """Set the visibility of a given action or a list of an action"""
        for action_name in actions_name:
            self.set_action_visible(action_name, visible)

    @dispatch(str, bool)
    def set_action_checked(self, action_name: str, checked=True):
        """Set the CheckedState of a given action or a list of actions"""
        if action_name in self._actions:
            self._actions[action_name].setChecked(checked)
        else:
            raise KeyError(f'The action with name: {action_name} is not referenced'
                           f' in the actions list: {self._actions}')

    @dispatch(Iterable, bool)
    def set_action_checked(self, actions_name: IterableType, checked=True):
        """Set the CheckedState of a given action or a list of actions"""
        for action_name in actions_name:
            self.set_action_checked(action_name, checked)

    @dispatch(str, bool)
    def set_action_enabled(self, action_name: str, enabled=True):
        """Set the EnabledState of a given action or a list of actions"""
        if action_name in self._actions:
            self._actions[action_name].setEnabled(enabled)
        else:
            raise KeyError(f'The action with name: {action_name} is not referenced'
                           f' in the actions list: {self._actions}')

    @dispatch(Iterable, bool)
    def set_action_enabled(self, actions_name: IterableType, enabled=True):
        """Set the EnabledState of a given action or a list of actions"""
        for action_name in actions_name:
            self.set_action_enabled(action_name, enabled)

    @dispatch(str)
    def is_action_enabled(self, action_name: str):
        """Get the EnabledState of a given action or a list of actions"""
        if action_name in self._actions:
            return self._actions[action_name].isEnabled()
        else:
            raise KeyError(f'The action with name: {action_name} is not referenced'
                           f' in the actions list: {self._actions}')

    @dispatch(Iterable)
    def is_action_checked(self, actions_name: IterableType):
        """Get the EnabledState of a given action or a list of actions"""
        is_enabled = False
        for action_name in actions_name:
            is_enabled = is_enabled and self.is_action_enabled(action_name)
        return is_enabled
