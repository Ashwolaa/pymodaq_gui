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
