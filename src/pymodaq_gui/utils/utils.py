import os
import sys

from qtpy.QtCore import QObject, Signal, QEvent, QBuffer, QIODevice, Qt
from qtpy import QtWidgets, QtCore, QtGui

from pathlib import Path
from pymodaq_utils.config import Config
from pymodaq_utils.logger import set_logger, get_module_name
from pyqtgraph import mkQApp as mkQApppg

config = Config()
logger = set_logger(get_module_name(__file__))

def first_available_integer(liste):
    i = 0
    while i in liste:
        i += 1
    return i

def set_dark_palette(app):
    from qtpy.QtGui import QPalette, QColor
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53,53,53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(42,42,42))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Dark, QColor(35,35,35))
    palette.setColor(QPalette.ColorRole.Shadow, QColor(20,20,20))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(127,127,127))
    palette.setColor(QPalette.Disabled, QPalette.ColorRole.ButtonText, Qt.GlobalColor.darkGray)
    palette.setColor(QPalette.Disabled, QPalette.ColorRole.WindowText, Qt.GlobalColor.darkGray)
    palette.setColor(QPalette.Disabled, QPalette.ColorRole.Text, Qt.GlobalColor.darkGray)
    palette.setColor(QPalette.Disabled, QPalette.ColorRole.Light, QColor(53, 53, 53))

    app.setPalette(palette)

def clickable(widget):
    class Filter(QObject):
        clicked = Signal()

        def eventFilter(self, obj, event):
            if obj == widget:
                if event.type() == QEvent.MouseButtonRelease:
                    if obj.rect().contains(event.pos()):
                        self.clicked.emit()
                        # The developer can opt for .emit(obj) to get the object within the slot.
                        return True
            return False

    filter = Filter(widget)
    widget.installEventFilter(filter)
    return filter.clicked


def h5tree_to_QTree(base_node, base_tree_elt=None, pixmap_items=[]):
    """
        | Convert a loaded h5 file to a QTreeWidgetItem element structure containing two columns.
        | The first is the name of the h5 current node, the second is the path of the node in the h5 structure.
        |
        | Recursive function discreasing on base_node.

        ==================   ======================================== ===============================
        **Parameters**        **Type**                                 **Description**

          *h5file*            instance class File from tables module   loaded h5 file

          *base_node*         pytables h5 node                         parent node

          *base_tree_elt*     QTreeWidgetItem                          parent QTreeWidgetItem element
        ==================   ======================================== ===============================

        Returns
        -------
        QTreeWidgetItem
            h5 structure copy converted into QtreeWidgetItem structure.

        See Also
        --------
        h5tree_to_QTree

    """

    if base_tree_elt is None:
        base_tree_elt = QtWidgets.QTreeWidgetItem([base_node.name, "", base_node.path])
    for node_name, node in base_node.children().items():
        child = QtWidgets.QTreeWidgetItem([node_name, "", node.path])
        if 'pixmap' in node.attrs.attrs_name:
            pixmap_items.append(dict(node=node, item=child))
        klass = node.attrs['CLASS']
        if klass == 'GROUP':
            h5tree_to_QTree(node, child, pixmap_items)

        base_tree_elt.addChild(child)
    return base_tree_elt, pixmap_items


def set_enable_recursive(children, enable=False):
    """Apply the enable state on all children widgets, do it recursively

    Parameters
    ----------
    children: (list) elements children ofa pyqt5 element
    enable: (bool) set enabled state (True) of all children widgets
    """
    for child in children:
        if not children:
            return
        elif isinstance(child, QtWidgets.QSpinBox) or isinstance(child, QtWidgets.QComboBox) or \
                isinstance(child, QtWidgets.QPushButton) or isinstance(child, QtWidgets.QListWidget):
            child.setEnabled(enable)
        else:
            set_enable_recursive(child.children(), enable)


def widget_to_png_to_bytes(widget, keep_aspect=True, width=200, height=100):
    """
    Renders the widget content in a png format as a bytes string
    Parameters
    ----------
    widget: (QWidget) the widget to render
    keep_aspect: (bool) if True use width and the widget aspect ratio to calculate the height
                        if False use set values of width and height to produce the png
    width: (int) the rendered width of the png
    height: (int) the rendered width of the png

    Returns
    -------
    binary string

    """
    png = widget.grab().toImage()
    wwidth = widget.width()
    wheight = widget.height()
    if keep_aspect:
        height = width * wheight / wwidth

    png = png.scaled(int(width), int(height), QtCore.Qt.KeepAspectRatio)
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.WriteOnly)
    png.save(buffer, "png")
    return buffer.data().data()


def pngbinary2Qlabel(databinary, scale_height: int = None):
    buff = QBuffer()
    buff.open(QIODevice.WriteOnly)
    buff.write(databinary)
    dat = buff.data()
    pixmap = QtGui.QPixmap()
    pixmap.loadFromData(dat, 'PNG')
    if scale_height is not None and isinstance(scale_height, int):
        pixmap = pixmap.scaledToHeight(scale_height)
    label = QtWidgets.QLabel()
    label.setPixmap(pixmap)
    return label


def start_qapplication() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication(sys.argv)
    if config('style', 'darkstyle'):
        set_dark_palette(app)
    return app


def mkQApp(name: str):
    app = mkQApppg(name)
    if config('style', 'darkstyle'):
        set_dark_palette(app)
    return app


def exec():
    app = mkQApp('a name')
    return app.exec() if hasattr(app, 'exec') else app.exec_()
