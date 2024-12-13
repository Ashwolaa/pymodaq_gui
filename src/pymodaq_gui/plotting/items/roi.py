
from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np
import pyqtgraph as pg
from pymodaq_data.post_treatment.process_to_scalar import DataProcessorFactory
from pymodaq_utils.logger import get_module_name, set_logger
from pymodaq_utils.math_utils import rotate2D
from pymodaq_utils.utils import plot_colors
from pyqtgraph import ROI as pgROI
from pyqtgraph import LinearRegionItem as pgLinearROI
from pyqtgraph import functions as fn
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QSignalBlocker, Signal, Slot

from pymodaq_gui.config import get_set_roi_path
from pymodaq_gui.parameter import (Parameter, ParameterTree,
                                   )
from pymodaq_gui.plotting.utils import plot_utils
data_processors = DataProcessorFactory()

roi_path = get_set_roi_path()
logger = set_logger(get_module_name(__file__))
translate = QtCore.QCoreApplication.translate


ROI_NAME_PREFIX = 'ROI_'
def roi_format(index):
    return f'{ROI_NAME_PREFIX}{index:02d}'


class ROI(pgROI):
    index_signal = Signal(int)
    sigCopyRequested = Signal(object)
    sigDoubleClicked = Signal(object,object)

    def __init__(self, index=0, name='roi', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.index = index
        self.signalBlocker = QSignalBlocker(self)
        self.signalBlocker.unblock()
        self._clipboard = QtGui.QGuiApplication.clipboard()

    def _emitCopyRequest(self):
        self.sigCopyRequested.emit(self)

    def contextMenuEnabled(self):
        return True
    
    def raiseContextMenu(self, ev):
        menu = self.getMenu()
        menu = self.scene().addParentContextMenus(self, menu, ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))

    def getMenu(self):
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            self.menu.setTitle(translate("ROI", "ROI"))
            self.menu.addAction('Copy ROI to clipboard', self.copy_clipboard)            
            self.menu.addAction("Copy ROI",self._emitCopyRequest)
            self.menu.addAction("Remove ROI",self._emitRemoveRequest)       
        return self.menu
    
    def contextMenuEvent(self, event):
        if self.menu is not None:
            self.menu.exec(event.screenPos())

    def mouseDoubleClickEvent(self,ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            print(self.display_state())
            self.sigDoubleClicked.emit(self,ev)

    def emit_index_signal(self):
        self.index_signal.emit(self.index)

    @property
    def color(self):
        return self.pen.color()

    def center(self) -> pg.Point:
        """ Get the center position of the ROI """
        return pg.Point(self.pos() + rotate2D(point =(self.width()/2,self.height()/2), angle=np.deg2rad(self.angle())))

    def set_center(self, center: Union[pg.Point, Tuple[float, float]]):
        """ Set the center position of the ROI """
        self.setPos(center - rotate2D(point =(self.width()/2,self.height()/2), angle=np.deg2rad(self.angle())))

    def copy_clipboard(self):
        info = plot_utils.RoiInfo.info_from_rect_roi(self)
        self._clipboard.setText(str(info.to_slices()))

    def width(self) -> float:
        return self.size().x()

    def height(self) -> float:
        return self.size().y()

    def key(self) -> str:
        return roi_format(self.index)
    
    def type(self) -> str:
        return type(self).__name__    
    
    def doShow(self,status,):
        if status:
            self.show()
        else:
            self.hide()

class ROIBrushable(ROI):
    def __init__(self, brush=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if brush is None:
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 255, 50))
        self.setBrush(brush)

    def setBrush(self, *br, **kargs):
        """Set the brush that fills the region. Can have any arguments that are valid
        for :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        self.brush = fn.mkBrush(*br, **kargs)
        self.currentBrush = self.brush

    def paint(self, p, opt, widget):
        # p.save()
        # Note: don't use self.boundingRect here, because subclasses may need to redefine it.
        r = QtCore.QRectF(0, 0, self.state['size'][0], self.state['size'][1]).normalized()

        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.currentPen)
        p.setBrush(self.currentBrush)
        p.translate(r.left(), r.top())
        p.scale(r.width(), r.height())
        p.drawRect(0, 0, 1, 1)
        # p.restore()


class LinearROI(pgLinearROI):
    index_signal = Signal(int)
    sigCopyRequested = Signal(object)
    sigDoubleClicked = Signal(object,object)
    def __init__(self, index=0, pos=[0, 10], name = 'roi', **kwargs):
        super().__init__(values=pos, **kwargs)
        self.name = name
        self.index = index
        self.signalBlocker = QSignalBlocker(self)

        self._menu = QtWidgets.QMenu()
        self._menu.addAction('Copy ROI to clipboard', self.copy_clipboard)
        self._clipboard = QtGui.QGuiApplication.clipboard()

    def copy_clipboard(self):
        info = plot_utils.RoiInfo.info_from_linear_roi(self)
        self._clipboard.setText(str(info.to_slices()))

    def contextMenuEvent(self, event):
        if self._menu is not None:
            self._menu.exec(event.screenPos())

    def pos(self) -> Tuple[float, float]:
        return self.getRegion()

    def center(self) -> float:
        pos = self.pos()
        return (pos[0] + pos[1]) / 2

    def setPos(self, pos: Tuple[int, int]):
        self.setRegion(pos)

    def setPen(self, color):
        self.setBrush(color)

    @property
    def color(self):
        return self.brush.color()

    def emit_index_signal(self):
        self.index_signal.emit(self.index)

    def key(self,):
        return roi_format(self.index)
    
    def doShow(self,status,):
        if status:
            self.show()
        else:
            self.hide()
class EllipseROI(ROI):
    """
    Elliptical ROI subclass with one scale handle and one rotation handle.


    ============== =============================================================
    **Arguments**
    pos            (length-2 sequence) The position of the ROI's origin.
    size           (length-2 sequence) The size of the ROI's bounding rectangle.
    **args         All extra keyword arguments are passed to ROI()
    ============== =============================================================

    """

    def __init__(self, index=0, pos=[0, 0], size=[10, 10], **kwargs):
        # QtGui.QGraphicsRectItem.__init__(self, 0, 0, size[0], size[1])
        super().__init__(pos=pos, size=size, index=index, **kwargs)
        self.addRotateHandle([1.0, 0.5], [0.5, 0.5])
        self.addScaleHandle([0.5 * 2. ** -0.5 + 0.5, 0.5 * 2. ** -0.5 + 0.5], [0.5, 0.5])

    def getArrayRegion(self, arr, img=None, axes=(0, 1), **kwds):
        """
        Return the result of ROI.getArrayRegion() masked by the elliptical shape
        of the ROI. Regions outside the ellipse are set to 0.
        """
        # Note: we could use the same method as used by PolyLineROI, but this
        # implementation produces a nicer mask.
        if kwds["returnMappedCoords"]:
            arr, coords = pgROI.getArrayRegion(self, arr, img, axes, **kwds)
        else:
            arr = pgROI.getArrayRegion(self, arr, img, axes, **kwds)
        if arr is None or arr.shape[axes[0]] == 0 or arr.shape[axes[1]] == 0:
            return arr
        w = arr.shape[axes[0]]
        h = arr.shape[axes[1]]
        # generate an ellipsoidal mask
        mask = np.fromfunction(
            lambda x, y: (((x + 0.5) / (w / 2.) - 1) ** 2 + ((y + 0.5) / (h / 2.) - 1) ** 2) ** 0.5 < 1, (w, h))

        # reshape to match array axes
        if axes[0] > axes[1]:
            mask = mask.T
        shape = [(n if i in axes else 1) for i, n in enumerate(arr.shape)]
        mask = mask.reshape(shape)
        if kwds["returnMappedCoords"]:
            return arr * mask, coords
        else:
            return arr * mask

    def paint(self, p, opt, widget):
        r = self.boundingRect()
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.currentPen)

        p.scale(r.width(), r.height())  # workaround for GL bug
        r = QtCore.QRectF(r.x() / r.width(), r.y() / r.height(), 1, 1)

        p.drawEllipse(r)

    def shape(self):
        self.path = QtGui.QPainterPath()
        self.path.addEllipse(self.boundingRect())
        return self.path


class CircularROI(EllipseROI):
    def __init__(self, index=0, pos=[0, 0], size=[10, 10], **kwargs):
        ROI.__init__(self, pos=pos, size=size, index=index, **kwargs)
        self.addScaleHandle([0.5 * 2. ** -0.5 + 0.5, 0.5 * 2. ** -0.5 + 0.5], [0.5, 0.5],
                            lockAspect=True)


class SimpleRectROI(ROI):
    r"""
    Rectangular ROI subclass with a single scale handle at the top-right corner.
    """

    def __init__(self, pos=[0, 0], size=[10, 10], centered=False, sideScalers=False, **args):
        super().__init__(pos, size, **args)
        if centered:
            center = [0.5, 0.5]
        else:
            center = [0, 0]

        self.addScaleHandle([1, 1], center)
        if sideScalers:
            self.addScaleHandle([1, 0.5], [center[0], 0.5])
            self.addScaleHandle([0.5, 1], [0.5, center[1]])


class RectROI(ROI):
    def __init__(self, index=0, pos=[0, 0], size=[10, 10], **kwargs):
        super().__init__(pos=pos, size=size, index=index, **kwargs)  # , scaleSnap=True, translateSnap=True)
        self.addScaleHandle([1, 1], [0, 0])
        self.addRotateHandle([0, 0], [0.5, 0.5])


class ROIPositionMapper(QtWidgets.QWidget):
    """ Widget presenting a Tree structure representing a ROI positions.
    """

    def __init__(self, roi_pos, roi_size):
        super().__init__()
        self.roi_pos = roi_pos
        self.roi_size = roi_size

    def show_dialog(self):
        self.params = [
            {'name': 'position', 'type': 'group', 'children': [
                {'name': 'x0', 'type': 'float', 'value': self.roi_pos[0] + self.roi_size[0] / 2,
                 'step': 1},
                {'name': 'y0', 'type': 'float', 'value': self.roi_pos[1] + self.roi_size[1] / 2,
                 'step': 1}
            ]},
            {'name': 'size', 'type': 'group', 'children': [
                {'name': 'width', 'type': 'float', 'value': self.roi_size[0], 'step': 1},
                {'name': 'height', 'type': 'float', 'value': self.roi_size[1], 'step': 1}]
             }]

        dialog = QtWidgets.QDialog(self)
        vlayout = QtWidgets.QVBoxLayout()
        self.settings_tree = ParameterTree()
        vlayout.addWidget(self.settings_tree, 10)
        self.settings_tree.setMinimumWidth(300)
        self.settings = Parameter.create(name='settings', type='group', children=self.params)
        self.settings_tree.setParameters(self.settings, showTop=False)
        dialog.setLayout(vlayout)

        buttonBox = QtWidgets.QDialogButtonBox(parent=self)
        buttonBox.addButton('Apply', buttonBox.AcceptRole)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.addButton('Cancel', buttonBox.RejectRole)
        buttonBox.rejected.connect(dialog.reject)

        vlayout.addWidget(buttonBox)
        self.setWindowTitle('Set Precise positions for the ROI')
        res = dialog.exec()

        if res == dialog.Accepted:

            return self.settings
        else:
            return None

