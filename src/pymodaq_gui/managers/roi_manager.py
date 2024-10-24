
import os
import sys
from collections import OrderedDict
from pathlib import Path
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
from pyqtgraph.parametertree.parameterTypes.basetypes import GroupParameter
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QObject, QSignalBlocker, Signal, Slot
from qtpy.QtGui import QIcon, QPixmap

from pymodaq_gui.config import get_set_roi_path
from pymodaq_gui.managers.action_manager import QAction
from pymodaq_gui.parameter import (Parameter, ParameterTree, ioxml,
                                   pymodaq_ptypes)
from pymodaq_gui.parameter import utils as putils
from pymodaq_gui.plotting.utils import plot_utils
from pymodaq_gui.utils import select_file

data_processors = DataProcessorFactory()

roi_path = get_set_roi_path()
logger = set_logger(get_module_name(__file__))


ROI_NAME_PREFIX = 'ROI_'
def roi_format(index):
    return f'{ROI_NAME_PREFIX}{index:02d}'

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


class ROI(pgROI):
    index_signal = Signal(int)

    def __init__(self, *args, index=0, name='roi', **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.index = index
        self.signalBlocker = QSignalBlocker(self)
        self.signalBlocker.unblock()
        self._menu = QtWidgets.QMenu()
        self._menu.addAction('Set ROI positions', self.set_positions)
        self._menu.addAction('Copy ROI to clipboard', self.copy_clipboard)
        # self.sigRegionChangeFinished.connect(self.emit_index_signal)
        self._clipboard = QtGui.QGuiApplication.clipboard()

    def emit_index_signal(self):
        self.index_signal.emit(self.index)


    @property
    def color(self):
        return self.pen.color()

    def center(self) -> pg.Point:
        """ Get the center position of the ROI """
        return pg.Point(self.pos() + rotate2D(point =(self.width()/2,self.height()/2), angle=np.deg2rad(self.angle())))

    def set_center(self, center: Union[pg.Point, Tuple[float, float]]):
        self.setPos(center - rotate2D(point =(self.width()/2,self.height()/2), angle=np.deg2rad(self.angle())))

    def set_positions(self):
        mapper = ROIPositionMapper(self.pos(), self.size())
        settings = mapper.show_dialog()
        if settings is not None:
            self.setSize((settings['size', 'width'], settings['size', 'height']))
            self.setPos((settings['position', 'x0'] - settings['size', 'width'] / 2,
                         settings['position', 'y0'] - settings['size', 'height'] / 2))

    def copy_clipboard(self):
        info = plot_utils.RoiInfo.info_from_rect_roi(self)
        self._clipboard.setText(str(info.to_slices()))

    def contextMenuEvent(self, event):
        if self._menu is not None:
            self._menu.exec(event.screenPos())

    def width(self) -> float:
        return self.size().x()

    def height(self) -> float:
        return self.size().y()

    def key(self,):
        return roi_format(self.index)
    
    def type(self)-> str:
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

    def __init__(self, index=0, pos=[0, 10], name = 'roi', **kwargs):
        super().__init__(values=pos, **kwargs)
        self.name = name
        self.index = index
        self.signalBlocker = QSignalBlocker(self)

        self._menu = QtWidgets.QMenu()
        self._menu.addAction('Copy ROI to clipboard', self.copy_clipboard)
        # self.sigRegionChangeFinished.connect(self.emit_index_signal)
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


ROI2D_TYPES = ['RectROI', 'EllipseROI', 'CircularROI']

class ROIScalableGroup(GroupParameter):
    def __init__(self, roi_type='1D', **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add"
        self.roi_type = roi_type
        if roi_type != '1D':
            opts['addList'] = ROI2D_TYPES        
        super().__init__(**opts)

    def addNew(self, typ=''):
        name_prefix = ROI_NAME_PREFIX
        child_indexes = [int(par.name()[len(name_prefix) + 1:]) for par in self.children()]
        if not child_indexes:
            newindex = 0
        else:
            newindex = max(child_indexes) + 1

        self.addChild(self.makeChild(newindex,typ))


    def makeChild(self,index,roi_type):
        child = {'name': roi_format(index), 'type': 'bool','value':True, 'removable': True, 'renamable': False, 'expanded': False,'context':['Copy',]}
        if self.roi_type =='2D':
            child['children'] = ROIScalableGroup.makeROIParam2D(roi_type,index)
        elif self.roi_type =='1D':
            child['children'] = ROIScalableGroup.makeROIParam1D(roi_type,index)
        return child  

    def makeChannelsParam(dim='2D'):
        if dim =='2D':
            child = [{'title': 'Use channel', 'name': 'use_channel', 'type': 'itemselect', 'checkbox': True,
                      'value': dict(all_items=['red', 'green', 'blue'],
                           selected=['red',]),
                              },]
        else:
            child = [{'title': 'Use channel', 'name': 'use_channel', 'type': 'itemselect'},]
        return child 
    @staticmethod
    def makeDisplayParam(index):
        return [{'name': 'Color', 'type': 'color', 'value': list(np.roll(ROIManager.color_list, index)[0])},
             {'name': 'zlevel', 'title':'Z-level','type': 'int', 'expanded': False, 'value':10},] 
        
    @staticmethod
    def makeMathParam(dim='2D'):
        return [{'title': 'Math type:', 'name': 'math_function', 'type': 'list',
                             'limits': data_processors.functions_filtered(f'Data{dim}')},]
    @staticmethod    
    def makeROIParam2D(roi_type,index):
            children = []    
            children.extend([{'title': 'Type', 'name': 'roi_type', 'type': 'list', 'value': roi_type, 'limits':['RectROI','EllipseROI','CircularROI'], 'readonly': False,}])
            children.extend(ROIScalableGroup.makeChannelsParam('2D'))
            children.extend(ROIScalableGroup.makeMathParam('2D'))
            children.extend(ROIScalableGroup.makeDisplayParam(index))
            children.extend([{'name': 'center', 'type': 'group', 'expanded': False, 'children': [
                    {'name': 'x', 'type': 'float', 'value': 0, 'step': 1,'decimals':6},
                    {'name': 'y', 'type': 'float', 'value': 0, 'step': 1,'decimals':6}
                ]}, ])                
            children.extend([{'name': 'position', 'type': 'group', 'expanded': False, 'children': [
                    {'name': 'x', 'type': 'float', 'value': 0, 'step': 1,'decimals':6},
                    {'name': 'y', 'type': 'float', 'value': 0, 'step': 1,'decimals':6}
                ]}, ])          
            children.extend([
                    {'name': 'size', 'type': 'group', 'expanded': False, 'children': [
                        {'name': 'width', 'type': 'float', 'value': 10, 'step': 1,'decimals':6},
                        {'name': 'height', 'type': 'float', 'value': 10, 'step': 1,'decimals':6}
                    ]},
                    {'name': 'angle', 'type': 'float', 'value': 0, 'step': 1}])    
            return children

    @staticmethod    
    def makeROIParam1D(roi_type,index):
            children = []    
            children.extend(ROIScalableGroup.makeChannelsParam('1D'))
            children.extend(ROIScalableGroup.makeMathParam('1D'))
            children.extend(ROIScalableGroup.makeDisplayParam(index))
            children.extend([{'name': 'position', 'type': 'group', 'children': [
                {'name': 'left', 'type': 'float', 'value': 0, 'step': 1},
                {'name': 'right', 'type': 'float', 'value': 10, 'step': 1}
                    ]}, ])
            
            return children
class ROIManager(QObject):

    new_ROI_signal = Signal(int, str, str)
    remove_ROI_signal = Signal(str)
    roi_value_changed = Signal(str, tuple)
    color_signal = Signal(list)
    roi_update_children = Signal(list)
    roi_changed = Signal()
    color_list = np.array(plot_colors)

    def __init__(self, viewer_widget=None, ROI_type='1D'):
        super().__init__()
        self.ROI_type = ROI_type
        self.roiwidget = QtWidgets.QWidget()
        self.viewer_widget = viewer_widget  # either a PlotWidget or a ImageWidget
        self._ROIs: OrderedDict[str, ROI] = OrderedDict([])
        self.setupUI()

    @property
    def ROIs(self):
        return self._ROIs

    def __len__(self):
        return len(self._ROIs)

    def get_roi_from_index(self, index: int) -> ROI:
        return self.ROIs[roi_format(index)]

    def _set_roi_from_index(self, index: int, roi):
        self.ROIs[roi_format(index)] = roi

    def get_roi(self, roi_key):
        if roi_key in self.ROIs:
            return self.ROIs[roi_key]
        else:
            raise KeyError(f'{roi_key} is not a valid ROI identifier for {self.ROIs}')

    def emit_colors(self):
        self.color_signal.emit([self._ROIs[roi_key].color for roi_key in self._ROIs])

    def add_roi_programmatically(self, roitype=ROI2D_TYPES[0]):
        self.settings.child('ROIs').addNew(roitype)

    def remove_roi_programmatically(self, index: int):
        self.settings.child('ROIs').removeChild(self.settings.child('ROIs', roi_format(index)))

    def setupUI(self):

        vlayout = QtWidgets.QVBoxLayout()
        self.roiwidget.setLayout(vlayout)

        self.toolbar = QtWidgets.QToolBar()
        vlayout.addWidget(self.toolbar)

        self.save_ROI_pb = QAction(QIcon(QPixmap(":/icons/Icon_Library/save_ROI.png")), 'Save ROIs')
        self.load_ROI_pb = QAction(QIcon(QPixmap(":/icons/Icon_Library/load_ROI.png")), 'Load ROIs')
        self.clear_ROI_pb = QAction(QIcon(QPixmap(":/icons/Icon_Library/clear_ROI.png")), 'Clear ROIs')
        self.toolbar.addActions([self.save_ROI_pb, self.load_ROI_pb, self.clear_ROI_pb])


        self.roitree = ParameterTree()
        vlayout.addWidget(self.roitree)
        self.roiwidget.setMinimumWidth(250)
        self.roiwidget.setMaximumWidth(300)

        params = [
            {'title': 'Measurements:', 'name': 'measurements', 'type': 'table', 'value': OrderedDict([]), 'Ncol': 2,
             'header': ["LO", "Value"]},
            ROIScalableGroup(roi_type=self.ROI_type, name="ROIs")]
        self.settings = Parameter.create(title='ROIs Settings', name='rois_settings', type='group', children=params)
        self.roitree.setParameters(self.settings, showTop=False)
        self.settings.sigTreeStateChanged.connect(self.roi_tree_changed)
        self.settings_signalBlocker = QSignalBlocker(self.settings)
        self.settings_signalBlocker.unblock()
        self.save_ROI_pb.triggered.connect(self.save_ROI)
        self.load_ROI_pb.triggered.connect(lambda: self.load_ROI(None))
        self.clear_ROI_pb.triggered.connect(self.clear_ROI)

    def getIndexes(self,):
        return [roi.index for roi in self.ROIs.values()]

    def roi_tree_changed(self, param, changes):

        for param, change, data in changes:
            path = self.settings.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()
            if change == 'childAdded':  # new roi to create
                par: Parameter = data[0]
                newindex = int(par.name()[-2:])
                roi_type = ''
                pos = self.viewer_widget.plotItem.vb.viewRange()
                if self.ROI_type == '1D':
                    roi_type = ''
                    pos = pos[0]
                    pos = pos[0] + np.diff(pos)*np.array([2,4])/6
                    newroi = LinearROI(index=newindex, pos=pos)

                    newroi.setZValue(-10)
                    newroi.setBrush(par.child('Color').value())
                    newroi.setOpacity(0.2)

                elif self.ROI_type == '2D':
                    roi_type = par.child('roi_type').value()
                    xrange,yrange=pos                    
                    width = np.max(((xrange[1] - xrange[0]) / 10, 2))
                    height = np.max(((yrange[1] - yrange[0]) / 10, 2))
                    pos = [int(np.mean(xrange) - width / 2), int(np.mean(yrange) - width / 2)]

                    roi = self.makeROI2D(roi_type,index=newindex, pos=pos,size=[width, height],pen=par['Color'])


                roi.sigRegionChangeFinished.connect(lambda: self.roi_changed.emit())
                roi.sigRegionChangeFinished.connect(self.update_roi_tree)

                self.new_ROI_signal.emit(newindex, roi_type, par.name())
                self.emit_colors()
                self.roi_changed.emit()

            elif change == 'value':
                if param.name() in putils.iter_children(self.settings.child('ROIs'), []):
                    parent_name = putils.get_param_path(param)[putils.get_param_path(param).index('ROIs')+1]
                    if parent_name in self._ROIs.keys():
                        roi_changed = self._ROIs[parent_name]                                    
                        self.update_roi(roi_changed, param)
                    self.roi_value_changed.emit(parent_name, (param, param.value()))
                if param.name() == 'Color':
                    self.emit_colors()

            elif change == 'parent':
                if 'ROI' in param.name():
                    self.removeROI(self.ROIs[param.name()])

    def makeROI2D(self,roi_type,index,pos,size,**kwargs):
        if roi_type == 'RectROI':
            roi = RectROI(index=index, pos=pos,
                                size=size, name=roi_format(index),**kwargs)
        elif roi_type == 'EllipseROI':
            roi = EllipseROI(index=index, pos=pos,
                                size=size, name=roi_format(index),**kwargs)
        elif roi_type == 'CircularROI':
            roi = CircularROI(index=index, pos=pos,
                                    size=size, name=roi_format(index),**kwargs)

        self.ROIs[roi.key()]=roi
        self.viewer_widget.plotItem.addItem(roi)
        self.update_roi_tree(roi)
        return roi
    

    def removeROI(self,roi):
        roi_group = self.settings.child('ROIs')
        for param in roi_group.children():                
                if roi.key() == param.name():
                    self.settings_signalBlocker.reblock()
                    roi_group.removeChild(param)
                    self.settings_signalBlocker.unblock()
        roi = self.ROIs.pop(roi.key())
        self.viewer_widget.plotItem.removeItem(roi)
        self.remove_ROI_signal.emit(roi.key())
        self.emit_colors()

    def update_use_channel(self, channels: List[str]):
        for ind in range(len(self)):
            param = self.settings.child('ROIs', roi_format(ind), 'use_channel')
            sel = param.value()['selected'] #Get selection
            sel = [s for s in sel if s in channels] #Remove selection if channel no longer exists
            param.setValue(dict(all_items=channels,
                           selected=sel))
    def update_roi(self, roi:ROI, param):

        roi.signalBlocker.reblock()
        parent_name = param.parent().opts['name']

        if param.name() == roi.key():
            roi.doShow(param.value())
        if param.name() == 'roi_type':
            state = roi.saveState()
            self.viewer_widget.plotItem.removeItem(roi)            
            if self.ROI_type =='2D':
                roi = self.makeROI2D(roi_type=param.value(),index=roi.index,pos=state['pos'],size=state['size'],angle=state['angle'],pen=roi.pen)                
        elif param.name() == 'Color':
            roi.setPen(param.value())
            self.emit_colors()
        elif parent_name == 'center':
            center = roi.center()
            pos = self.update_roi_pos(center,param)
            if self.ROI_type =='1D':
                pos.sort()
            else:
                roi.set_center(pos)
        elif parent_name == 'position':
            position = roi.pos()
            pos = self.update_roi_pos(position,param)
            if self.ROI_type =='1D':
                pos.sort()
            roi.setPos(pos)          
        elif param.name() == 'angle':
            roi.setAngle(param.value(),center=[0.5,0.5])
        elif param.name() == 'zlevel':
            roi.setZValue(param.value())
        elif param.name() == 'width':
            size = roi.size()
            roi.setSize((param.value(), size[1]))
        elif param.name() == 'height':
            size = roi.size()
            roi.setSize((size[0], param.value()))
        roi.signalBlocker.unblock()

    def update_roi_pos(self,pos,param):
        if param.name() == 'x' or param.name() == 'left':
            poss = pg.Point(param.value(), pos.y())
        elif param.name() == 'y' or param.name() == 'right':         
            poss = pg.Point(pos.x(), param.value())                   
        return poss
    @Slot()
    def update_roi_tree(self, roi):

        par = self.settings.child(*('ROIs', roi.key()))
        if isinstance(roi, LinearROI):
            pos = roi.getRegion()
        else:
            pos = roi.pos()
            size = roi.size()
            angle = roi.angle()
            center = roi.center()
            Zvalue = roi.zValue()

        self.settings_signalBlocker.reblock()
        if isinstance(roi, LinearROI):
            par.child(*('position', 'left')).setValue(pos[0])
            par.child(*('position', 'right')).setValue(pos[1])
        if not isinstance(roi, LinearROI):
            par.child(*('position', 'x')).setValue(pos.x())
            par.child(*('position', 'y')).setValue(pos.y())
            par.child(*('center', 'x')).setValue(center.x())
            par.child(*('center', 'y')).setValue(center.y())        
            par.child(*('size', 'width')).setValue(size.x())
            par.child(*('size', 'height')).setValue(size.y())
            par.child('angle').setValue(angle)
            par.child('zlevel').setValue(Zvalue)
        self.settings_signalBlocker.unblock()

    def save_ROI(self):

        try:
            data = ioxml.parameter_to_xml_string(self.settings.child(('ROIs')))
            path = select_file(start_path=Path.home(), ext='xml', save=True, force_save_extension=True)

            if path != '':
                with open(path, 'wb') as f:
                    f.write(data)
        except Exception as e:
            print(e)

    def clear_ROI(self):
        indexes = [roi.index for roi in self._ROIs.values()]
        for index in indexes:
            self.settings.child(*('ROIs', roi_format(index))).remove()
            # self.settings.sigTreeStateChanged.connect(self.roi_tree_changed)

    def load_ROI(self, path=None, params=None):
        try:
            if params is None:
                if path is None:
                    path = select_file(start_path=Path.home(), save=False, ext='xml', filter='XML files (*.xml)')
                    if path != '':
                        params = Parameter.create(title='Settings', name='settings', type='group',
                                                  children=ioxml.XML_file_to_parameter(path))

            if params is not None:
                self.clear_ROI()
                QtWidgets.QApplication.processEvents()

                for param in params:
                    if 'roi_type' in putils.iter_children(param, []):
                        self.settings.child('ROIs').addNew(param.child('roi_type').value())
                    else:
                        self.settings.child('ROIs').addNew()
                QtWidgets.QApplication.processEvents()
                self.set_roi(self.settings.child('ROIs').children(), params)
        except Exception as e:
            logger.exception(str(e))

    def set_roi(self, roi_params, roi_params_new):
        for child, new_child in zip(roi_params, roi_params_new):
            if 'group' not in child.opts['type']:
                child.setValue(new_child.value())
            else:
                self.set_roi(child.children(), new_child.children())


class ROISaver:
    def __init__(self, msgbox=False, det_modules=[]):

        self.roi_presets = None
        self.detector_modules = det_modules

        if msgbox:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("ROI Manager?")
            msgBox.setInformativeText("What do you want to do?")
            cancel_button = msgBox.addButton(QtWidgets.QMessageBox.Cancel)
            modify_button = msgBox.addButton('Modify', QtWidgets.QMessageBox.AcceptRole)
            msgBox.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = msgBox.exec()

            if msgBox.clickedButton() == modify_button:
                path = select_file(start_path=roi_path, save=False, ext='xml')
                if path != '':
                    self.set_file_roi(str(path))
            else:  # cancel
                pass

    def set_file_roi(self, filename, show=True):
        """

        """

        children = ioxml.XML_file_to_parameter(filename)
        self.roi_presets = Parameter.create(title='roi', name='rois', type='group', children=children)

        det_children = [child for child in self.roi_presets.children() if 'det' in child.opts['name']]
        det_names = [child.child('detname').value() for child in self.roi_presets.children() if
                     'det' in child.opts['name']]
        det_module_names = [det.title for det in self.detector_modules]
        for ind_det, det_roi in enumerate(det_children):
            det_module = self.detector_modules[det_module_names.index(det_names[ind_det])]
            viewer_children = [child for child in det_roi.children() if 'viewer' in child.opts['name']]
            for ind_viewer, viewer in enumerate(det_module.viewers):
                rois_params = [child for child in viewer_children[ind_viewer].children() if 'ROI' in child.opts['name']]
                if len(rois_params) > 0:
                    if hasattr(viewer, 'roi_manager'):
                        if hasattr(viewer, 'activate_roi'):  # because for viewer 0D it is irrelevant
                            viewer.activate_roi()
                        viewer.roi_manager.load_ROI(params=rois_params)
                        QtWidgets.QApplication.processEvents()

        if show:
            self.show_rois()

    def set_new_roi(self, file=None):
        if file is None:
            file = 'roi_default'

        self.roi_presets = Parameter.create(name='roi_settings', type='group', children=[
            {'title': 'Filename:', 'name': 'filename', 'type': 'str', 'value': file}, ])

        for ind_det, det in enumerate(self.detector_modules):
            det_param = Parameter.create(name=f'det_{ind_det:03d}', type='group', children=[
                {'title': 'Det Name:', 'name': 'detname', 'type': 'str', 'value': det.title}, ])

            for ind_viewer, viewer in enumerate(det.ui.viewers):
                viewer_param = Parameter.create(
                    name=f'viewer_{ind_viewer:03d}', type='group',
                    children=[
                        {'title': 'Viewer:', 'name': 'viewername', 'type': 'str',
                         'value': det.ui.viewer_docks[ind_viewer].name()}, ])

                if hasattr(viewer, 'roi_manager'):
                    viewer_param.addChild(
                        {'title': 'ROI type:', 'name': 'roi_type', 'type': 'str',
                         'value': viewer.roi_manager.settings.child('ROIs').roi_type})
                    viewer_param.addChildren(viewer.roi_manager.settings.child('ROIs').children())
                det_param.addChild(viewer_param)
            self.roi_presets.addChild(det_param)

        ioxml.parameter_to_xml_file(self.roi_presets, os.path.join(roi_path, file))
        self.show_rois()

    def show_rois(self):
        """

        """
        dialog = QtWidgets.QDialog()
        vlayout = QtWidgets.QVBoxLayout()
        tree = ParameterTree()
        tree.setMinimumWidth(400)
        tree.setMinimumHeight(500)
        tree.setParameters(self.roi_presets, showTop=False)

        vlayout.addWidget(tree)
        dialog.setLayout(vlayout)
        buttonBox = QtWidgets.QDialogButtonBox(parent=dialog)

        buttonBox.addButton('Save', buttonBox.AcceptRole)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.addButton('Cancel', buttonBox.RejectRole)
        buttonBox.rejected.connect(dialog.reject)

        vlayout.addWidget(buttonBox)
        dialog.setWindowTitle('Fill in information about this manager')
        res = dialog.exec()

        if res == dialog.Accepted:
            # save managers parameters in a xml file
            # start = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
            # start = os.path.join("..",'daq_scan')
            ioxml.parameter_to_xml_file(
                self.roi_presets, os.path.join(
                    roi_path, self.roi_presets.child('filename').value()))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    from pyqtgraph import PlotWidget

    from pymodaq_gui.plotting.widgets import ImageWidget

    # im = ImageWidget()
    im = PlotWidget()
    prog = ROIManager(im, '2D')
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout()
    widget.setLayout(layout)
    layout.addWidget(im)
    layout.addWidget(prog.roiwidget)
    widget.show()
    prog.add_roi_programmatically(ROI2D_TYPES[0])
    prog.add_roi_programmatically(ROI2D_TYPES[1])
    prog.add_roi_programmatically(ROI2D_TYPES[2])
    sys.exit(app.exec_())