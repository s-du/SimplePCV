"""
This is a super simple script to create PCV renders from height maps
Based on
Duguet, Florent & Girardeau-Montaut, Daniel. (2004). Rendu en Portion de Ciel Visible de Gros Nuages de Points 3D.
"""

import os
import numpy as np
import rasterio as rio
import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import *
import simplepcv as spcv


SCALE_FACTOR = 100
IMAGE_PATH = 'dtm_large.tif'  # Replace with path to image

class UiLoader(QUiLoader):
    """
    Subclass :class:`~PySide.QtUiTools.QUiLoader` to create the user interface
    in a base instance.

    Unlike :class:`~PySide.QtUiTools.QUiLoader` itself this class does not
    create a new instance of the top-level widget, but creates the user
    interface in an existing instance of the top-level class.

    This mimics the behaviour of :func:`PyQt4.uic.loadUi`.
    """

    def __init__(self, baseinstance, customWidgets=None):
        """
        Create a loader for the given ``baseinstance``.

        The user interface is created in ``baseinstance``, which must be an
        instance of the top-level class in the user interface to load, or a
        subclass thereof.

        ``customWidgets`` is a dictionary mapping from class name to class object
        for widgets that you've promoted in the Qt Designer interface. Usually,
        this should be done by calling registerCustomWidget on the QUiLoader, but
        with PySide 1.1.2 on Ubuntu 12.04 x86_64 this causes a segfault.

        ``parent`` is the parent object of this loader.
        """

        QUiLoader.__init__(self, baseinstance)
        self.baseinstance = baseinstance
        self.customWidgets = customWidgets

    def createWidget(self, class_name, parent=None, name=''):
        """
        Function that is called for each widget defined in ui file,
        overridden here to populate baseinstance instead.
        """

        if parent is None and self.baseinstance:
            # supposed to create the top-level widget, return the base instance
            # instead
            return self.baseinstance

        else:
            if class_name in self.availableWidgets():
                # create a new widget for child widgets
                widget = QUiLoader.createWidget(self, class_name, parent, name)

            else:
                # if not in the list of availableWidgets, must be a custom widget
                # this will raise KeyError if the user has not supplied the
                # relevant class_name in the dictionary, or TypeError, if
                # customWidgets is None
                try:
                    widget = self.customWidgets[class_name](parent)

                except (TypeError, KeyError) as e:
                    raise Exception(
                        'No custom widget ' + class_name + ' found in customWidgets param of UiLoader __init__.')

            if self.baseinstance:
                # set an attribute for the new child widget on the base
                # instance, just like PyQt4.uic.loadUi does.
                setattr(self.baseinstance, name, widget)

                # this outputs the various widget names, e.g.
                # sampleGraphicsView, dockWidget, samplesTableView etc.
                # print(name)

            return widget


def loadUi(uifile, baseinstance=None, customWidgets=None,
           workingDirectory=None):
    """
    Dynamically load a user interface from the given ``uifile``.

    ``uifile`` is a string containing a file name of the UI file to load.

    If ``baseinstance`` is ``None``, the a new instance of the top-level widget
    will be created.  Otherwise, the user interface is created within the given
    ``baseinstance``.  In this case ``baseinstance`` must be an instance of the
    top-level widget class in the UI file to load, or a subclass thereof.  In
    other words, if you've created a ``QMainWindow`` interface in the designer,
    ``baseinstance`` must be a ``QMainWindow`` or a subclass thereof, too.  You
    cannot load a ``QMainWindow`` UI file with a plain
    :class:`~PySide.QtGui.QWidget` as ``baseinstance``.

    ``customWidgets`` is a dictionary mapping from class name to class object
    for widgets that you've promoted in the Qt Designer interface. Usually,
    this should be done by calling registerCustomWidget on the QUiLoader, but
    with PySide 1.1.2 on Ubuntu 12.04 x86_64 this causes a segfault.

    :method:`~PySide.QtCore.QMetaObject.connectSlotsByName()` is called on the
    created user interface, so you can implemented your slots according to its
    conventions in your widget class.

    Return ``baseinstance``, if ``baseinstance`` is not ``None``.  Otherwise
    return the newly created instance of the user interface.
    """

    loader = UiLoader(baseinstance, customWidgets)

    if workingDirectory is not None:
        loader.setWorkingDirectory(workingDirectory)

    widget = loader.load(uifile)
    QMetaObject.connectSlotsByName(widget)
    return widget


class PhotoViewer(QGraphicsView):
    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setBackgroundBrush(QBrush(QColor(255, 255, 255)))
        self.setFrameShape(QFrame.NoFrame)

        self.setMinimumSize(400, 600)


        self.rect = False
        self.select_point = False

        self.setMouseTracking(True)
        self.origin = QPoint()

        self._current_rect_item = None
        self._current_point = None

    def has_photo(self):
        return not self._empty

    def showEvent(self, event):
        self.fitInView()
        super(PhotoViewer, self).showEvent(event)

    def fitInView(self, scale=True):
        rect = QRectF(self._photo.pixmap().rect())
        print(rect)
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.has_photo():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                print('unity: ', unity)
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                print('view: ', viewrect)
                scenerect = self.transform().mapRect(rect)
                print('scene: ', viewrect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QPixmap())
        self.fitInView()

    def toggleDragMode(self):
        if self.rect or self.select_point:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            if self.dragMode() == QGraphicsView.ScrollHandDrag:
                self.setDragMode(QGraphicsView.NoDrag)
            elif not self._photo.pixmap().isNull():
                self.setDragMode(QGraphicsView.ScrollHandDrag)


    def get_selected_point(self):
        print(self._current_point)
        return self._current_point

    # mouse events
    def wheelEvent(self, event):
        print(self._zoom)
        if self.has_photo():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def mousePressEvent(self, event):
        if self.rect:
            pass

        elif self.select_point:
            pass

        super(PhotoViewer, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.rect:
            pass

        super(PhotoViewer, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.rect:
            pass

        super(PhotoViewer, self).mouseReleaseEvent(event)


def numpy_array_to_qpixmap(arr):
    height, width = arr.shape[:2]

    arr_rgb = np.ascontiguousarray(arr[..., ::-1])  # Convert BGR to RGB and ensure C-contiguity
    qimg = QImage(arr_rgb.data, width, height, arr_rgb.strides[0], QImage.Format_RGB888)

    return QPixmap.fromImage(qimg)

class ImageViewer(QMainWindow):
    def __init__(self, image):
        super().__init__()

        # load the ui
        basepath = os.path.dirname(__file__)
        basename = 'simplepcv'
        uifile = os.path.join(basepath, '%s.ui' % basename)
        print(uifile)
        loadUi(uifile, self)

        self.setWindowTitle('Create PCV Renders')

        self.image = image

        self.comboBox.addItems(['Red', 'Green', 'Blue'])

        # add viewer
        self.viewer = PhotoViewer(self)
        self.horizontalLayout_6.addWidget(self.viewer)

        vmin = np.percentile(self.image, 1)  # 1st percentile
        vmax = np.percentile(self.image, 99)

        # Create the sliders and their labels
        self.slider_min.setMinimum(int(self.image.min() * SCALE_FACTOR))
        self.slider_min.setMaximum(int(self.image.max() * SCALE_FACTOR))
        self.slider_min.setValue(int(vmin * SCALE_FACTOR))
        self.slider_min.valueChanged.connect(self.update_image)

        self.slider_max.setMinimum(int(self.image.min() * SCALE_FACTOR))
        self.slider_max.setMaximum(int(self.image.max() * SCALE_FACTOR))
        self.slider_max.setValue(int(vmax * SCALE_FACTOR))
        self.slider_max.valueChanged.connect(self.update_image)

        self.dial.valueChanged.connect(self.update_image)
        self.comboBox.currentIndexChanged.connect(self.update_image)

        # Compute gradient
        gradient_y, gradient_x = np.gradient(image)
        self.magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        self.threshold = np.percentile(self.magnitude, 85)
        self.max_gradient_zones = self.magnitude > self.threshold

        self.update_image()


    def update_image(self):
        # Get the min and max values from the sliders and rescale them
        vmin = self.slider_min.value() / SCALE_FACTOR
        vmax = self.slider_max.value() / SCALE_FACTOR
        color_factor = self.dial.value() / 100
        color_choice = self.comboBox.currentIndex()

        # Update the labels with the current slider values
        self.min_value_label.setText(f"Current Min Value: {vmin:.4f}")
        self.max_value_label.setText(f"Current Max Value: {vmax:.4f}")

        pix = spcv.export_results(self.image, vmin, vmax, color_choice,color_factor)
        self.viewer.setPhoto(numpy_array_to_qpixmap(pix))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Test path
    with rio.open(IMAGE_PATH) as src:
        elevation = src.read(1)
        # Set masked values to np.nan
        elevation[elevation < 0] = np.nan
        height_data = elevation

    visibility = spcv.compute_sky_visibility(height_data)

    app = QApplication(sys.argv)
    viewer = ImageViewer(visibility)
    viewer.show()


    sys.exit(app.exec())
