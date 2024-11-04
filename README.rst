PyMoDAQ GUI
###########

.. image:: https://img.shields.io/pypi/v/pymodaq_gui.svg
   :target: https://pypi.org/project/pymodaq_gui/
   :alt: Latest Version

.. image:: https://readthedocs.org/projects/pymodaq/badge/?version=latest
   :target: https://pymodaq.readthedocs.io/en/stable/?badge=latest
   :alt: Documentation Status

.. image:: https://codecov.io/gh/PyMoDAQ/pymodaq_gui/branch/5.0.x_dev/graph/badge.svg
    :target: https://codecov.io/gh/PyMoDAQ/pymodaq_gui

====== ========== ======= ======
Python Qt Backend OS      Passed
====== ========== ======= ======
3.8    Qt5        Linux   |38Qt5|
3.9    Qt5        Linux   |39Qt5|
3.10   Qt5        Linux   |310Qt5|
3.11   Qt5        Linux   |311Qt5|
3.11   Qt5        Windows |311Qt5win|
3.11   PyQt6      Linux   |311pyqt6|
3.11   PyQt6      Windows |311pyqt6_win|
3.11   PySide6    Windows |311pyside6_win|
====== ========== ======= ======


.. |38Qt5| image:: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp38pyqt5.yml/badge.svg
    :target: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp38pyqt5.yml

.. |39Qt5| image:: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp39pyqt5.yml/badge.svg
    :target: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp39pyqt5.yml

.. |310Qt5| image:: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp310pyqt5.yml/badge.svg
    :target: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp310pyqt5.yml

.. |311Qt5| image:: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp311pyqt5.yml/badge.svg
    :target: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp311pyqt5.yml

.. |311Qt5win| image:: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp311pyqt5_win.yml/badge.svg
    :target: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp311pyqt5_win.yml

.. |311pyqt6| image:: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp311pyqt6.yml/badge.svg
    :target: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp311pyqt6.yml

.. |311pyqt6_win| image:: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp311pyqt6_win.yml/badge.svg
    :target: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp311pyqt6_win.yml

.. |311pyside6_win| image:: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp311pyside6_win.yml/badge.svg
    :target: https://github.com/PyMoDAQ/pymodaq_gui/actions/workflows/Testp311pyside6_win.yml



.. figure:: http://pymodaq.cnrs.fr/en/latest/_static/splash.png
   :alt: shortcut

PyMoDAQ__, Modular Data Acquisition with Python, is a set of **python** modules used to interface any kind of
experiments. It simplifies the interaction with detector and actuator hardware to go straight to the data acquisition
of interest.

__ https://pymodaq.readthedocs.io/en/stable/?badge=latest

`PyMoDAQ GUI` is a set of utilities (constants, methods and classes) and graphical components
based on the Qt framework that are used to create high level user interfaces.

It contains both simple components to create GUI but also high level ones, for instance to display data objects
such as the one generated using the `pymodaq_data` package.


For instance, you'll find data viewers available out of the box:

.. figure:: https://pymodaq.cnrs.fr/en/latest/_images/data_femto_fs.png

    Specific GUI component to view and manipulate 2D data.



Published under the MIT FREE SOFTWARE LICENSE

GitHub repo: https://github.com/PyMoDAQ

Documentation: http://pymodaq.cnrs.fr/
