#!/usr/bin/python3
#
#            ECG Viewer
#   Written by Kevin Williams - 2022-2023
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import sys
import time
import serial
import numpy
import logging
import platform
import pyqtgraph as pg
from webbrowser import Error as wb_error
from webbrowser import open as wb_open
from PyQt5 import QtWidgets, QtCore, QtGui

# manual includes to fix occasional compile problem
if platform.platform in ["win32", "darwin"]:
    try:
        from pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt5 import *
        from pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyqt5 import *
        from pyqtgraph.imageview.ImageViewTemplate_pyqt5 import *
        from pyqtgraph.console.template_pyqt5 import *
    except:
        pass

# import locals
from debug import debug_timer
from ecg_viewer_window import Ui_MainWindow
from about import Ui_about_window
from license import Ui_license_window
import images_qr        # required for icon to work properly
import log_system

# ML Integration
try:
    from ml_classifier_ui import MLClassifierUI
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML Classifier UI not found. ML features will be disabled.")

# Simulator Integration
try:
    from ecg_simulator import ECGGenerator  # Assuming generator is in ecg_simulator.py
    SIMULATOR_AVAILABLE = True
except ImportError:
    SIMULATOR_AVAILABLE = False
    print("ECG Generator not found. Simulator features will be disabled.")

# String used in the title-bar and about window
VERSION = "v2.2.2"
LOG_LEVEL = logging.DEBUG

# About window. The class is so tiny it might as well be defined here.
class AboutWindow(QtWidgets.QDialog, Ui_about_window):
    """
    About dialog box window.
    """

    def __init__(self, *args, **kwargs):
        super(AboutWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.version.setText(VERSION)
        self.icon.setPixmap(QtGui.QPixmap(":/icon/icon.png"))
        self.setWindowIcon(QtGui.QIcon(':/icon/icon.png'))


# Same for license window
class LicenseWindow(QtWidgets.QDialog, Ui_license_window):
    """
    License dialog box window.
    """

    def __init__(self, *args, **kwargs):
        super(LicenseWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(':/icon/icon.png'))


class ECGViewer(QtWidgets.QMainWindow, Ui_MainWindow):
    """
    Main class for the ECG Viewer application.
    """

    # import class methods
    from _ecg_serial_handler import ser_com_connect as original_ser_com_connect, \
                                ser_com_refresh as original_ser_com_refresh, \
                                ser_get_input, ser_start_capture_timer, \
                                ser_stop_capture_timer, ser_check_device, ser_do_calibrate
    from _ecg_grapher import graph_draw, graph_fit, graph_bold_toggle, graph_restart_timer, \
        graph_stop_timer, graph_start_timer
    from _ecg_math import math_detect_peaks, math_calc_hr, math_calc_sps
    from _ecg_ui_handler import ui_alarm_on, ui_alarm_off, ui_set_message, ui_clear_message, ui_force_invert, \
        ui_run_toggle, ui_export_data_raw, ui_export_data_png, ui_export_data_csv, ui_show_about, \
        ui_display_error_message, ui_set_tooltips, ui_statusbar_message, ui_holdoff_box_update, ui_show_license

    def __init__(self, *args, **kwargs):
        super(ECGViewer, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.about_window = AboutWindow()
        self.license_window = LicenseWindow()
        self.graph.disableAutoRange()
        self.setWindowTitle("My Custom ECG Viewer - " + VERSION)
        self.setWindowIcon(QtGui.QIcon(':/icon/icon.png'))

        # ML Classifier UI Integration
        if ML_AVAILABLE:
            self.ml_classifier_ui = MLClassifierUI(self)
            # The MLClassifierUI's __init__ -> setup_ui method should handle adding its own widgets
            # to the parent layout. We don't need to add a specific panel here.
        else:
            self.ml_classifier_ui = None

        # Capture timer
        self.capture_timer = QtCore.QTimer()
        self.capture_timer.timeout.connect(self.do_update)
        self.capture_rate_ms = 0
        self.capture_timer_qt = QtCore.QElapsedTimer()
        self.capture_index = 0

        # graph timer
        self.graph_timer = QtCore.QTimer()
        self.graph_timer.timeout.connect(self.graph_draw)
        self.graph_frame_rate = 30
        self.graph_timer_ms = int(1 / (self.graph_frame_rate / 1000))

        # set FPS menu option metadata
        self.action30_FPS.setData(30)
        self.action15_FPS.setData(15)
        self.action8_FPS.setData(8)

        # set Window Size menu option metadata
        self.action2000.setData(2000)
        self.action5000.setData(5000)
        self.action8000.setData(8000)
        self.action10000.setData(10000)

        # --- Simulator Setup --- NEW CODE
        self.ecg_generator = None
        if SIMULATOR_AVAILABLE:
            self.ecg_generator = ECGGenerator()
            # Populate ECG Type ComboBox
            self.ecgTypeCombo.addItems(self.ecg_generator.get_available_types())
            # Connect simulator UI signals
            self.heartRateSlider.valueChanged.connect(self.update_generator_hr)
            self.noiseSlider.valueChanged.connect(self.update_generator_noise)
            self.ecgTypeCombo.currentIndexChanged.connect(self.update_generator_type)
            self.dataSourceGroup.buttonClicked.connect(self.switch_data_source) # Connect ButtonGroup
            # Set initial generator state from UI defaults
            self.update_generator_hr(self.heartRateSlider.value())
            self.update_generator_noise(self.noiseSlider.value())
            self.update_generator_type()
            # Initially disable simulator controls if serial is default
            self.set_simulator_controls_enabled(False)
        else:
            # Disable simulator controls if generator not available
            self.simulatorControlsGroup.setEnabled(False)
            self.useGeneratorRadio.setEnabled(False)
            self.useSerialRadio.setChecked(True) # Force serial if no generator

        # Connect buttons to methods
        self.button_refresh.clicked.connect(self.ser_com_refresh)
        self.button_connect.clicked.connect(self.connect_toggle)
        self.button_reset.clicked.connect(self.reset)
        self.button_run.clicked.connect(self.ui_run_toggle)
        self.button_ui_force_invert.clicked.connect(self.ui_force_invert)
        self.graph_zoom_slider.sliderReleased.connect(self.graph_fit)
        self.show_track.stateChanged.connect(self.reset)
        self.FPSGroup.triggered.connect(self.graph_restart_timer)
        self.button_run.setDisabled(True)
        self.actionBold_Line.toggled.connect(self.graph_bold_toggle)
        self.actionRAW.triggered.connect(self.ui_export_data_raw)
        self.actionPNG.triggered.connect(self.ui_export_data_png)
        self.actionCSV.triggered.connect(self.ui_export_data_csv)
        self.actionAbout.triggered.connect(self.ui_show_about)
        self.actionLicense.triggered.connect(self.ui_show_license)
        self.actionGet_Source_Code.triggered.connect(self.open_source_code_webpage)
        self.actionQuit.triggered.connect(sys.exit)
        self.WindowSizeGroup.triggered.connect(self.window_size_update)
        self.actionStart_Stop.triggered.connect(self.ui_run_toggle)
        self.actionStart_Stop.setDisabled(True)
        self.actionReset.triggered.connect(self.reset)
        self.actionAuto_Holdoff.triggered.connect(self.ui_holdoff_box_update)

        # set tooltips
        self.ui_set_tooltips()

        # Serial Variables
        self.ser: serial.Serial = serial.Serial(baudrate = 115200, timeout = 1, write_timeout = 1)

        # data variables
        self.current_reading = 0
        self.value_history_max = 8000
        self.value_history = numpy.zeros(self.value_history_max)
        self.time_history  = numpy.zeros(self.value_history_max)
        self.invert_modifier = 1
        self.calibrating = self.value_history_max
        self.peaks = list()
        self.holdoff_factor = 0.31

        # graph properties
        self.graph.showGrid(True, True, alpha = 0.5)
        self.graph_padding_factor = 0.667
        self.green_pen = pg.mkPen('g', width = 2)
        self.red_pen = pg.mkPen('r', width = 2)
        self.yellow_pen = pg.mkPen('y')

        # ecg rate average
        self.rate_history = [80] * 3

        # ecg rate alarm limits
        self.rate_alarm_max = 120
        self.rate_alarm_min = 40
        self.rate_alarm_active = False

        # perform initial reset
        self.reset()
        self.ser_com_refresh()

    def open_source_code_webpage(self):
        """
        Opens a link to the project source code.
        """
        try:
            wb_open("https://github.com/HTM-Workshop/ECG-Viewer", autoraise = True)
        except wb_error as error:
            error_msg = "Could not open URL.\n\n" + error
            logging.warning(error_msg)
            self.ui_display_error_message("Open URL Error", error_msg)


    def do_update(self) -> None:
        """Modified update loop to handle both serial and generator data."""
        
        samples_per_update = 10 # Number of samples to generate/process per update cycle
        new_values = None
        avg_rate = 0
        inst_rate = 0
        classification_status = "Unknown" # Initialize status

        if self.useGeneratorRadio.isChecked() and self.ecg_generator:
            # --- Use ECG Generator --- 
            new_values = self.ecg_generator.generate(samples_per_update)
            if new_values is not None and len(new_values) > 0:
                # Create corresponding time values (simple increment based on index/fs)
                start_index = self.capture_index
                end_index = start_index + len(new_values)
                # Assuming time_history stores time in ms
                new_times = (numpy.arange(start_index, end_index) / self.ecg_generator.fs) * 1000 

                # Add generated data to history buffers
                self.value_history = numpy.roll(self.value_history, -len(new_values))
                self.time_history = numpy.roll(self.time_history, -len(new_values))
                self.value_history[-len(new_values):] = new_values
                self.time_history[-len(new_values):] = new_times

                self.capture_index += len(new_values)

                # Peak detection and HR calculation (using existing math methods)
                self.math_detect_peaks()
                # --- MODIFIED CALL: Unpack status --- 
                inst_rate, avg_rate, classification_status = self.math_calc_hr()
                self.math_calc_sps(samples_generated=len(new_values))
                
                # Auto-invert logic (if needed for generator)
                # self.auto_invert_check(new_values) 
            else:
                inst_rate, avg_rate = 0, 0 # No new values, HR is 0
                classification_status = "Cannot Classify"

            # Update HR Display
            self.update_hr_display(inst_rate, avg_rate)

        elif self.useSerialRadio.isChecked() and self.ser.is_open:
             # --- Use Serial Input --- (Keep original peak detection logic here)
             reading_ok = self.ser_get_input() 
             if reading_ok:
                 if self.calibrating > -1:
                     self.ser_do_calibrate()
                 
                 # Calculate HR from detected peaks for serial data
                 self.math_detect_peaks()
                 # --- MODIFIED CALL: Unpack status --- 
                 inst_rate, avg_rate, classification_status = self.math_calc_hr()
                 self.math_calc_sps()
             else:
                 inst_rate, avg_rate = 0, 0
                 classification_status = "Cannot Classify - Signal Loss"
                 self.ui_set_message("SIGNAL LOSS")
             
             # Update HR Display
             self.update_hr_display(inst_rate, avg_rate)

        else:
            # No data source active or available
            self.update_hr_display(0, 0) # Display 0 HR
            classification_status = "No Data Source"
            pass
        
        # Store the latest status
        self.current_classification_status = classification_status

        # Update ML classifier UI if available 
        # (The UI object itself checks if ML is enabled for display purposes)
        if hasattr(self, 'ml_classifier_ui') and self.ml_classifier_ui:
            # Trigger classification UI update periodically (e.g., every second)
            if not hasattr(self, 'last_ml_update_time') or time.time() - self.last_ml_update_time > 1.0:
                 # Call classification UI update via the ML UI object
                 if hasattr(self.ml_classifier_ui, 'update_results'):
                     self.ml_classifier_ui.update_results(self.current_classification_status)
                 else:
                     print("Error: ML Classifier UI has no 'update_results' method.")
                 
                 self.last_ml_update_time = time.time()

    def update_hr_display(self, inst_rate: int, avg_rate: int) -> None:
        """
        Updates the HR LCD display and handles alarms based on average rate.
        Separated from the original do_update for clarity.
        """
        display_rate = 0
        if inst_rate > 0: # Use average rate if available and checked, else instant
            if self.actionBPM_Averaging.isChecked() and avg_rate > 0:
                display_rate = avg_rate
            else:
                display_rate = inst_rate
        
        self.lcdNumber.display(display_rate)

        # Handle display message (Signal Loss)
        if inst_rate <= 0 and self.useSerialRadio.isChecked() and self.ser.is_open:
             # Only show signal loss if serial is active source
             # self.ui_set_message("SIGNAL LOSS") # This might be handled elsewhere now
             pass
        else:
             self.ui_clear_message() # Clear message if rate is valid or using generator

        # Handle Alarms based on average rate
        current_avg_rate = avg_rate if avg_rate > 0 else display_rate # Use best available rate for alarm check
        if current_avg_rate > 0:
            high_limit = self.high_limit_box.value()
            low_limit = self.low_limit_box.value()

            if current_avg_rate > high_limit:
                if not self.rate_alarm_active:
                    self.rate_alarm_active = True
                    self.ui_alarm_on("MAX RATE ALARM")
            elif current_avg_rate < low_limit:
                if not self.rate_alarm_active:
                    self.rate_alarm_active = True
                    self.ui_alarm_on("MIN RATE ALARM")
            else:
                # Rate is within limits
                if self.rate_alarm_active:
                    self.rate_alarm_active = False
                    self.ui_alarm_off()

    def ser_com_refresh(self) -> None:
        """ Modified to respect data source selection """
        if self.useSerialRadio.isChecked():
             # Call the original function imported from the handler module (without passing self)
             self.original_ser_com_refresh()
        else:
             # Clear ports if using generator
             self.port_combo_box.clear()
             self.port_combo_box.setEnabled(False)
             self.button_connect.setEnabled(False)
             self.button_refresh.setEnabled(False)

    def connect_toggle(self) -> None:
        """ Modified to prevent connection if generator is active """
        if self.useGeneratorRadio.isChecked():
             self.ui_statusbar_message("Switch to Serial Input to connect.")
             return

        # Original connect/disconnect logic using imported functions
        if not self.ser.isOpen():
            # Call the original connect function (without passing self)
            if self.original_ser_com_connect(): 
                self.button_refresh.setDisabled(True)
                self.button_run.setDisabled(False)
                self.actionStart_Stop.setDisabled(False)
                self.button_connect.setText("Disconnect")
                self.invert_modifier = 1
                self.reset() # Reset graph and data on successful connect
                self.ser_start_capture_timer() # Call imported timer start
        else:
            try:
                self.ser_stop_capture_timer() # Call imported timer stop
                self.ser.close()
            except OSError as err_msg:
                logging.error(f"Error closing serial port: {err_msg}")
                del self.ser
                self.ser = serial.Serial(baudrate = 115200, timeout = 1, write_timeout = 1)
            finally:
                self.button_refresh.setDisabled(False)
                self.button_run.setDisabled(True)
                self.actionStart_Stop.setDisabled(True)
                self.button_connect.setText("Connect")
                # Call the original refresh function (without passing self)
                self.original_ser_com_refresh()

    def reset(self) -> None:
        """ Modified reset to handle generator state and restore original logic """
        # Original reset logic for graph and data buffers
        self.graph.clear()
        # Re-create the plot curve
        pen = self.green_pen if not self.actionBold_Line.isChecked() else pg.mkPen('g', width=4)
        self.curve = self.graph.plot(numpy.arange(self.value_history_max), numpy.zeros(self.value_history_max), pen=pen, skipFiniteCheck=True)

        self.value_history = numpy.zeros(self.value_history_max)
        self.time_history  = numpy.zeros(self.value_history_max)
        self.capture_index = 0
        self.calibrating = self.value_history_max + 1 # Reset calibration state
        self.invert_modifier = 1 # Reset inversion
        self.peaks = list() # Clear detected peaks
        self.rate_history = [80] * 3 # Reset HR history
        self.ui_alarm_off() # Turn off any active alarm
        self.rate_alarm_active = False

        # Reset generator state if applicable
        if SIMULATOR_AVAILABLE and self.ecg_generator:
            self.ecg_generator.reset()
            # Reset UI to default generator state only if generator is currently active
            if self.useGeneratorRadio.isChecked():
                self.heartRateSlider.setValue(75)
                self.noiseSlider.setValue(10)
                self.ecgTypeCombo.setCurrentIndex(0) # Assuming index 0 is 'Normal'
                self.update_generator_hr(75)
                self.update_generator_noise(10)
                self.update_generator_type()

        # Reset generator-specific timing state
        if hasattr(self, 'last_gen_time_ms'):
            del self.last_gen_time_ms
        if self.capture_timer_qt.isValid():
            self.capture_timer_qt.invalidate()

        # Apply default graph fit
        self.graph_fit()

        # Reset status bar etc.
        self.ui_statusbar_message("Reset complete.")
        self.lcdNumber.display(0)

    def window_size_update(self):
        """
        Updates value_history_max size based on the selection from the UI. Calls
        reset on exit to resize/redraw the graph to fit the new window size.
        """

        self.value_history_max = self.WindowSizeGroup.checkedAction().data()
        self.graph_fit()
        self.reset()
    
    # --- Simulator Control Methods --- NEW CODE

    def set_simulator_controls_enabled(self, enabled):
        """Enable/disable simulator-specific UI controls."""
        if hasattr(self, 'ecgTypeCombo'): # Check if UI elements exist
            self.ecgTypeCombo.setEnabled(enabled)
            self.heartRateSlider.setEnabled(enabled)
            self.noiseSlider.setEnabled(enabled)
            self.heartRateLabel.setEnabled(enabled)
            self.noiseLabel.setEnabled(enabled)
            self.currentHeartRateLabel.setEnabled(enabled)
            self.currentNoiseLabel.setEnabled(enabled)
            self.ecgTypeLabel.setEnabled(enabled)

    def switch_data_source(self):
        """Handle switching between Serial and Generator data sources."""
        use_generator = self.useGeneratorRadio.isChecked()
        self.set_simulator_controls_enabled(use_generator)

        if use_generator:
            # Stop serial capture if it was running
            if self.ser.is_open:
                self.connect_toggle() # Use existing toggle to disconnect
            self.ser_com_refresh() # Update port list state
            self.port_combo_box.setEnabled(False)
            self.button_connect.setEnabled(False)
            self.button_refresh.setEnabled(False)
            # Start using generator data (handled in do_update)
            self.ui_statusbar_message("Switched to ECG Generator.")
        else:
            # Stop generator (if applicable) and enable serial controls
            self.port_combo_box.setEnabled(True)
            self.button_connect.setEnabled(True)
            self.button_refresh.setEnabled(True)
            # Re-enable connect button if ports are available
            self.ser_com_refresh()
            self.ui_statusbar_message("Switched to Serial Input.")

        # Reset graph/data when switching source
        self.reset()
        # Restart capture timer if it was running
        if self.button_run.text() == "Stop":
            self.ser_start_capture_timer()

    def update_generator_hr(self, value):
        """Update heart rate for the ECG generator."""
        if self.ecg_generator:
            self.ecg_generator.set_heart_rate(value)
            self.currentHeartRateLabel.setText(str(value))

    def update_generator_noise(self, value):
        """Update noise level for the ECG generator."""
        if self.ecg_generator:
            noise_level = value / 100.0 # Slider is 0-100, map to 0.0-1.0
            self.ecg_generator.set_noise_level(noise_level)
            self.currentNoiseLabel.setText(f"{noise_level:.2f}")

    def update_generator_type(self):
        """Update ECG type for the generator."""
        if self.ecg_generator:
            selected_type = self.ecgTypeCombo.currentText()
            self.ecg_generator.set_ecg_type(selected_type)


@debug_timer
def check_resolution(app: QtWidgets.QApplication) -> None:
    """
    Checks the resolution to make sure it meets or exceeds the reccomended size.\n
    Displays a message to the user\n
    Does not prevent the program from running if the resolution is too low.
    """

    screen = app.primaryScreen().size()
    size_string = f"{screen.width()}x{screen.height()}"
    logging.info(f"Detected resolution: {size_string}")
    if(screen.width() < 1024 or screen.height() < 768):
        error_message = QtWidgets.QMessageBox()
        error_message.setWindowTitle("Notice")
        error_message.setText("The reccomended minimum display resolution is 1024x768.\n\nYour resolution: " + size_string)
        error_message.exec_()


def main():
    """
    Main Function.
    Starts logging system and GUI.
    Passes control to the ECGViewer class. 
    """

    # Init logging
    log_system.init_logging()
    start_time = time.time()

    # start program
    app = QtWidgets.QApplication(sys.argv)
    main_app = ECGViewer()
    main_app.show()
    check_resolution(app)
    ret = app.exec_()       # main loop call
    
    # Close program
    logging.info("PROGRAM EXIT")
    logging.info(f"Runtime: {time.time() - start_time}")
    logging.shutdown()
    sys.exit(ret)


if __name__ == '__main__':
    main()
