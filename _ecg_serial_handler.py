#!/usr/bin/python3
#
#            ECG Viewer
#   Written by Kevin Williams - 2022
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

import time
import logging
import serial
import serial.tools.list_ports
import statistics as stat
from debug import debug_timer
import math
import random
import numpy as np
import pandas as pd
import os


# refresh available devices, store in dropdown menu storage
def ser_com_refresh(self):
    """
    Refreshes the list of available serial devices.\n
    Results are stored in the dropdown menu.\n
    Uses addItem to store the device string."""
    self.port_combo_box.clear()
    
    # Add simulation option at the top
    self.port_combo_box.addItem("SIMULATION MODE (No Arduino required)", "SIM")
    
    available_ports = serial.tools.list_ports.comports()
    for device in available_ports:
        d_name = device.device + ": " + device.description
        self.port_combo_box.addItem(d_name, device.device)
        logging.info(f"Detected port: {d_name}")


@debug_timer
def ser_check_device(self) -> bool:
    """
    Checks to see if the Arduino is responding the way we expect.\n
    Returns True if device is responding properly.\n
    Returns False if device is not responding or is giving improper responses.
    """
    
    # Get the current port from the ser object
    current_port = getattr(self.ser, 'port', None)
    
    # Simple direct string comparison for simulation mode
    if current_port == "SIM":
        self.ui_statusbar_message('Connected to ECG SIMULATION')
        return True

    # Regular hardware device checking
    self.ui_statusbar_message('Connecting...')
    max_attempts = 10
    device_ok = False
    while max_attempts > 0 and not device_ok:
        try:
            self.ser.write('\n'.encode())
            self.ser.flush()
            while self.ser.inWaiting() > 0:
                c = str(self.ser.read().decode())
                if c == '$':
                    device_ok = True
                    break
        except Exception as e:
            logging.debug(f"Retrying connection: {e}")
            time.sleep(1)
        max_attempts -= 1
        time.sleep(0.2)
    return device_ok


@debug_timer
def ser_com_connect(self) -> bool:
    """
    Connect/Disconnect from the serial device selected in the devices dropdown menu.\n
    Returns True if the connection was sucessful.\n
    False if the connection was unsucessful.
    """

    # fetch port name from dropdown menu
    try:
        current_index = self.port_combo_box.currentIndex()
        com_port = self.port_combo_box.itemData(current_index)
        if not com_port:
            raise ValueError
    except ValueError:
        self.ui_statusbar_message('No device selected!')
        return False
    except TypeError as e:
        self.ui_display_error_message("Invalid port type", e)
        logging.error(e)
        return False

    # Special handling for simulation mode
    if com_port == "SIM":
        self.ser.port = com_port
        # For simulation, we don't need to open a real port
        # Just set it as "SIM" and consider the connection successful
        self.ui_statusbar_message('Connected to ECG SIMULATION')
        logging.info(f"Connected to simulation mode")
        return True
        
    # connect to real port
    try:
        self.ser.port = com_port
        self.ser.open()
    except serial.serialutil.SerialException as e:
        self.ui_display_error_message("Connection Failure", e)
        logging.warning(f"Connection Failure: {e}")
        return False

    # detect if device is responding properly
    if not self.ser_check_device():
        logging.info(f"Serial device check failed: {self.ser.port}")
        self.ui_display_error_message("Device Error", "Connected device is not responding.\n\nThis may be the incorrect device. Please choose a different device in the menu and try again.")
        self.ser.close()
        return False

    # device is connected and test has passed
    logging.info(f"Connection to {com_port} succesful.")
    return True


# Fetch a value from the Arduino
def ser_get_input(self) -> bool:
    """Fetches a measurement from the Arduino, stores value in value_history and time_history.\n
    Returns True if reading was valid.
    Returns False if reading was invalid or unsucessful.
    """
    
    # Get the current port from the ser object 
    current_port = getattr(self.ser, 'port', None)
    
    # SIMULATION MODE: Generate fake ECG data if the port is "SIM"
    if current_port == "SIM":
        # Check if we've loaded the CSV data already
        if not hasattr(self, 'ecg_csv_data'):
            # Load sample ECG data from CSV file
            self.use_algorithmic_ecg = False
            self.use_millivolts = True  # Set to true to use millivolt-based display
            print("Using millivolt-based ECG display")
            
            # Let's first check if we have test data in the working directory
            csv_path = None
            self.is_millivolt_data = False
            self.ecg_source_file = "default"

            # Priority order for files:
            # 1. Original sample ECG converted to millivolts
            if os.path.exists("sample_ecg_mv_original.csv"):
                csv_path = "sample_ecg_mv_original.csv"
                self.is_millivolt_data = True
                self.ecg_source_file = "original_millivolts"
                print("Using original ECG data converted to millivolts")
            # 2. Specialized final millivolt ECG data
            elif os.path.exists("sample_ecg_final_mv2.csv"):
                csv_path = "sample_ecg_final_mv2.csv"
                self.is_millivolt_data = True
                self.ecg_source_file = "millivolts_final"
                print("Using simplified ECG data in millivolts with 80 BPM heart rate")
            # 3. Regular millivolt ECG data
            elif os.path.exists("sample_ecg_final_mv.csv"):
                csv_path = "sample_ecg_final_mv.csv"
                self.is_millivolt_data = True
                self.ecg_source_file = "millivolts_final"
                print("Using simplified ECG data in millivolts with 80 BPM heart rate")
            # 4. Regular millivolt ECG data
            elif os.path.exists("sample_ecg_mv.csv"):
                csv_path = "sample_ecg_mv.csv"
                self.is_millivolt_data = True
                self.ecg_source_file = "millivolts"
                print("Using millivolt ECG data with 60 BPM heart rate")
            # 5. Optimal raw ADC ECG data 
            elif os.path.exists("sample_ecg_optimal.csv"):
                csv_path = "sample_ecg_optimal.csv"
                self.ecg_source_file = "optimal"
                print("Using optimal ECG data with 80 BPM heart rate")
            # 6. Regular raw ADC test data
            elif os.path.exists("sample_ecg.csv"):
                csv_path = "sample_ecg.csv"
                print("Using standard ECG test data")
            else:
                # Fall back to algorithmic generation if no CSV file exists
                logging.warning("No ECG CSV file found. Using algorithmic ECG generation.")
                self.use_algorithmic_ecg = True
                self.ecg_source_file = None
                self.is_millivolt_data = False
                csv_path = None
            
            if csv_path:
                try:
                    # Load the CSV data
                    self.ecg_csv_data = pd.read_csv(csv_path)
                    self.ecg_data_index = 0
                    self.csv_sample_counter = 0
                    self.ui_statusbar_message(f'Loaded sample ECG data from {csv_path}')
                    logging.info(f"Loaded ECG data from {csv_path}")
                except Exception as e:
                    # Fall back to algorithmic generation if CSV loading fails
                    logging.warning(f"Failed to load ECG CSV: {e}. Using algorithmic generation.")
                    self.use_algorithmic_ecg = True
            
        # If we have CSV data, use it
        if hasattr(self, 'ecg_csv_data') and not getattr(self, 'use_algorithmic_ecg', False):
            # Get the next value from CSV, looping back to beginning if we reach the end
            row = self.ecg_csv_data.iloc[self.ecg_data_index]
            
            # Use the value directly if we're in millivolt mode
            if getattr(self, 'use_millivolts', False):
                # For millivolt data, use the value directly
                self.current_reading = float(row['value'])
            # For ADC data, convert if needed
            elif getattr(self, 'is_millivolt_data', False):
                # Convert millivolts to Arduino's ADC range
                # For a 3.3V reference voltage, 1023 ADC units = 3300 mV
                # Assuming typical ECG values are 0-1mV, scale appropriately for visibility
                # Scale to mid-range (around 512) with ~200 units of range
                mv_value = float(row['value'])
                
                # Scale the signal to be centered at 512 with ~200 units of range
                # Typical ECG is ±0.5mV, map to 412-612 range
                MV_TO_ADC = 200  # 200 ADC units per mV for good visibility
                self.current_reading = int(512 + mv_value * MV_TO_ADC)
                
                # Ensure values stay within Arduino's analog range (0-1023)
                self.current_reading = min(max(self.current_reading, 0), 1023)
            else:
                # Regular ADC data, use as is
                self.current_reading = int(row['value'])
            
            # Control the simulation playback speed
            # For synthetic data with known BPM, we need precise timing
            if not hasattr(self, 'csv_sample_counter'):
                self.csv_sample_counter = 0
            
            self.csv_sample_counter += 1
            
            # Adjust speed based on data source
            # For synthetic data, use faster playback (closer to real-time)
            if hasattr(self, 'ecg_source_file') and self.ecg_source_file == "synthetic":
                increment_threshold = 4  # Faster rate for synthetic data
            else:
                increment_threshold = 8  # Slower rate for other data
            
            if self.csv_sample_counter >= increment_threshold:
                # Increment index and loop back to start when we reach the end
                self.ecg_data_index = (self.ecg_data_index + 1) % len(self.ecg_csv_data)
                self.csv_sample_counter = 0
        else:
            # Fallback to algorithmic generation 
            time_val = self.capture_timer_qt.elapsed() / 1000.0  # Convert to seconds
            
            # Create a single peak every 3 seconds (20 BPM) for very clear detection
            seconds_per_beat = 3.0
            beat_position = (time_val % seconds_per_beat) / seconds_per_beat
            
            # Create a mostly flat signal with a very prominent single spike
            # Baseline - flat for most of the cycle
            val = 400
            
            # Add a very large, isolated spike at one point in the cycle
            if 0.1 < beat_position < 0.15:
                # Tall, narrow spike that's easy to detect
                val = 900
            
            # Ensure values stay within Arduino's analog range (0-1023)
            self.current_reading = int(min(max(val, 0), 1023))
        
        val = self.invert_modifier * self.current_reading
        self.value_history[self.capture_index] = val
        self.time_history[self.capture_index] = self.capture_timer_qt.elapsed()
        self.capture_index = (self.capture_index + 1) % self.value_history_max
        return True

    # Regular mode: Read from actual Arduino
    # send character to Arduino to trigger the Arduino to begin a analogRead capture
    try:
        self.ser.write('\n'.encode())
    except Exception as e:
        logging.warn(f"Device write error: {e}")
        self.ser_stop_capture_timer()
        self.connect_toggle()
        err_msg = f"Connection to Arduino lost. \nPlease check cable and click connect.\n\nError information:\n{e}"
        self.ui_display_error_message("Connection Error", err_msg)
        return False

    # get response from Arduino, terminated by newline character
    buf = ''
    try:
        # read and discard incoming bytes until the start character is found
        while self.ser.inWaiting() > 0:
            chr = str(self.ser.read().decode())
            if chr == '$':
                break

        # read characters until newline is detected, this is faster than serial's read_until
        while self.ser.inWaiting() > 0:
            chr = str(self.ser.read().decode())
            if chr == '\n':
                break
            buf = buf + chr
    # disconnecting during inWaiting() may throw this
    except OSError as err_msg:
        logging.warn(err_msg)
        return False
    # this may occur during str conversion if the device is disconnected abrutply
    except UnicodeDecodeError as err_msg:
        logging.warn(err_msg)
        return False


    # all measurements are exactly three characters in size
    if len(buf) != 3:
        return False
    self.current_reading = int(buf)
    val = self.invert_modifier * self.current_reading
    self.value_history[self.capture_index] = val
    self.time_history[self.capture_index] = self.capture_timer_qt.elapsed()
    self.capture_index = (self.capture_index + 1) % self.value_history_max
    return True


def ser_do_calibrate(self) -> None:
    """
    Perform calibration. Capture data as normal until self.calibrating counter is zero.\n
    If the peak value is below the mean, invert the signal.
    """

    if self.calibrating > 0:
        self.calibrating = self.calibrating - 1
    elif self.calibrating == 0:
        self.ui_clear_message()
        window = 150
        peak_samples = 3
        temp_array = self.value_history[window:self.value_history_max - window].copy()
        temp_array.sort()
        period_mean = self.value_history[window:self.value_history_max - window].mean()
        min_delta = period_mean - stat.mean(temp_array[0:peak_samples])
        temp_array = temp_array[::-1]
        max_delta = stat.mean(temp_array[0:peak_samples]) - period_mean
        if abs(max_delta - min_delta) > 1.5:
            if self.autoinvert_checkbox.isChecked():
                if min_delta > max_delta:
                    self.invert_modifier = self.invert_modifier * -1
                    self.statusBar.showMessage('Inverting input signal')
                    logging.debug("*** INVERTING SIGNAL ***")
        else:
            logging.debug("*** NO SIGNAL DETECTED ***")
        self.calibrating = -1
        logging.debug("DYNAMIC CALIBRATION INFO:")
        logging.debug(f"RANGE     : {window} - {self.value_history_max - window}")
        logging.debug(f"PK SAMPLES: {peak_samples}")
        logging.debug(f"MEAN      : {period_mean}")
        logging.debug(f"MAX DELTA : {max_delta}")
        logging.debug(f"MIN DELTA : {min_delta}")
        logging.debug(f"CIDX      : {self.capture_index}")


def ser_stop_capture_timer(self):
    """Stops the capture timer AND graph update timer."""
    if self.capture_timer.isActive():
        self.graph_stop_timer()
        self.capture_timer.stop()


def ser_start_capture_timer(self):
    """Starts the capture timer AND graph update timer."""
    # Get the current port from the ser object
    current_port = getattr(self.ser, 'port', None)
    
    # Skip reset_input_buffer for simulation mode
    if current_port != "SIM" and getattr(self.ser, 'is_open', False):
        self.ser.reset_input_buffer()
    if not self.capture_timer.isActive():
        self.capture_timer.start(self.capture_rate_ms)
        self.graph_start_timer()
