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

import math
import numpy
import logging
import pyqtgraph as pg
from PyQt5 import QtCore
from scipy.signal import savgol_filter


def graph_draw(self) -> None:
    """
    Draws the data stored in self.value_history to the pyqtgraph window.\n
    Pulls the status of the 'Show Signal Tracking' checkbox.\n
    If the checkbox is checked, it draws the raw, unfiltered waveform with tracking information.\n
    If the checkbox is not checked, it draws the savgol-filtered data.
    """

    # display the filtered graph
    if not self.show_track.isChecked():
        try:
            fdat = savgol_filter(
                self.value_history,
                window_length = self.window_length_box.value(),
                polyorder = self.polyorder_box.value(),
                mode = 'interp',
                )[25:self.value_history_max - 25]
        except ValueError as e:
            self.window_length_box.setValue(199)
            self.polyorder_box.setValue(7)
            logging.warning(f"Invalid filter settings: \n{e}")
            return
        # Use the corresponding time history for the X axis
        time_data_slice = self.time_history[25:self.value_history_max - 25]
        self.curve.setData(time_data_slice, fdat, skipFiniteCheck = True)

        # --- ADD SCROLLING LOGIC HERE ---
        if self.capture_index > 0: # Ensure we have some data
            try:
                latest_time_ms = self.time_history[-1]
                start_time_ms = max(0, latest_time_ms - 20000) # 20 seconds window
                logging.debug(f"[Grapher Debug] Setting XRange (20s): {start_time_ms:.0f}ms to {latest_time_ms:.0f}ms")
                # Set padding=0 to prevent auto-scaling beyond the desired range
                self.graph.setXRange(start_time_ms, latest_time_ms, padding=0)
            except Exception as e:
                logging.warning(f"Error setting graph XRange in graph_draw: {e}")

    # Otherwise, display raw waveform with tracking information. VERY SLOW IF ENABLED
    else:
        mean = self.value_history.mean()
        center = self.math_detect_peaks()
        self.graph.clear()
        self.graph.plot(numpy.arange(self.value_history.size), self.value_history, pen = self.green_pen, skipFiniteCheck = True)
        center_line = pg.InfiniteLine(pos = center, angle = 0, movable = False, pen = self.yellow_pen)
        self.graph.addItem(center_line)
        mean_line = pg.InfiniteLine(pos = mean, angle = 0, movable = False, pen = self.red_pen)
        self.graph.addItem(mean_line)

        # display a vertical line intersecting each detected peak
        for p in self.peaks:
            l = pg.InfiniteLine(pos = p, angle = 90, movable = False)
            self.graph.addItem(l)

        # display holdoff
        for p in self.peaks:
            l = pg.InfiniteLine(pos = p + self.holdoff_box.value(), angle = 90, movable = False, pen = pg.mkPen(color=(200, 200, 255)))
            self.graph.addItem(l)


def graph_fit(self) -> None:
    """
    When called, this automatically rescales the graph to fit the data plus
    a padding factor. The padding factor is fetched from the "zoom" slider in the
    GUI. This function is fairly slow and should only be called periodically. By
    default, it's only called once a complete sample period has elapsed.
    """

    self.graph_padding_factor = self.graph_zoom_slider.value() / 100
    high = self.value_history.max()
    low  = self.value_history.min()
    pad = 0
    
    # Adjust scaling based on whether we're using millivolts or ADC values
    if getattr(self, 'use_millivolts', False):
        # Millivolt mode - use fixed range for medical ECG display
        # Standard ECG display is typically -0.5mV to +1.5mV
        pad = 0.2  # 0.2mV padding
        
        # Update Y-axis label to show millivolts
        if hasattr(self, 'graph') and hasattr(self.graph, 'setLabel'):
            self.graph.setLabel('left', 'Voltage', units='mV')
    else:
        # ADC mode - calculate padding based on signal range
        pad = ((high) - (low)) * self.graph_padding_factor
        
        # Update Y-axis label to show ADC units
        if hasattr(self, 'graph') and hasattr(self.graph, 'setLabel'):
            self.graph.setLabel('left', 'ADC Value', units='')
    
    self.graph.setRange(
        xRange = (0, self.value_history_max),
        yRange = (high + pad, low - pad)
    )


def graph_bold_toggle(self) -> None:
    """
    Doubles the width of the "pen" used to draw the graph. There is a performance
    penalty associated with a heavier line. On slower computers, it can be useful
    to reduce the thickness of the line to increase speed. This should be called 
    from a UI checkbox" 
    """

    if self.actionBold_Line.isChecked():
        self.green_pen = pg.mkPen('g', width = 2)
    else:
        self.green_pen = pg.mkPen('g', width = 1)
    self.graph.clear()
    self.curve = self.graph.plot(numpy.arange(self.value_history.size), self.value_history, pen = self.green_pen, skipFiniteCheck = True)


def graph_stop_timer(self):
    """Stops the graph update timer."""
    if self.graph_timer.isActive():
        self.graph_timer.stop()


def graph_start_timer(self):
    """Starts the graph update timer."""
    if not self.graph_timer.isActive():
        self.graph_timer.start(self.graph_timer_ms)


def graph_restart_timer(self):
    """
    Updates the refresh rate timings for the graph drawing routine. If the
    timer is running, the timer will stop and restart with the new timing.
    This will not start the timer if the timer wasn't running when called. 
    """
    self.graph_frame_rate = self.FPSGroup.checkedAction().data()
    self.graph_timer_ms = int(1 / (self.graph_frame_rate / 1000))
    if self.graph_timer.isActive():
        self.graph_timer.stop()
        self.graph_timer.start(self.graph_timer_ms)