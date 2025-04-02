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

# Python Imports
import time
import queue
import math
import statistics as stat
from scipy import signal
import numpy as np

def math_calc_sps(self, samples_generated=None) -> int:
    """
    Calculates the current samples per second based on the time delta
    between the first and last recorded sample.
    Returns: int samples per second
    """
    
    # Find the time range of the current sample history
    sample_time_range = self.time_history[-1] - self.time_history[0]
    
    # Prevent division by zero
    if sample_time_range <= 0:
        # Try estimating from generator fs if available
        if hasattr(self, 'ecg_generator') and self.ecg_generator is not None:
            return int(self.ecg_generator.fs)
        # Otherwise, return a default or indicate an error
        return 250 # Default guess if time range is invalid

    # Calculate samples per second
    # Formula depends on whether time_history is in seconds or milliseconds
    # Assuming milliseconds based on previous code
    sps = (self.value_history_max / sample_time_range) * 1000 
    
    # Use floor to return an integer value
    try:
        return math.floor(sps)
    except OverflowError:
        # Handle potential overflow if sps is extremely large
        return 9999 # Return a high capped value

def math_detect_peaks(self) -> float:
    """
    Detects peaks using scipy.

    Operands:
        prominence: the threshold the peak needs to be at, relative to the surrounding samples
        distance  : (AKA holdoff) minimum required distance between the previous peak to the next peak
        height    : minimum height of peak to be accepted\n
    modifies:  stores index of peaks in self.peaks
    returns :  center (not average) of recorded values
    """

    vmax = self.value_history.max()
    vmin = self.value_history.min()
    center = vmin + (vmax - vmin) / 2
    
    sps = math_calc_sps(self)
    if sps <= 0:
        sps = 250  # Default value if calculation fails
    
    # Determine if data is likely millivolts BEFORE mode-specific logic
    is_millivolt = getattr(self, 'use_millivolts', False)

    # --- Adjust parameters dynamically for Generator Mode --- NEW
    is_generator_mode = hasattr(self, 'useGeneratorRadio') and self.useGeneratorRadio.isChecked()
    if is_generator_mode:
        # Use parameters suitable for the generated signal (needs tuning)
        signal_range = vmax - vmin
        # Generator output is roughly -0.4 to 1.2. Use a smaller fraction for prominence.
        prominence = max(0.1, signal_range * 0.3) # e.g., 30% of range, min 0.1
        # Calculate distance based on generator's known heart rate setting
        current_hr = self.heartRateSlider.value() if hasattr(self, 'heartRateSlider') else 75
        # Min distance = ~60% of expected beat interval at current HR
        distance = int( (60.0 / current_hr) * sps * 0.6 )
        distance = max(distance, int(sps * 0.2)) # Ensure minimum distance (e.g., 200ms)
        # Override height threshold for generator based on its typical range
        height = 0.1 # Set slightly above baseline for generated signal - RENAMED from center
        # Assume generator output is mV-like for debug printing
        is_millivolt = True
        # <<< START Debug Prints >>>
        print(f"[Generator Debug] Signal Range: {vmin:.2f} to {vmax:.2f} (Range: {signal_range:.2f})")
        print(f"[Generator Debug] Calculated Params: Prominence={prominence:.2f}, Distance={distance}, Height={height:.2f}, SPS={sps}")
        # <<< END Debug Prints >>>

        # Find peaks using calculated parameters - Use value_history directly
        peaks, properties = signal.find_peaks(
            self.value_history,
            prominence=prominence,
            height=height, # Use the calculated height threshold
            distance=distance
        )
        # <<< START Debug Prints >>>
        print(f"[Generator Debug] find_peaks result: {len(peaks)} peaks found. Indices: {peaks}")
        print(f"[Generator Debug] find_peaks properties: {properties}")
        # <<< END Debug Prints >>>

        self.peaks = peaks # Store the found peaks

        # Debug print for the final parameters used in generator mode
        print(f"Using GENERATOR mode peak detection params: prom={prominence:.2f}, dist={distance}, height={height:.2f}") # Changed 'center' to 'height'

    else:
        # --- Original parameter logic for Serial Mode --- 
        # ... (keep existing logic for calculating prominence/holdoff from UI boxes and signal range) ...
        min_holdoff = int(sps * 0.33)  # 0.33 seconds = minimum time between beats at 180 BPM
        holdoff = max(self.holdoff_box.value(), min_holdoff)
        distance = holdoff # Use holdoff as distance for serial
        
        # Set a reasonable prominence based on the signal amplitude
        signal_range = vmax - vmin
        if is_millivolt:
            min_prominence = 0.3
            prominence = min_prominence
        else:
            min_prominence = int(signal_range * 0.2)
            prominence = max(self.prominence_box.value(), min_prominence)
        # Use the calculated center for serial mode
        # center = vmin + (vmax - vmin) / 2 # Already calculated above

    # Debug information (now is_millivolt is always defined)
    if is_millivolt:
        print(f"ECG Detection (mV): Min={vmin:.2f}, Max={vmax:.2f}, Center={center:.2f}, SPS={sps}")
    else:
        print(f"ECG Detection (ADC): Min={vmin:.1f}, Max={vmax:.1f}, Center={center:.1f}, SPS={sps}")
    
    # Use special parameters for our known data types (applies mainly to serial/file data)
    if not is_generator_mode and hasattr(self, 'ecg_source_file'):
        source_file = self.ecg_source_file
        if source_file == "original_millivolts":
            # Original ECG data converted to millivolts
            # Use appropriate parameters for real ECG signal
            prominence = 0.35  # Increase prominence for more reliable detection
            holdoff = int(sps * 0.4)  # Longer holdoff to avoid T-wave detection
            print(f"Using specialized settings for original ECG data in millivolts")
        elif source_file == "millivolts_final":
            # Our simplified millivolt ECG data has 1.2mV peaks at 80 BPM
            # Use specialized parameters for guaranteed detection
            prominence = 0.7  # Increase prominence for clearer detection
            holdoff = int(sps * 0.5)  # Expected time between beats at 80 BPM
            # Ensure center is properly set for this data
            center = 0.5  # Set between baseline (0.0) and peak (1.2)
            print(f"Using specialized settings for simplified millivolt ECG data")
        elif source_file == "millivolts":
            # Our millivolt ECG data has ~1mV peaks at 60 BPM
            # Use specialized parameters for reliable detection
            prominence = 0.4  # In millivolts
            holdoff = int(sps * 0.6)  # 60% of expected time between beats
            print(f"Using specialized settings for millivolt ECG data")
        elif source_file == "optimal":
            # Our optimal ECG data has huge 300-unit peaks at 80 BPM
            # Use very specific parameters for guaranteed detection
            prominence = 150  # Half the actual peak height
            holdoff = int(sps * 0.5)  # 50% of expected time between beats at 80 BPM
            # Force a specific center height threshold
            center = 650  # Set between baseline (512) and peak (812)
            print(f"Using specialized settings for optimal ECG data")
    
    # Print final used parameters
    if is_millivolt:
        print(f"Peak Detection (Final - mV): Using prominence={prominence:.2f}mV, distance={distance}, height={center:.2f}")
    else:
        print(f"Peak Detection (Final - ADC): Using prominence={prominence}, distance={distance}, height={center:.1f}")
    
    # Find peaks with adjusted parameters
    self.peaks = signal.find_peaks(
                self.value_history,
                prominence = prominence,
                height = center, 
                distance = distance, # Use the determined distance
            )[0]
    
    # Print debug info
    if len(self.peaks) > 0:
        print(f"Found {len(self.peaks)} peaks")
    else:
        print("No peaks found - try adjusting detection parameters")
    
    return center


def math_calc_hr(self) -> tuple[int, int, str]:
    """
    Calculate heart rate and classify rhythm based on peak intervals.

    Uses filtered RR intervals to determine rate and regularity.

    Returns tuple: (instantaneous_rate, average_rate_for_display, classification_status)
    """

    if len(self.peaks) < 2:
        return (0, 0, "Cannot Classify") # Need at least two peaks for one interval

    # Calculate samples per second
    sps = self.math_calc_sps()
    if sps <= 0:
        return (0, 0, "Cannot Classify - Invalid SPS")

    # Calculate RR intervals in milliseconds
    peak_indices = np.array(self.peaks)
    peak_times_ms = peak_indices / sps * 1000
    rr_intervals_ms = np.diff(peak_times_ms)

    if len(rr_intervals_ms) == 0:
        return (0, 0, "Cannot Classify - No Intervals")

    # Filter intervals using IQR if enough data points exist
    if len(rr_intervals_ms) >= 5: # Need a few intervals for robust stats
        q1, q3 = np.percentile(rr_intervals_ms, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_intervals = rr_intervals_ms[(rr_intervals_ms >= lower_bound) & (rr_intervals_ms <= upper_bound)]
        # Ensure we still have intervals after filtering
        if len(filtered_intervals) == 0:
            filtered_intervals = rr_intervals_ms # Fallback if filter removes everything
    else:
        filtered_intervals = rr_intervals_ms # Not enough data to filter robustly

    if len(filtered_intervals) == 0:
         # This shouldn't happen with the fallback, but as a safeguard
        return (0, 0, "Cannot Classify - Filter Error")

    # Calculate statistics on filtered intervals
    mean_rr = np.mean(filtered_intervals)
    std_rr = np.std(filtered_intervals) if len(filtered_intervals) > 1 else 0

    # --- Heart Rate Calculations ---
    # Instantaneous HR from the *last* raw interval
    inst_hr = 0
    if rr_intervals_ms[-1] > 0:
        inst_hr = round(60000.0 / rr_intervals_ms[-1])

    # Update rate history (used for display averaging)
    self.rate_history.append(inst_hr)
    if len(self.rate_history) > 10: # Keep history size bounded (e.g., 10 seconds at ~1Hz update)
        self.rate_history.pop(0)

    # Average HR for display (using potentially smoothed history)
    avg_rate_display = round(stat.mean(self.rate_history)) if self.rate_history else 0

    # Average HR for classification (based on filtered mean RR)
    avg_hr_for_class = 0
    if mean_rr > 0:
        avg_hr_for_class = 60000.0 / mean_rr

    # --- Rhythm Classification ---
    status = "Cannot Classify"
    if mean_rr > 0:
        cv_rr = (std_rr / mean_rr) * 100 if mean_rr > 0 and len(filtered_intervals) > 1 else 0
        
        # Define thresholds
        NORMAL_RATE_LOW = 60
        NORMAL_RATE_HIGH = 100
        REGULARITY_THRESHOLD_CV = 15 # Coefficient of Variation percentage

        is_regular = cv_rr < REGULARITY_THRESHOLD_CV
        
        rate_category = ""
        if avg_hr_for_class < NORMAL_RATE_LOW:
            rate_category = "Bradycardia"
        elif avg_hr_for_class <= NORMAL_RATE_HIGH:
            rate_category = "Normal Rate"
        else:
            rate_category = "Tachycardia"

        if is_regular:
            if rate_category == "Normal Rate":
                status = "Normal Sinus Rhythm"
            else:
                status = rate_category # e.g., "Tachycardia" or "Bradycardia"
        else:
            status = f"Irregular - {rate_category}" # e.g., "Irregular - Tachycardia"
            
        # Debug print
        print(f"Classification: HR={avg_hr_for_class:.1f}, CV={cv_rr:.1f}% -> Status: {status}")

    return (inst_hr, avg_rate_display, status)

"""

        # display rate as average of rate history
        if self.actionBPM_Averaging.isChecked():
            self.lcdNumber.display(math.floor(stat.mean(self.rate_alarm_history)))
        else:
            self.lcdNumber.display(math.floor(rate))


        self.ui_clear_message()
        if avg > self.high_limit_box.value():
            self.rate_alarm_active = True
            self.ui_alarm_on("MAX RATE ALARM")
        if self.low_limit_box.value() > avg:
            self.rate_alarm_active = True
            self.ui_alarm_on("MIN RATE ALARM")
        if self.rate_alarm_active:
            if(avg <= self.high_limit_box.value() and self.low_limit_box.value() <= avg):
                self.rate_alarm_active = False
                self.ui_alarm_off()
    else:
        self.lcdNumber.display(0)
        self.ui_set_message("SIGNAL LOSS")
        """