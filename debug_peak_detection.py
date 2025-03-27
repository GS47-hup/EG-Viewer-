import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# Check for the final millivolt ECG data
csv_path = 'sample_ecg_final_mv.csv'
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

# Load the ECG data
data = pd.read_csv(csv_path)
ecg_data = data['value'].values
sample_rate = 250  # Hz

# Calculate time array
time = np.arange(len(ecg_data)) / sample_rate

# Detection parameters (same as in _ecg_math.py)
center = 0.5  # mV
prominence = 0.7  # mV
distance = int(sample_rate * 0.5)  # Expected time between beats at 80 BPM

# Detect peaks
peaks, peak_props = find_peaks(ecg_data, 
                              height=(center, None),  # Only peaks above center
                              prominence=prominence,  # Must be prominent peaks
                              distance=distance)      # Minimum distance between peaks

# Calculate BPM
if len(peaks) >= 2:
    peak_times = time[peaks]
    intervals = np.diff(peak_times)
    mean_interval = np.mean(intervals)
    bpm = 60 / mean_interval
    print(f"Detected {len(peaks)} peaks")
    print(f"Average interval: {mean_interval:.3f} seconds")
    print(f"Calculated heart rate: {bpm:.1f} BPM")
else:
    print(f"Not enough peaks detected: {len(peaks)}")

# Plot the ECG with detected peaks
plt.figure(figsize=(15, 6))
plt.plot(time, ecg_data, 'b-', label='ECG Signal')
plt.plot(time[peaks], ecg_data[peaks], 'ro', label='Detected Peaks')
plt.axhline(y=center, color='g', linestyle='--', label='Detection Threshold')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (mV)')
plt.title('ECG Peak Detection')
plt.legend()
plt.grid(True)
plt.savefig('peak_detection_debug.png')
plt.close()

print(f"Debug plot saved to peak_detection_debug.png")

# If we have the peaks, calculate all intervals and display statistics
if len(peaks) >= 2:
    print("\nDetailed statistics:")
    intervals_ms = intervals * 1000  # Convert to milliseconds
    print(f"Min interval: {np.min(intervals_ms):.1f} ms")
    print(f"Max interval: {np.max(intervals_ms):.1f} ms")
    print(f"Std deviation: {np.std(intervals_ms):.1f} ms")
    
    # Calculate instantaneous BPM for each interval
    inst_bpm = 60 / intervals
    print(f"Min BPM: {np.min(inst_bpm):.1f}")
    print(f"Max BPM: {np.max(inst_bpm):.1f}")
    print(f"Std deviation BPM: {np.std(inst_bpm):.1f}") 