import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import scipy.signal as signal

# Load the original ECG data
data = pd.read_csv('sample_ecg.csv')
print(f'Original data: {len(data)} samples spanning {data.time.iloc[-1]/1000:.1f} seconds')

# Get the values
ecg_values = data['value'].values
time_values = data['time'].values

# Apply a stronger low-pass filter to smooth the signal and remove high-frequency noise
fs = 250  # Sample rate in Hz
order = 4  # Filter order
cutoff = 5.0  # Cutoff frequency in Hz
nyq = 0.5 * fs
normalized_cutoff = cutoff / nyq
b, a = signal.butter(order, normalized_cutoff, btype='low')
smoothed_ecg = signal.filtfilt(b, a, ecg_values)

# Apply Savitzky-Golay filter for additional smoothing
window_length = 51  # Must be odd and less than the data length
smoothed_ecg = savgol_filter(smoothed_ecg, window_length, 3)

# Create a new DataFrame with the smoothed data
smoothed_data = pd.DataFrame({
    'time': time_values,
    'value': smoothed_ecg
})

# Save to a new file
smoothed_data.to_csv('sample_ecg_smoothed.csv', index=False)
print(f'Smoothed data saved to sample_ecg_smoothed.csv')

# Analyze the smoothed data
normalized_ecg = (smoothed_ecg - np.min(smoothed_ecg)) / (np.max(smoothed_ecg) - np.min(smoothed_ecg))
peaks, _ = find_peaks(normalized_ecg, distance=fs*0.5, prominence=0.3)

if len(peaks) >= 2:
    peak_times = time_values[peaks] / 1000  # Convert to seconds
    intervals = np.diff(peak_times)
    avg_interval = np.mean(intervals)
    bpm = 60 / avg_interval
    print(f"Detected {len(peaks)} peaks in smoothed data, calculated heart rate: {bpm:.1f} BPM")

# Plot comparison of original vs. smoothed data
plt.figure(figsize=(15, 10))

# Original data
plt.subplot(2, 1, 1)
plt.plot(time_values/1000, ecg_values)
plt.title('Original ECG Data')
plt.xlabel('Time (seconds)')
plt.ylabel('Value')
plt.grid(True)

# Smoothed data with detected peaks
plt.subplot(2, 1, 2)
plt.plot(time_values/1000, smoothed_ecg)
if len(peaks) > 0:
    plt.plot(time_values[peaks]/1000, smoothed_ecg[peaks], 'ro')
plt.title(f'Smoothed ECG Data with Detected Peaks (Heart Rate: {bpm:.1f} BPM)')
plt.xlabel('Time (seconds)')
plt.ylabel('Value')
plt.grid(True)

plt.tight_layout()
plt.savefig('ecg_original_vs_smoothed.png')
print('Comparison plot saved to ecg_original_vs_smoothed.png')

# Count zero crossings in smoothed data
mean_value = np.mean(smoothed_ecg)
crossings = np.where(np.diff(np.signbit(smoothed_ecg - mean_value)))[0]
crossing_rate = len(crossings) / (time_values[-1]/1000 - time_values[0]/1000)
print(f"Smoothed signal crosses its mean {len(crossings)} times in {time_values[-1]/1000:.1f} seconds")
print(f"Crossing rate: {crossing_rate:.1f} per second, which would result in {crossing_rate*60:.1f} BPM if each crossing is detected as a beat")
print(f"This is significantly lower than the ~800+ BPM in the original data") 