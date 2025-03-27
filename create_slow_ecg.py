import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy.signal as signal

# Create a new ECG signal with fewer, clearer peaks that won't confuse the ECG viewer
# We'll base it on our detected peaks from the smoothed data but make them more pronounced

# First, load the smoothed data
data = pd.read_csv('sample_ecg_smoothed.csv')
print(f'Loaded smoothed data: {len(data)} samples spanning {data.time.iloc[-1]/1000:.1f} seconds')

# Get the values
time_values = data['time'].values
ecg_values = data['value'].values

# Apply an even stronger filter to simplify the signal
fs = 250  # Sample rate in Hz
order = 6  # Higher order for sharper cutoff
cutoff = 1.0  # Even lower cutoff to remove more high-frequency components
nyq = 0.5 * fs
normalized_cutoff = cutoff / nyq
b, a = signal.butter(order, normalized_cutoff, btype='low')
ultra_smooth_ecg = signal.filtfilt(b, a, ecg_values)

# Use a peak detector on this ultra-smooth data
normalized_ecg = (ultra_smooth_ecg - np.min(ultra_smooth_ecg)) / (np.max(ultra_smooth_ecg) - np.min(ultra_smooth_ecg))

from scipy.signal import find_peaks
peaks, _ = find_peaks(normalized_ecg, distance=fs*0.5, prominence=0.2)

if len(peaks) >= 2:
    peak_times = time_values[peaks] / 1000  # Convert to seconds
    intervals = np.diff(peak_times)
    avg_interval = np.mean(intervals)
    bpm = 60 / avg_interval
    print(f"Detected {len(peaks)} peaks, calculated heart rate: {bpm:.1f} BPM")

# Now create a very clean, simplified ECG signal based on these peaks
# Start with a baseline value
baseline = 400
simplified_ecg = np.ones(len(time_values)) * baseline

# Add clear, prominent peaks at each detected R-peak
for peak_idx in peaks:
    # Create a 100-sample window around each peak (50 before, 50 after)
    start_idx = max(0, peak_idx - 30)
    end_idx = min(len(simplified_ecg), peak_idx + 30)
    
    # Create a triangular peak (very simple, clear ECG-like shape)
    peak_height = 500  # Prominent peak
    for i in range(start_idx, end_idx):
        # Distance from peak (0 at peak, increases as we move away)
        dist = abs(i - peak_idx)
        if dist < 10:
            # Create a clear, triangular peak
            height_factor = 1 - (dist / 10)
            simplified_ecg[i] = baseline + (peak_height * height_factor)

# Create a new DataFrame with the simplified data
simplified_data = pd.DataFrame({
    'time': time_values,
    'value': simplified_ecg
})

# Save to a new file
simplified_data.to_csv('sample_ecg_simple.csv', index=False)
print(f'Simplified ECG data saved to sample_ecg_simple.csv')

# Plot comparison of original vs. simplified data
plt.figure(figsize=(15, 10))

# Original data
plt.subplot(3, 1, 1)
plt.plot(time_values/1000, data['value'].values)
plt.title('Original ECG Data')
plt.xlabel('Time (seconds)')
plt.ylabel('Value')
plt.grid(True)

# Smoothed data with detected peaks
plt.subplot(3, 1, 2)
plt.plot(time_values/1000, ultra_smooth_ecg)
if len(peaks) > 0:
    plt.plot(time_values[peaks]/1000, ultra_smooth_ecg[peaks], 'ro')
plt.title(f'Ultra Smoothed ECG Data with Detected Peaks (Heart Rate: {bpm:.1f} BPM)')
plt.xlabel('Time (seconds)')
plt.ylabel('Value')
plt.grid(True)

# Simplified data
plt.subplot(3, 1, 3)
plt.plot(time_values/1000, simplified_ecg)
plt.title('Simplified ECG Data (Should be correctly interpreted by ECG Viewer)')
plt.xlabel('Time (seconds)')
plt.ylabel('Value')
plt.grid(True)

plt.tight_layout()
plt.savefig('ecg_simplified.png')
print('Comparison plot saved to ecg_simplified.png')

# Count zero crossings in the simplified data
mean_value = np.mean(simplified_ecg)
crossings = np.where(np.diff(np.signbit(simplified_ecg - mean_value)))[0]
crossing_rate = len(crossings) / (time_values[-1]/1000 - time_values[0]/1000)
print(f"Simplified signal crosses its mean {len(crossings)} times in {time_values[-1]/1000:.1f} seconds")
print(f"Crossing rate: {crossing_rate:.1f} per second, which would result in {crossing_rate*60:.1f} BPM if each crossing is detected as a beat")
print(f"This should match closely with our calculated {bpm:.1f} BPM") 