import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy.signal as signal

# Load the ECG data
data = pd.read_csv('sample_ecg.csv')
print(f'Total samples: {len(data)}')
print(f'Duration in seconds: {data.time.iloc[-1]/1000}')

# Get the values
ecg_values = data['value'].values
time_values = data['time'].values / 1000  # Convert to seconds

# Apply a bandpass filter to remove noise
fs = 250  # Sample rate in Hz (from the conversion)
lowcut = 0.5  # Hz
highcut = 50  # Hz

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# Filter the signal
filtered_ecg = butter_bandpass_filter(ecg_values, lowcut, highcut, fs)

# Find R-peaks (the highest points in the ECG)
# We need to normalize the data first for better peak detection
normalized_ecg = (filtered_ecg - np.min(filtered_ecg)) / (np.max(filtered_ecg) - np.min(filtered_ecg))

# Try different peak detection methods and parameters
print("\n=== MULTIPLE DETECTION METHODS ===")

# Method 1: Standard peak detection with distance and prominence
peaks1, _ = find_peaks(normalized_ecg, distance=fs*0.5, prominence=0.3)
if len(peaks1) >= 2:
    peak_times = time_values[peaks1]
    intervals = np.diff(peak_times)
    avg_interval = np.mean(intervals)
    bpm1 = 60 / avg_interval
    print(f"Method 1: {len(peaks1)} peaks, BPM: {bpm1:.1f}")
else:
    print("Method 1: Not enough peaks detected")

# Method 2: More sensitive detection
peaks2, _ = find_peaks(normalized_ecg, distance=fs*0.2, prominence=0.1)
if len(peaks2) >= 2:
    peak_times = time_values[peaks2]
    intervals = np.diff(peak_times)
    avg_interval = np.mean(intervals)
    bpm2 = 60 / avg_interval
    print(f"Method 2: {len(peaks2)} peaks, BPM: {bpm2:.1f}")
else:
    print("Method 2: Not enough peaks detected")

# Method 3: Height-based detection
peaks3, _ = find_peaks(normalized_ecg, height=0.5, distance=fs*0.3)
if len(peaks3) >= 2:
    peak_times = time_values[peaks3]
    intervals = np.diff(peak_times)
    avg_interval = np.mean(intervals)
    bpm3 = 60 / avg_interval
    print(f"Method 3: {len(peaks3)} peaks, BPM: {bpm3:.1f}")
else:
    print("Method 3: Not enough peaks detected")

# Plot all detected peaks for comparison
plt.figure(figsize=(15, 10))

# Plot 1: Original Signal
plt.subplot(2, 1, 1)
plt.plot(time_values, normalized_ecg)
plt.title('Normalized ECG Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot 2: Signal with different peak detections
plt.subplot(2, 1, 2)
plt.plot(time_values, normalized_ecg)
if len(peaks1) > 0:
    plt.plot(time_values[peaks1], normalized_ecg[peaks1], 'ro', label=f'Method 1: {bpm1:.1f} BPM')
if len(peaks2) > 0:
    plt.plot(time_values[peaks2], normalized_ecg[peaks2], 'go', label=f'Method 2: {bpm2:.1f} BPM')
if len(peaks3) > 0:
    plt.plot(time_values[peaks3], normalized_ecg[peaks3], 'bo', label=f'Method 3: {bpm3:.1f} BPM')
plt.title('ECG with Different Peak Detection Methods')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('ecg_multiple_detection.png')
print("Comparison plot saved to 'ecg_multiple_detection.png'")

# Now let's check what might be causing the high BPM in the ECG viewer
# Let's look at the first 5 seconds of data to see if there are high-frequency oscillations
plt.figure(figsize=(15, 5))
plt.plot(time_values[:5*int(fs)], ecg_values[:5*int(fs)])
plt.title('First 5 seconds of raw ECG data')
plt.xlabel('Time (seconds)')
plt.ylabel('ECG Value')
plt.grid(True)
plt.savefig('ecg_first_5sec.png')
print("First 5 seconds plot saved to 'ecg_first_5sec.png'")

# Analyze the frequency of value changes that could be misinterpreted as peaks
print("\n=== FREQUENCY ANALYSIS ===")
# Count the number of times the signal crosses its mean value (potential issue for peak detection)
mean_value = np.mean(ecg_values)
crossings = np.where(np.diff(np.signbit(ecg_values - mean_value)))[0]
crossing_rate = len(crossings) / (time_values[-1] - time_values[0])
print(f"Signal crosses its mean {len(crossings)} times in {time_values[-1]:.1f} seconds")
print(f"Crossing rate: {crossing_rate:.1f} per second, which could result in {crossing_rate*60:.1f} BPM if each crossing is detected as a beat") 