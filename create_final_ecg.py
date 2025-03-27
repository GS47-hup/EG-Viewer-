import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic ECG signal with very prominent, easy-to-detect peaks
# This will have a known, fixed heart rate of 60 BPM

# Constants
SAMPLE_RATE = 250  # Hz (samples per second)
DURATION = 32  # seconds
HEART_RATE = 60  # beats per minute
BASELINE = 512  # Baseline ADC value (mid-range of Arduino's 0-1023 ADC)
PEAK_HEIGHT = 250  # Peak height in ADC units above baseline (VERY prominent)

# Calculate total sample count and time array
sample_count = int(SAMPLE_RATE * DURATION)
time_ms = np.arange(sample_count) * (1000 / SAMPLE_RATE)  # in milliseconds

# Create an array with the baseline value
ecg_values = np.ones(sample_count) * BASELINE

# Calculate the interval between heartbeats in samples
beat_interval_ms = 60000 / HEART_RATE  # in milliseconds
beat_interval_samples = int(beat_interval_ms * SAMPLE_RATE / 1000)

# First peak will be positioned after some initial delay
initial_delay = int(SAMPLE_RATE * 0.5)  # 0.5 seconds initial delay

# Add peaks at regular intervals with a known heart rate
for i in range(int(DURATION * HEART_RATE / 60)):
    peak_index = initial_delay + i * beat_interval_samples
    if peak_index >= sample_count:
        break
        
    # Create a very simple spike pattern - just the R peak with minimal other features
    # This maximizes the chance of correct detection
    
    # R wave (large positive spike)
    r_up_start = max(0, peak_index - 7)
    r_peak = peak_index
    r_down_end = min(sample_count, peak_index + 7)
    
    # R wave upstroke
    if r_peak > r_up_start and r_peak < sample_count:
        r_up_range = r_peak - r_up_start
        if r_up_range > 0:
            r_up_values = np.linspace(BASELINE, BASELINE + PEAK_HEIGHT, r_up_range)
            ecg_values[r_up_start:r_peak] = r_up_values
    
    # R wave downstroke
    if r_down_end > r_peak and r_peak < sample_count:
        r_down_range = r_down_end - r_peak
        if r_down_range > 0:
            r_down_values = np.linspace(BASELINE + PEAK_HEIGHT, BASELINE, r_down_range)
            ecg_values[r_peak:r_down_end] = r_down_values
    
    # Add a small T wave - just enough to look realistic but not confuse detection
    t_start = min(sample_count - 1, peak_index + 30)
    t_peak = min(sample_count - 1, peak_index + 40)
    t_end = min(sample_count - 1, peak_index + 50)
    
    # T wave upstroke
    if t_peak > t_start:
        t_up_range = t_peak - t_start
        if t_up_range > 0:
            t_up_values = np.linspace(BASELINE, BASELINE + 40, t_up_range)
            ecg_values[t_start:t_peak] = t_up_values
    
    # T wave downstroke
    if t_end > t_peak:
        t_down_range = t_end - t_peak
        if t_down_range > 0:
            t_down_values = np.linspace(BASELINE + 40, BASELINE, t_down_range)
            ecg_values[t_peak:t_end] = t_down_values

# Create DataFrame for the synthetic data
synthetic_data = pd.DataFrame({
    'time': time_ms,
    'value': ecg_values
})

# Save to CSV
synthetic_data.to_csv('sample_ecg_final.csv', index=False)
print(f"Created synthetic ECG with:")
print(f"- Heart rate: {HEART_RATE} BPM")
print(f"- Duration: {DURATION} seconds")
print(f"- Sample rate: {SAMPLE_RATE} Hz")
print(f"- Total samples: {sample_count}")
print(f"- Baseline: {BASELINE} ADC units")
print(f"- Peak height: {BASELINE + PEAK_HEIGHT} ADC units (very prominent for detection)")
print(f"Saved to sample_ecg_final.csv")

# Plot the synthetic ECG
plt.figure(figsize=(15, 5))
plt.plot(time_ms/1000, ecg_values)  # Convert to seconds for x-axis
plt.title(f'Synthetic ECG Signal with {HEART_RATE} BPM Heart Rate (optimized for detection)')
plt.xlabel('Time (seconds)')
plt.ylabel('ADC Value')
plt.ylim(400, 800)  # Match the application's display range, expanded for taller peaks
plt.grid(True)

# Mark the peaks for clarity
peaks = [initial_delay + i * beat_interval_samples for i in range(int(DURATION * HEART_RATE / 60)) 
         if initial_delay + i * beat_interval_samples < sample_count]
plt.plot([time_ms[p]/1000 for p in peaks], [ecg_values[p] for p in peaks], 'ro')

plt.savefig('synthetic_ecg_final.png')
print("Plot saved to synthetic_ecg_final.png") 