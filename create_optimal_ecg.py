import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic ECG signal with optimized parameters for heart rate detection
# This will have a heart rate of 80 BPM to help with peak detection

# Constants
SAMPLE_RATE = 250  # Hz (samples per second)
DURATION = 32  # seconds
HEART_RATE = 80  # beats per minute
BASELINE = 512  # Baseline ADC value (mid-range of Arduino's 0-1023 ADC)
PEAK_HEIGHT = 300  # Very large peak height for clear detection

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
        
    # Create a very simple spike pattern with extremely prominent peaks
    # and minimal other features - optimized for detection
    
    # R wave (huge positive spike)
    r_up_start = max(0, peak_index - 5)
    r_peak = peak_index
    r_down_end = min(sample_count, peak_index + 5)
    
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
    
    # Flat between beats - no T wave or other features to avoid confusion
    # Just maintain baseline between spikes

# Create DataFrame for the synthetic data
synthetic_data = pd.DataFrame({
    'time': time_ms,
    'value': ecg_values
})

# Save to CSV
synthetic_data.to_csv('sample_ecg_optimal.csv', index=False)
print(f"Created optimal ECG with:")
print(f"- Heart rate: {HEART_RATE} BPM")
print(f"- Duration: {DURATION} seconds")
print(f"- Sample rate: {SAMPLE_RATE} Hz")
print(f"- Total samples: {sample_count}")
print(f"- Baseline: {BASELINE} ADC units")
print(f"- Peak height: {BASELINE + PEAK_HEIGHT} ADC units (extremely prominent)")
print(f"Saved to sample_ecg_optimal.csv")

# Plot the synthetic ECG
plt.figure(figsize=(15, 5))
plt.plot(time_ms/1000, ecg_values)  # Convert to seconds for x-axis
plt.title(f'Optimal ECG Signal with {HEART_RATE} BPM Heart Rate')
plt.xlabel('Time (seconds)')
plt.ylabel('ADC Value')
plt.ylim(400, 850)  # Set y-axis range to show peaks clearly
plt.grid(True)

# Mark the peaks for clarity
peaks = [initial_delay + i * beat_interval_samples for i in range(int(DURATION * HEART_RATE / 60)) 
         if initial_delay + i * beat_interval_samples < sample_count]
plt.plot([time_ms[p]/1000 for p in peaks], [ecg_values[p] for p in peaks], 'ro')

plt.savefig('synthetic_ecg_optimal.png')
print("Plot saved to synthetic_ecg_optimal.png") 