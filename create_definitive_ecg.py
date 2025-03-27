import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create an extremely simple ECG signal for unmistakable heart rate detection
# This will have a very slow heart rate of 40 BPM (1.5 seconds between peaks)

# Constants
SAMPLE_RATE = 100  # Hz (samples per second) - reduced for clarity
DURATION = 32  # seconds
HEART_RATE = 40  # beats per minute - very slow for clear detection
BASELINE = 512  # Baseline ADC value
PEAK_HEIGHT = 400  # Extremely tall peaks

# Calculate total sample count and time array
sample_count = int(SAMPLE_RATE * DURATION)
time_ms = np.arange(sample_count) * (1000 / SAMPLE_RATE)  # in milliseconds

# Create an array with the baseline value
ecg_values = np.ones(sample_count) * BASELINE

# Calculate the interval between heartbeats in samples
beat_interval_ms = 60000 / HEART_RATE  # in milliseconds
beat_interval_samples = int(beat_interval_ms * SAMPLE_RATE / 1000)

# Add peaks at regular intervals
for i in range(int(DURATION * HEART_RATE / 60)):
    # Calculate peak position
    peak_index = i * beat_interval_samples
    
    # Skip if beyond array bounds
    if peak_index >= sample_count:
        break
    
    # Create very sharp, unmistakable spike
    spike_width = 3  # Very narrow spike
    
    # Set the peak value
    if peak_index < sample_count:
        ecg_values[peak_index] = BASELINE + PEAK_HEIGHT
    
    # Taper down on both sides
    for j in range(1, spike_width + 1):
        # Left side of peak
        if peak_index - j >= 0:
            ecg_values[peak_index - j] = BASELINE + PEAK_HEIGHT * (1 - j/spike_width)
        
        # Right side of peak
        if peak_index + j < sample_count:
            ecg_values[peak_index + j] = BASELINE + PEAK_HEIGHT * (1 - j/spike_width)
    
    # Ensure we have a completely flat baseline between peaks
    start_flat = min(sample_count, peak_index + spike_width + 1)
    end_flat = min(sample_count, (i + 1) * beat_interval_samples - spike_width - 1)
    if end_flat > start_flat:
        ecg_values[start_flat:end_flat] = BASELINE

# Create DataFrame for the synthetic data
synthetic_data = pd.DataFrame({
    'time': time_ms,
    'value': ecg_values
})

# Save to CSV
synthetic_data.to_csv('sample_ecg_definitive.csv', index=False)
print(f"Created definitive ECG with:")
print(f"- Heart rate: {HEART_RATE} BPM")
print(f"- Duration: {DURATION} seconds")
print(f"- Sample rate: {SAMPLE_RATE} Hz")
print(f"- Total samples: {sample_count}")
print(f"- Peak height: {BASELINE + PEAK_HEIGHT} ADC units")
print(f"- Time between peaks: {beat_interval_ms} ms")
print(f"Saved to sample_ecg_definitive.csv")

# Plot the synthetic ECG
plt.figure(figsize=(15, 5))
plt.plot(time_ms/1000, ecg_values)  # Convert to seconds for x-axis
plt.title(f'Definitive ECG Signal with {HEART_RATE} BPM Heart Rate')
plt.xlabel('Time (seconds)')
plt.ylabel('ADC Value')
plt.ylim(400, 950)  # Set y-axis range to show peaks clearly
plt.grid(True)

# Mark the peaks for clarity
peaks = [i * beat_interval_samples for i in range(int(DURATION * HEART_RATE / 60)) 
         if i * beat_interval_samples < sample_count]
plt.plot([time_ms[p]/1000 for p in peaks], [ecg_values[p] for p in peaks], 'ro')

plt.savefig('synthetic_ecg_definitive.png')
print("Plot saved to synthetic_ecg_definitive.png") 