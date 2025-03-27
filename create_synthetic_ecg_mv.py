import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic ECG signal with very precise timing and clear peaks
# This will have a known, fixed heart rate of 60 BPM, in millivolts

# Constants
SAMPLE_RATE = 250  # Hz (samples per second)
DURATION = 32  # seconds
HEART_RATE = 60  # beats per minute
BASELINE_MV = 0.0  # Baseline in millivolts
PEAK_HEIGHT_MV = 1.0  # Peak height in millivolts (above baseline)

# Conversion factor from millivolts to ADC range (for Arduino 10-bit ADC)
# For a 3.3V reference voltage, 1023 ADC units = 3.3V
# So 1mV = (1023 / 3300) ADC units
MV_TO_ADC = 1023 / 3300  # ADC units per millivolt

# Calculate total sample count and time array
sample_count = int(SAMPLE_RATE * DURATION)
time_ms = np.arange(sample_count) * (1000 / SAMPLE_RATE)  # in milliseconds

# Create an array with the baseline value
ecg_values_mv = np.ones(sample_count) * BASELINE_MV

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
        
    # Create a QRS complex (typical ECG waveform) around each peak
    
    # Q wave (small negative deflection)
    q_width = 5
    q_start = max(0, peak_index - 10)
    q_end = max(0, peak_index - 5)
    if q_end > q_start and q_end < sample_count:
        q_range = q_end - q_start
        q_values = np.linspace(BASELINE_MV, BASELINE_MV - 0.1, q_range)
        ecg_values_mv[q_start:q_end] = q_values
    
    # R wave (large positive spike)
    r_up_width = 5
    r_down_width = 5
    r_up_start = max(0, peak_index - 5)
    r_peak = peak_index
    r_down_end = min(sample_count, peak_index + 5)
    
    # R wave upstroke
    if r_peak > r_up_start and r_peak < sample_count:
        r_up_range = r_peak - r_up_start
        if r_up_range > 0:
            r_up_values = np.linspace(BASELINE_MV - 0.1, BASELINE_MV + PEAK_HEIGHT_MV, r_up_range)
            ecg_values_mv[r_up_start:r_peak] = r_up_values
    
    # R wave downstroke
    if r_down_end > r_peak and r_peak < sample_count:
        r_down_range = r_down_end - r_peak
        if r_down_range > 0:
            r_down_values = np.linspace(BASELINE_MV + PEAK_HEIGHT_MV, BASELINE_MV, r_down_range)
            ecg_values_mv[r_peak:r_down_end] = r_down_values
    
    # S wave (small negative deflection after R)
    s_width = 10
    s_recovery_width = 5
    s_start = min(sample_count - 1, peak_index + 5)
    s_end = min(sample_count - 1, peak_index + 15)
    s_recovery_end = min(sample_count - 1, s_end + 5)
    
    # S wave downstroke
    if s_end > s_start:
        s_range = s_end - s_start
        if s_range > 0:
            s_values = np.linspace(BASELINE_MV, BASELINE_MV - 0.05, s_range)
            ecg_values_mv[s_start:s_end] = s_values
    
    # S wave recovery
    if s_recovery_end > s_end:
        s_recovery_range = s_recovery_end - s_end
        if s_recovery_range > 0:
            s_recovery_values = np.linspace(BASELINE_MV - 0.05, BASELINE_MV, s_recovery_range)
            ecg_values_mv[s_end:s_recovery_end] = s_recovery_values
    
    # T wave (smaller positive bump after QRS)
    t_up_width = 10
    t_down_width = 10
    t_start = min(sample_count - 1, peak_index + 20)
    t_peak = min(sample_count - 1, peak_index + 30)
    t_end = min(sample_count - 1, peak_index + 40)
    
    # T wave upstroke
    if t_peak > t_start:
        t_up_range = t_peak - t_start
        if t_up_range > 0:
            t_up_values = np.linspace(BASELINE_MV, BASELINE_MV + 0.2, t_up_range)
            ecg_values_mv[t_start:t_peak] = t_up_values
    
    # T wave downstroke
    if t_end > t_peak:
        t_down_range = t_end - t_peak
        if t_down_range > 0:
            t_down_values = np.linspace(BASELINE_MV + 0.2, BASELINE_MV, t_down_range)
            ecg_values_mv[t_peak:t_end] = t_down_values

# Create DataFrame for the synthetic data
synthetic_data = pd.DataFrame({
    'time': time_ms,
    'value': ecg_values_mv
})

# Save to CSV
synthetic_data.to_csv('sample_ecg_mv.csv', index=False)
print(f"Created synthetic ECG with:")
print(f"- Heart rate: {HEART_RATE} BPM")
print(f"- Duration: {DURATION} seconds")
print(f"- Sample rate: {SAMPLE_RATE} Hz")
print(f"- Total samples: {sample_count}")
print(f"- Values in millivolts: Baseline {BASELINE_MV}mV, Peak {BASELINE_MV + PEAK_HEIGHT_MV}mV")
print(f"Saved to sample_ecg_mv.csv")

# Plot the synthetic ECG
plt.figure(figsize=(15, 5))
plt.plot(time_ms/1000, ecg_values_mv)  # Convert to seconds for x-axis
plt.title(f'Synthetic ECG Signal with {HEART_RATE} BPM Heart Rate (in millivolts)')
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (mV)')
plt.grid(True)

# Mark the peaks for clarity
peaks = [initial_delay + i * beat_interval_samples for i in range(int(DURATION * HEART_RATE / 60)) 
         if initial_delay + i * beat_interval_samples < sample_count]
plt.plot([time_ms[p]/1000 for p in peaks], [ecg_values_mv[p] for p in peaks], 'ro')

plt.savefig('synthetic_ecg_mv.png')
print("Plot saved to synthetic_ecg_mv.png") 