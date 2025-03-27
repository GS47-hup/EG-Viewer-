import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# This script generates simulated normal ECG signals for testing classification

def create_ecg_directory():
    """Create directory for normal ECG data if it doesn't exist"""
    if not os.path.exists("normal_ecg_samples"):
        os.makedirs("normal_ecg_samples")
        print("Created directory: normal_ecg_samples")

# Constants
SAMPLE_RATE = 250  # Hz (samples per second)
DURATION = 32  # seconds
BASELINE_MV = 0.0  # Baseline in millivolts
PEAK_HEIGHT_MV = 1.0  # Normal R-wave peak height in millivolts

def generate_normal_ecg(heart_rate=70, noise_level=0.01, amplitude_variation=0.05):
    """
    Generate a normal ECG signal with specified parameters
    
    Parameters:
    - heart_rate: BPM (beats per minute)
    - noise_level: Amount of background noise to add (standard deviation)
    - amplitude_variation: Variation in R peak amplitude (percentage)
    """
    print(f"Generating normal ECG with heart rate of {heart_rate} BPM...")
    
    sample_count = int(SAMPLE_RATE * DURATION)
    time_ms = np.arange(sample_count) * (1000 / SAMPLE_RATE)
    ecg_values_mv = np.zeros(sample_count)
    
    # Calculate beat interval
    beat_interval_ms = 60000 / heart_rate
    beat_interval_samples = int(beat_interval_ms * SAMPLE_RATE / 1000)
    
    # Initial delay
    initial_delay = int(SAMPLE_RATE * 0.5)
    
    # Generate individual heartbeats
    for i in range(int(DURATION * heart_rate / 60) + 1):
        beat_start = initial_delay + i * beat_interval_samples
        if beat_start >= sample_count:
            break
            
        # Get normal heartbeat pattern with slight variations
        heartbeat = create_normal_heartbeat(beat_start, amplitude_variation)
        
        # Place beat in the ECG
        for j in range(len(heartbeat)):
            idx = beat_start + j
            if idx < sample_count:
                ecg_values_mv[idx] = heartbeat[j]
    
    # Add background noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, sample_count)
        ecg_values_mv += noise
    
    return time_ms, ecg_values_mv

def create_normal_heartbeat(start_idx, amplitude_variation=0.05):
    """Create a realistic normal heartbeat with P, QRS, and T waves"""
    # Total heartbeat length around 200 samples (800ms)
    beat_length = 200
    beat = np.zeros(beat_length)
    
    # Add very slight random variation to amplitude
    amplitude = 1.0 + np.random.uniform(-amplitude_variation, amplitude_variation)
    
    # P wave (atrial depolarization)
    p_start = 10
    p_duration = 25
    for i in range(p_duration):
        idx = p_start + i
        if idx < beat_length:
            beat[idx] = 0.25 * amplitude * np.sin(np.pi * i / p_duration)
    
    # PR segment (conduction delay)
    # Just baseline, nothing to add
    
    # QRS complex (ventricular depolarization)
    q_start = p_start + p_duration + 10
    
    # Q wave (small downward deflection)
    q_duration = 5
    for i in range(q_duration):
        idx = q_start + i
        if idx < beat_length:
            beat[idx] = -0.2 * amplitude * (i / q_duration)
    
    # R wave (large upward spike)
    r_start = q_start + q_duration
    r_upstroke = 10
    r_downstroke = 10
    for i in range(r_upstroke):
        idx = r_start + i
        if idx < beat_length:
            beat[idx] = amplitude * (i / r_upstroke)
    
    for i in range(r_downstroke):
        idx = r_start + r_upstroke + i
        if idx < beat_length:
            beat[idx] = amplitude - amplitude * (i / r_downstroke)
    
    # S wave (downward deflection after R)
    s_start = r_start + r_upstroke + r_downstroke
    s_duration = 10
    for i in range(s_duration):
        idx = s_start + i
        if idx < beat_length:
            if i < s_duration/2:
                beat[idx] = -0.3 * amplitude * (i / (s_duration/2))
            else:
                beat[idx] = -0.3 * amplitude * (1 - (i - s_duration/2) / (s_duration/2))
    
    # ST segment (early repolarization)
    st_start = s_start + s_duration
    st_duration = 20
    for i in range(st_duration):
        idx = st_start + i
        if idx < beat_length:
            # Slight upward slope from S to T
            beat[idx] = -0.05 * amplitude * (1 - i / st_duration)
    
    # T wave (ventricular repolarization)
    t_start = st_start + st_duration
    t_duration = 40
    for i in range(t_duration):
        idx = t_start + i
        if idx < beat_length:
            beat[idx] = 0.3 * amplitude * np.sin(np.pi * i / t_duration)
    
    return beat

def save_normal_ecg(hr, prefix="normal"):
    """Generate, save and plot a normal ECG with specified heart rate"""
    time_ms, ecg_mv = generate_normal_ecg(heart_rate=hr)
    
    # Create DataFrame
    ecg_data = pd.DataFrame({
        'time': time_ms,
        'value': ecg_mv
    })
    
    # Save to CSV
    filename = f"{prefix}_{hr}bpm"
    output_file = f"normal_ecg_samples/{filename}.csv"
    ecg_data.to_csv(output_file, index=False)
    print(f"Saved normal ECG ({hr} BPM) to {output_file}")
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(time_ms/1000, ecg_mv)  # Convert ms to seconds for x-axis
    plt.title(f'Normal ECG - Heart Rate: {hr} BPM')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    plt.grid(True)
    plt.savefig(f"normal_ecg_samples/{filename}.png")
    
    # Generate a zoomed-in view of a few beats for detailed inspection
    plt.figure(figsize=(15, 5))
    # Show just 3 seconds of data
    start_idx = int(SAMPLE_RATE * 3)
    end_idx = int(SAMPLE_RATE * 6)
    plt.plot(time_ms[start_idx:end_idx]/1000, ecg_mv[start_idx:end_idx])
    plt.title(f'Normal ECG Detail View - Heart Rate: {hr} BPM')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    plt.grid(True)
    plt.savefig(f"normal_ecg_samples/{filename}_detail.png")
    
    return output_file

def create_normal_ecg_with_variations():
    """Create normal ECG samples with different heart rates and variations"""
    # Various normal heart rates
    heart_rates = [60, 70, 75, 85, 90]
    
    files = []
    for hr in heart_rates:
        output_file = save_normal_ecg(hr)
        files.append(output_file)
    
    # Create a normal ECG with slightly more noise (but still normal)
    time_ms, ecg_mv = generate_normal_ecg(heart_rate=75, noise_level=0.03)
    
    # Create DataFrame
    ecg_data = pd.DataFrame({
        'time': time_ms,
        'value': ecg_mv
    })
    
    # Save to CSV
    output_file = "normal_ecg_samples/normal_75bpm_noise.csv"
    ecg_data.to_csv(output_file, index=False)
    print(f"Saved normal ECG with noise to {output_file}")
    files.append(output_file)
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(time_ms/1000, ecg_mv)
    plt.title('Normal ECG with Noise - Heart Rate: 75 BPM')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    plt.grid(True)
    plt.savefig("normal_ecg_samples/normal_75bpm_noise.png")
    
    # Create a normal ECG with slight respiration effects (but still normal)
    time_ms, ecg_mv = generate_normal_ecg(heart_rate=70)
    
    # Add subtle respiratory modulation (amplitude changes with breathing)
    resp_rate = 15  # breaths per minute
    resp_cycle_samples = int(SAMPLE_RATE * 60 / resp_rate)
    resp_effect = 0.1 * np.sin(2 * np.pi * np.arange(len(ecg_mv)) / resp_cycle_samples)
    ecg_mv = ecg_mv * (1 + resp_effect)
    
    # Create DataFrame
    ecg_data = pd.DataFrame({
        'time': time_ms,
        'value': ecg_mv
    })
    
    # Save to CSV
    output_file = "normal_ecg_samples/normal_70bpm_respiration.csv"
    ecg_data.to_csv(output_file, index=False)
    print(f"Saved normal ECG with respiratory modulation to {output_file}")
    files.append(output_file)
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(time_ms/1000, ecg_mv)
    plt.title('Normal ECG with Respiratory Modulation - Heart Rate: 70 BPM')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    plt.grid(True)
    plt.savefig("normal_ecg_samples/normal_70bpm_respiration.png")
    
    return files

if __name__ == "__main__":
    create_ecg_directory()
    files = create_normal_ecg_with_variations()
    
    print("\nAll normal ECG samples created successfully.")
    print("These can be used as reference data for your ECG classification system.")
    print("The files are saved in the normal_ecg_samples directory.") 