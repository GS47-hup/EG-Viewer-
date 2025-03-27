import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# This script generates simulated abnormal ECG signals for testing classification

def create_ecg_directory():
    """Create directory for abnormal ECG data if it doesn't exist"""
    if not os.path.exists("abnormal_ecg_samples"):
        os.makedirs("abnormal_ecg_samples")
        print("Created directory: abnormal_ecg_samples")

# Constants
SAMPLE_RATE = 250  # Hz (samples per second)
DURATION = 32  # seconds
BASELINE_MV = 0.0  # Baseline in millivolts
PEAK_HEIGHT_MV = 1.0  # Normal R-wave peak height in millivolts

def generate_normal_heartbeat(start_idx, amplitude=1.0, noise=0.0):
    """Generate a normal QRS complex with P and T waves"""
    # Total heartbeat length is around 150 samples (600ms)
    beat_length = 150
    beat = np.zeros(beat_length)
    
    # P wave (small bump before QRS)
    p_start = 10
    p_width = 20
    for i in range(p_width):
        idx = p_start + i
        if idx < beat_length:
            beat[idx] = 0.2 * amplitude * np.sin(np.pi * i / p_width)
    
    # QRS complex
    q_start = 50
    # Q wave (small negative deflection)
    for i in range(5):
        idx = q_start + i
        if idx < beat_length:
            beat[idx] = -0.2 * amplitude * (i / 5)
    
    # R wave (large positive spike)
    r_start = q_start + 5
    for i in range(10):
        idx = r_start + i
        if idx < beat_length:
            if i < 5:
                beat[idx] = amplitude * (i / 5)
            else:
                beat[idx] = amplitude * (1 - (i - 5) / 5)
    
    # S wave (negative deflection after R)
    s_start = r_start + 10
    for i in range(8):
        idx = s_start + i
        if idx < beat_length:
            if i < 4:
                beat[idx] = -0.3 * amplitude * (i / 4)
            else:
                beat[idx] = -0.3 * amplitude * (1 - (i - 4) / 4)
    
    # T wave (positive bump after QRS)
    t_start = 100
    t_width = 30
    for i in range(t_width):
        idx = t_start + i
        if idx < beat_length:
            beat[idx] = 0.3 * amplitude * np.sin(np.pi * i / t_width)
    
    # Add random noise
    if noise > 0:
        beat += np.random.normal(0, noise, beat_length)
    
    return beat

def create_tachycardia_ecg():
    """Create ECG with tachycardia (fast heart rate >100 BPM)"""
    print("Generating tachycardia ECG (abnormally fast heart rate)...")
    
    # Heart rate of 130 BPM
    HEART_RATE = 130
    sample_count = int(SAMPLE_RATE * DURATION)
    time_ms = np.arange(sample_count) * (1000 / SAMPLE_RATE)
    ecg_values_mv = np.zeros(sample_count)
    
    # Calculate beat interval
    beat_interval_ms = 60000 / HEART_RATE
    beat_interval_samples = int(beat_interval_ms * SAMPLE_RATE / 1000)
    
    # Initial delay
    initial_delay = int(SAMPLE_RATE * 0.5)
    
    # Generate individual heartbeats at a fast rate
    for i in range(int(DURATION * HEART_RATE / 60) + 1):
        beat_start = initial_delay + i * beat_interval_samples
        if beat_start >= sample_count:
            break
            
        # Get normal heartbeat pattern
        beat = generate_normal_heartbeat(beat_start, amplitude=0.9, noise=0.01)
        
        # Place beat in the ECG
        for j in range(len(beat)):
            idx = beat_start + j
            if idx < sample_count:
                ecg_values_mv[idx] = beat[j]
    
    # Create DataFrame
    tachycardia_data = pd.DataFrame({
        'time': time_ms,
        'value': ecg_values_mv
    })
    
    # Save to CSV
    output_file = "abnormal_ecg_samples/tachycardia_ecg.csv"
    tachycardia_data.to_csv(output_file, index=False)
    print(f"Saved tachycardia ECG to {output_file}")
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(time_ms/1000, ecg_values_mv)
    plt.title(f'Tachycardia ECG - {HEART_RATE} BPM (Abnormally Fast)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    plt.grid(True)
    plt.savefig("abnormal_ecg_samples/tachycardia_ecg.png")
    
    return output_file

def create_bradycardia_ecg():
    """Create ECG with bradycardia (slow heart rate <60 BPM)"""
    print("Generating bradycardia ECG (abnormally slow heart rate)...")
    
    # Heart rate of 45 BPM
    HEART_RATE = 45
    sample_count = int(SAMPLE_RATE * DURATION)
    time_ms = np.arange(sample_count) * (1000 / SAMPLE_RATE)
    ecg_values_mv = np.zeros(sample_count)
    
    # Calculate beat interval
    beat_interval_ms = 60000 / HEART_RATE
    beat_interval_samples = int(beat_interval_ms * SAMPLE_RATE / 1000)
    
    # Initial delay
    initial_delay = int(SAMPLE_RATE * 0.5)
    
    # Generate individual heartbeats at a slow rate
    for i in range(int(DURATION * HEART_RATE / 60) + 1):
        beat_start = initial_delay + i * beat_interval_samples
        if beat_start >= sample_count:
            break
            
        # Get normal heartbeat pattern
        beat = generate_normal_heartbeat(beat_start, amplitude=1.1, noise=0.01)
        
        # Place beat in the ECG
        for j in range(len(beat)):
            idx = beat_start + j
            if idx < sample_count:
                ecg_values_mv[idx] = beat[j]
    
    # Create DataFrame
    bradycardia_data = pd.DataFrame({
        'time': time_ms,
        'value': ecg_values_mv
    })
    
    # Save to CSV
    output_file = "abnormal_ecg_samples/bradycardia_ecg.csv"
    bradycardia_data.to_csv(output_file, index=False)
    print(f"Saved bradycardia ECG to {output_file}")
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(time_ms/1000, ecg_values_mv)
    plt.title(f'Bradycardia ECG - {HEART_RATE} BPM (Abnormally Slow)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    plt.grid(True)
    plt.savefig("abnormal_ecg_samples/bradycardia_ecg.png")
    
    return output_file

def create_arrhythmia_ecg():
    """Create ECG with arrhythmia (irregular heartbeat pattern)"""
    print("Generating arrhythmia ECG (irregular heartbeat)...")
    
    # Base heart rate around 75 BPM but with variability
    BASE_HEART_RATE = 75
    sample_count = int(SAMPLE_RATE * DURATION)
    time_ms = np.arange(sample_count) * (1000 / SAMPLE_RATE)
    ecg_values_mv = np.zeros(sample_count)
    
    # Base interval
    base_interval_ms = 60000 / BASE_HEART_RATE
    base_interval_samples = int(base_interval_ms * SAMPLE_RATE / 1000)
    
    # Initial delay
    beat_time = int(SAMPLE_RATE * 0.5)
    
    # Generate irregular heartbeats 
    while beat_time < sample_count:
        # Normal beat
        beat = generate_normal_heartbeat(beat_time, amplitude=1.0, noise=0.01)
        
        # Place beat in the ECG
        for j in range(len(beat)):
            idx = beat_time + j
            if idx < sample_count:
                ecg_values_mv[idx] = beat[j]
        
        # Determine next beat timing with irregularity
        if random.random() < 0.3:  # 30% chance of arrhythmia
            # Either a very short interval (early beat)
            if random.random() < 0.5:
                interval = int(base_interval_samples * 0.6)  # Premature beat
            else:
                interval = int(base_interval_samples * 1.5)  # Delayed beat
        else:
            # Normal interval with small variability
            variation = random.uniform(0.9, 1.1)
            interval = int(base_interval_samples * variation)
        
        beat_time += interval
    
    # Create DataFrame
    arrhythmia_data = pd.DataFrame({
        'time': time_ms,
        'value': ecg_values_mv
    })
    
    # Save to CSV
    output_file = "abnormal_ecg_samples/arrhythmia_ecg.csv"
    arrhythmia_data.to_csv(output_file, index=False)
    print(f"Saved arrhythmia ECG to {output_file}")
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(time_ms/1000, ecg_values_mv)
    plt.title('Arrhythmia ECG - Irregular Heartbeat Pattern')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    plt.grid(True)
    plt.savefig("abnormal_ecg_samples/arrhythmia_ecg.png")
    
    return output_file

def create_st_elevation_ecg():
    """Create ECG with ST segment elevation (potential myocardial infarction sign)"""
    print("Generating ECG with ST segment elevation...")
    
    # Normal heart rate of 75 BPM
    HEART_RATE = 75
    sample_count = int(SAMPLE_RATE * DURATION)
    time_ms = np.arange(sample_count) * (1000 / SAMPLE_RATE)
    ecg_values_mv = np.zeros(sample_count)
    
    # Calculate beat interval
    beat_interval_ms = 60000 / HEART_RATE
    beat_interval_samples = int(beat_interval_ms * SAMPLE_RATE / 1000)
    
    # Initial delay
    initial_delay = int(SAMPLE_RATE * 0.5)
    
    # Generate heartbeats with ST elevation
    for i in range(int(DURATION * HEART_RATE / 60) + 1):
        beat_start = initial_delay + i * beat_interval_samples
        if beat_start >= sample_count:
            break
            
        # Get base heartbeat
        beat = generate_normal_heartbeat(beat_start, amplitude=1.0, noise=0.01)
        
        # Modify ST segment (elevated)
        st_start = beat_start + 80  # Position after S wave
        st_end = beat_start + 100   # Before T wave
        
        # Modify beat to have ST elevation
        for j in range(len(beat)):
            idx = beat_start + j
            if idx < sample_count:
                # Add ST elevation of 0.2mV
                if beat_start + 80 <= idx <= beat_start + 100:
                    beat[j] += 0.2
                ecg_values_mv[idx] = beat[j]
    
    # Create DataFrame
    st_data = pd.DataFrame({
        'time': time_ms,
        'value': ecg_values_mv
    })
    
    # Save to CSV
    output_file = "abnormal_ecg_samples/st_elevation_ecg.csv"
    st_data.to_csv(output_file, index=False)
    print(f"Saved ST elevation ECG to {output_file}")
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(time_ms/1000, ecg_values_mv)
    plt.title('ECG with ST Segment Elevation')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    plt.grid(True)
    plt.savefig("abnormal_ecg_samples/st_elevation_ecg.png")
    
    return output_file

def create_afib_ecg():
    """Create ECG with atrial fibrillation (irregular rhythm, absent P waves)"""
    print("Generating atrial fibrillation ECG...")
    
    # Irregular heart rate around 100 BPM
    BASE_HEART_RATE = 100
    sample_count = int(SAMPLE_RATE * DURATION)
    time_ms = np.arange(sample_count) * (1000 / SAMPLE_RATE)
    ecg_values_mv = np.zeros(sample_count)
    
    # Base interval with irregularity
    base_interval_ms = 60000 / BASE_HEART_RATE
    base_interval_samples = int(base_interval_ms * SAMPLE_RATE / 1000)
    
    # Generate baseline noise (to simulate fibrillatory waves)
    baseline_noise = np.random.normal(0, 0.05, sample_count)
    ecg_values_mv += baseline_noise
    
    # Initial delay
    beat_time = int(SAMPLE_RATE * 0.5)
    
    # Generate irregular QRS complexes without P waves
    while beat_time < sample_count:
        # Modified heartbeat without P wave for AFib
        beat_length = 150
        beat = np.zeros(beat_length)
        
        # Skip P wave, only create QRS and T
        # QRS complex
        q_start = 30
        # Q wave
        for i in range(5):
            idx = q_start + i
            if idx < beat_length:
                beat[idx] = -0.2 * (i / 5)
        
        # R wave
        r_start = q_start + 5
        for i in range(10):
            idx = r_start + i
            if idx < beat_length:
                if i < 5:
                    beat[idx] = 1.0 * (i / 5)
                else:
                    beat[idx] = 1.0 * (1 - (i - 5) / 5)
        
        # S wave
        s_start = r_start + 10
        for i in range(8):
            idx = s_start + i
            if idx < beat_length:
                if i < 4:
                    beat[idx] = -0.3 * (i / 4)
                else:
                    beat[idx] = -0.3 * (1 - (i - 4) / 4)
        
        # T wave
        t_start = 80
        t_width = 30
        for i in range(t_width):
            idx = t_start + i
            if idx < beat_length:
                beat[idx] = 0.3 * np.sin(np.pi * i / t_width)
        
        # Place beat in the ECG
        for j in range(len(beat)):
            idx = beat_time + j
            if idx < sample_count:
                ecg_values_mv[idx] += beat[j]  # Add to existing baseline noise
        
        # Highly irregular intervals characteristic of AFib
        variation = random.uniform(0.7, 1.3)
        interval = int(base_interval_samples * variation)
        beat_time += interval
    
    # Create DataFrame
    afib_data = pd.DataFrame({
        'time': time_ms,
        'value': ecg_values_mv
    })
    
    # Save to CSV
    output_file = "abnormal_ecg_samples/afib_ecg.csv"
    afib_data.to_csv(output_file, index=False)
    print(f"Saved atrial fibrillation ECG to {output_file}")
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(time_ms/1000, ecg_values_mv)
    plt.title('Atrial Fibrillation ECG - Irregular Rhythm, Absent P Waves')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    plt.grid(True)
    plt.savefig("abnormal_ecg_samples/afib_ecg.png")
    
    return output_file

if __name__ == "__main__":
    create_ecg_directory()
    
    # Generate all abnormal ECG types
    create_tachycardia_ecg()
    create_bradycardia_ecg()
    create_arrhythmia_ecg()
    create_st_elevation_ecg()
    create_afib_ecg()
    
    print("\nAll abnormal ECG samples created successfully.")
    print("These can be used to test your ECG classification system.")
    print("The files are saved in the abnormal_ecg_samples directory.") 