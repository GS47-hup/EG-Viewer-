#!/usr/bin/env python
"""
Sample Data Generator - Creates synthetic ECG data for the standalone ECG simulator
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal

def generate_normal_ecg(duration=10, fs=250, heart_rate=70, noise_level=0.05):
    """
    Generate a normal ECG signal with consistent heart rate and regular R-R intervals
    
    Parameters:
    - duration: length of the signal in seconds
    - fs: sampling frequency in Hz
    - heart_rate: heart rate in beats per minute
    - noise_level: standard deviation of noise
    
    Returns:
    - ecg_signal: array of ECG signal values
    """
    # Calculate number of samples
    num_samples = int(duration * fs)
    
    # Calculate time axis
    t = np.arange(num_samples) / fs
    
    # Calculate R-R interval in seconds
    r_r_interval = 60 / heart_rate
    
    # Create periodic signal using sawtooth wave for basic shape
    base_signal = signal.sawtooth(2 * np.pi * t / r_r_interval, width=0.1)
    
    # Add QRS complex shape over sawtooth
    qrs_width = 0.1  # seconds
    qrs_samples = int(qrs_width * fs)
    
    # Normalize base signal to [-1, 1]
    base_signal = (base_signal - np.min(base_signal)) / (np.max(base_signal) - np.min(base_signal)) * 2 - 1
    
    # Create QRS template - a peaked wave
    qrs_template = np.zeros(qrs_samples)
    peak_location = int(qrs_samples / 3)
    for i in range(qrs_samples):
        # Gaussian-like peak for R wave
        qrs_template[i] = 2 * np.exp(-0.5 * ((i - peak_location) / (qrs_samples / 10))**2)
    
    # Apply QRS template at each R peak
    ecg_signal = np.zeros(num_samples)
    r_peaks = []
    
    # Find R peak locations based on heart rate
    for i in range(int(duration / r_r_interval) + 1):
        peak_idx = int(i * r_r_interval * fs)
        if peak_idx + qrs_samples <= num_samples:
            ecg_signal[peak_idx:peak_idx+qrs_samples] = qrs_template
            r_peaks.append(peak_idx + peak_location)
    
    # Add P and T waves
    for peak_idx in r_peaks:
        # P wave (before R peak)
        p_onset = peak_idx - int(0.2 * fs)
        p_duration = int(0.1 * fs)
        if p_onset > 0:
            p_wave = 0.3 * np.sin(np.linspace(0, np.pi, p_duration))
            p_indices = np.arange(p_onset, p_onset + p_duration)
            p_indices = p_indices[p_indices < num_samples]
            ecg_signal[p_indices] += p_wave[:len(p_indices)]
        
        # T wave (after R peak)
        t_onset = peak_idx + int(0.1 * fs)
        t_duration = int(0.2 * fs)
        if t_onset + t_duration < num_samples:
            t_wave = 0.4 * np.sin(np.linspace(0, np.pi, t_duration))
            ecg_signal[t_onset:t_onset+t_duration] += t_wave
    
    # Add slight baseline variation
    baseline = 0.1 * np.sin(2 * np.pi * 0.05 * t)
    
    # Add random noise
    noise = np.random.normal(0, noise_level, num_samples)
    
    # Combine signals
    ecg_signal = ecg_signal + baseline + noise
    
    # Normalize to a reasonable range (mimicking mV)
    ecg_signal = ecg_signal * 1.0
    
    return ecg_signal

def generate_afib_ecg(duration=10, fs=250, base_heart_rate=120, noise_level=0.05):
    """
    Generate an abnormal ECG with atrial fibrillation
    - Irregular R-R intervals
    - Absence of P waves
    - Fibrillatory waves
    
    Parameters:
    - duration: length of the signal in seconds
    - fs: sampling frequency in Hz
    - base_heart_rate: average heart rate in beats per minute
    - noise_level: standard deviation of noise
    
    Returns:
    - ecg_signal: array of ECG signal values
    """
    # Calculate number of samples
    num_samples = int(duration * fs)
    
    # Calculate time axis
    t = np.arange(num_samples) / fs
    
    # Create irregular R-R intervals
    # Base R-R interval with random variation
    mean_r_r = 60 / base_heart_rate
    r_r_intervals = []
    
    # Generate random RR intervals with variability 
    total_time = 0
    while total_time < duration:
        # High variability for AFib (±30%)
        variation = np.random.uniform(-0.3, 0.3)
        rr = mean_r_r * (1 + variation)
        r_r_intervals.append(rr)
        total_time += rr
    
    # Create QRS template for each beat
    qrs_width = 0.08  # seconds, narrower for AFib
    qrs_samples = int(qrs_width * fs)
    
    # Create QRS template - a peaked wave
    qrs_template = np.zeros(qrs_samples)
    peak_location = int(qrs_samples / 3)
    for i in range(qrs_samples):
        # Gaussian-like peak for R wave
        qrs_template[i] = 2 * np.exp(-0.5 * ((i - peak_location) / (qrs_samples / 10))**2)
    
    # Apply QRS template at each R peak
    ecg_signal = np.zeros(num_samples)
    r_peaks = []
    
    # Place QRS complexes at irregular intervals
    current_time = 0
    for rr in r_r_intervals:
        peak_idx = int(current_time * fs)
        if peak_idx + qrs_samples <= num_samples:
            ecg_signal[peak_idx:peak_idx+qrs_samples] = qrs_template
            r_peaks.append(peak_idx + peak_location)
        current_time += rr
    
    # Add T waves but no P waves (characteristic of AFib)
    for peak_idx in r_peaks:
        # T wave (after R peak)
        t_onset = peak_idx + int(0.1 * fs)
        t_duration = int(0.2 * fs)
        if t_onset + t_duration < num_samples:
            t_wave = 0.4 * np.sin(np.linspace(0, np.pi, t_duration))
            ecg_signal[t_onset:t_onset+t_duration] += t_wave
    
    # Add fibrillatory waves (chaotic baseline instead of P waves)
    # Create fast irregular oscillations
    fib_freq = np.random.uniform(8, 12)  # Frequency of fibrillatory waves (8-12 Hz)
    fib_amplitude = 0.2  # Lower amplitude than normal P waves
    fibrillatory_waves = fib_amplitude * np.sin(2 * np.pi * fib_freq * t)
    
    # Add some variation to fibrillatory waves
    fib_modulation = 0.05 * np.sin(2 * np.pi * 0.5 * t)
    fibrillatory_waves = fibrillatory_waves + fib_modulation
    
    # Baseline wander
    baseline = 0.15 * np.sin(2 * np.pi * 0.05 * t)
    
    # Add random noise
    noise = np.random.normal(0, noise_level, num_samples)
    
    # Combine signals
    ecg_signal = ecg_signal + fibrillatory_waves + baseline + noise
    
    # Normalize to a reasonable range
    ecg_signal = ecg_signal * 1.0
    
    return ecg_signal

def generate_st_elevation_ecg(duration=10, fs=250, heart_rate=70, noise_level=0.05, elevation_amount=0.3):
    """
    Generate an abnormal ECG with ST elevation (seen in myocardial infarction)
    
    Parameters:
    - duration: length of the signal in seconds
    - fs: sampling frequency in Hz
    - heart_rate: heart rate in beats per minute
    - noise_level: standard deviation of noise
    - elevation_amount: amount of ST segment elevation
    
    Returns:
    - ecg_signal: array of ECG signal values
    """
    # Start with a normal ECG
    ecg_signal = generate_normal_ecg(duration, fs, heart_rate, noise_level)
    
    # Calculate number of samples
    num_samples = len(ecg_signal)
    
    # Calculate R-R interval in seconds
    r_r_interval = 60 / heart_rate
    
    # Detect R peaks (we know they occur at regular intervals)
    r_peaks = []
    for i in range(int(duration / r_r_interval) + 1):
        peak_idx = int(i * r_r_interval * fs)
        if peak_idx < num_samples:
            # Find the actual peak within a window
            window_size = int(0.2 * fs)
            start_idx = max(0, peak_idx - window_size//2)
            end_idx = min(num_samples, peak_idx + window_size//2)
            r_peaks.append(start_idx + np.argmax(ecg_signal[start_idx:end_idx]))
    
    # Apply ST elevation after each R peak
    for peak_idx in r_peaks:
        # ST segment starts after the R peak
        st_onset = peak_idx + int(0.05 * fs)
        st_duration = int(0.15 * fs)
        
        if st_onset + st_duration < num_samples:
            # Apply elevation to the ST segment
            ecg_signal[st_onset:st_onset+st_duration] += elevation_amount
    
    return ecg_signal

def generate_dataset(output_file, num_normal=140, num_abnormal=50):
    """
    Generate a dataset of normal and abnormal ECGs
    
    Parameters:
    - output_file: name of the output CSV file
    - num_normal: number of normal ECG samples
    - num_abnormal: number of abnormal ECG samples
    """
    print(f"Generating {num_normal} normal and {num_abnormal} abnormal ECG samples...")
    
    # Parameters
    duration = 32  # seconds
    fs = 250  # Hz
    
    # Create normal samples with variations
    normal_samples = []
    for i in range(num_normal):
        # Vary heart rate for different normal samples
        heart_rate = np.random.uniform(60, 90)
        noise_level = np.random.uniform(0.03, 0.08)
        
        # Generate the ECG signal
        ecg = generate_normal_ecg(duration, fs, heart_rate, noise_level)
        
        # Add label (0 for normal)
        sample = np.append(ecg, 0)
        normal_samples.append(sample)
        
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{num_normal} normal samples")
    
    # Create abnormal samples (mix of AFib and ST elevation)
    abnormal_samples = []
    
    # AFib samples
    num_afib = num_abnormal // 2
    for i in range(num_afib):
        # Vary heart rate for different AFib samples
        heart_rate = np.random.uniform(100, 150)
        noise_level = np.random.uniform(0.04, 0.09)
        
        # Generate the AFib ECG signal
        ecg = generate_afib_ecg(duration, fs, heart_rate, noise_level)
        
        # Add label (1 for abnormal)
        sample = np.append(ecg, 1)
        abnormal_samples.append(sample)
        
        if (i+1) % 5 == 0:
            print(f"Generated {i+1}/{num_afib} AFib samples")
    
    # ST elevation samples
    num_st = num_abnormal - num_afib
    for i in range(num_st):
        # Vary heart rate and elevation for different ST elevation samples
        heart_rate = np.random.uniform(60, 100)
        noise_level = np.random.uniform(0.03, 0.07)
        elevation = np.random.uniform(0.2, 0.5)
        
        # Generate the ST elevation ECG signal
        ecg = generate_st_elevation_ecg(duration, fs, heart_rate, noise_level, elevation)
        
        # Add label (1 for abnormal)
        sample = np.append(ecg, 1)
        abnormal_samples.append(sample)
        
        if (i+1) % 5 == 0:
            print(f"Generated {i+1}/{num_st} ST elevation samples")
    
    # Combine all samples
    all_samples = normal_samples + abnormal_samples
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(all_samples)
    df.to_csv(output_file, index=False, header=False)
    
    print(f"Generated {len(all_samples)} samples and saved to {output_file}")
    print(f"- Normal samples: {len(normal_samples)}")
    print(f"- Abnormal samples: {len(abnormal_samples)}")
    print(f"Each sample has {len(all_samples[0])-1} data points + 1 label")
    
    return len(normal_samples), len(abnormal_samples)

def plot_sample_ecgs():
    """
    Generate and plot examples of normal and abnormal ECGs for visualization
    """
    # Parameters
    duration = 10  # seconds
    fs = 250  # Hz
    
    # Generate examples
    normal_ecg = generate_normal_ecg(duration, fs, heart_rate=70, noise_level=0.05)
    afib_ecg = generate_afib_ecg(duration, fs, base_heart_rate=120, noise_level=0.05)
    st_ecg = generate_st_elevation_ecg(duration, fs, heart_rate=75, noise_level=0.05, elevation_amount=0.3)
    
    # Time axis
    t = np.arange(len(normal_ecg)) / fs
    
    # Create plots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot normal ECG
    axs[0].plot(t, normal_ecg)
    axs[0].set_title("Normal ECG")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude (mV)")
    axs[0].grid(True)
    
    # Plot AFib ECG
    axs[1].plot(t, afib_ecg)
    axs[1].set_title("Atrial Fibrillation ECG")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude (mV)")
    axs[1].grid(True)
    
    # Plot ST elevation ECG
    axs[2].plot(t, st_ecg)
    axs[2].set_title("ST Elevation ECG")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Amplitude (mV)")
    axs[2].grid(True)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs("sample_plots", exist_ok=True)
    plt.savefig("sample_plots/sample_ecgs.png", dpi=150)
    plt.close()
    
    print("Sample ECGs plotted and saved to sample_plots/sample_ecgs.png")

if __name__ == "__main__":
    # Check if the output directory exists
    os.makedirs("data", exist_ok=True)
    
    # Generate dataset
    output_file = "data/synthetic_ecg.csv"
    num_normal, num_abnormal = generate_dataset(output_file, num_normal=140, num_abnormal=50)
    
    # Also create a smaller file for testing
    small_output_file = "Real ECG.csv"
    generate_dataset(small_output_file, num_normal=20, num_abnormal=10)
    
    # Plot sample ECGs
    plot_sample_ecgs()
    
    print("\nDone! Files created:")
    print(f"1. {output_file} - Full dataset with {num_normal} normal and {num_abnormal} abnormal samples")
    print(f"2. {small_output_file} - Small test dataset with 20 normal and 10 abnormal samples")
    print("3. sample_plots/sample_ecgs.png - Visual examples of the different ECG types")
    print("\nYou can now run standalone_ecg_simulator.py to visualize and interact with the data.")