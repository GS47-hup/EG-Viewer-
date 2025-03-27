import os
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal

def generate_normal_ecg(file_path, duration_sec=32, sampling_rate=250, heart_rate=70, 
                        noise_level=0.05, respiration_effect=False):
    """
    Generate synthetic normal ECG signal and save to CSV
    
    Parameters:
    - file_path: Path to save the generated ECG data
    - duration_sec: Duration in seconds
    - sampling_rate: Sampling rate in Hz
    - heart_rate: Heart rate in BPM
    - noise_level: Amount of noise to add (0.0-1.0)
    - respiration_effect: Add respiratory modulation
    """
    # Calculate time array
    num_samples = duration_sec * sampling_rate
    time = np.linspace(0, duration_sec * 1000, num_samples)  # Time in ms
    
    # Generate ECG components
    period = 60000.0 / heart_rate  # RR interval in ms
    
    # Add slight variability to heart rate (natural HRV)
    variability = np.random.normal(0, 0.05 * period, int(num_samples / (period * sampling_rate / 1000)))
    variability_extended = np.interp(time, 
                                    np.linspace(0, duration_sec * 1000, len(variability)), 
                                    variability)
    
    # Calculate phase
    phase = (time % (period + variability_extended)) / period * 2 * np.pi
    
    # Create ECG components
    p_wave = 0.15 * np.sin(phase)
    p_wave = np.where((phase >= 0) & (phase <= 0.7 * np.pi), p_wave, 0)
    
    qrs_complex = np.zeros_like(phase)
    qrs_complex = np.where((phase >= 0.7*np.pi) & (phase <= 0.8*np.pi), -0.5 * np.sin(phase*10), qrs_complex)
    qrs_complex = np.where((phase >= 0.8*np.pi) & (phase <= 0.9*np.pi), 1.5 * np.sin(phase*10), qrs_complex)
    qrs_complex = np.where((phase >= 0.9*np.pi) & (phase <= 1.0*np.pi), -0.3 * np.sin(phase*10), qrs_complex)
    
    t_wave = 0.3 * np.sin(phase)
    t_wave = np.where((phase >= 1.0*np.pi) & (phase <= 1.7*np.pi), t_wave, 0)
    
    ecg = p_wave + qrs_complex + t_wave
    
    # Add respiratory modulation if requested
    if respiration_effect:
        # Simulate respiration at 15 breaths per minute
        resp_freq = 15 / 60  # Hz
        resp_modulation = 0.1 * np.sin(2 * np.pi * resp_freq * time / 1000)
        ecg += resp_modulation
    
    # Add noise
    noise = noise_level * np.random.normal(0, 1, len(ecg))
    ecg += noise
    
    # Save to CSV
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data = np.column_stack((time, ecg))
    np.savetxt(file_path, data, delimiter=',', header='time_ms,ecg_mv', comments='')
    
    return time, ecg

def generate_abnormal_ecg(file_path, abnormality='tachycardia', **params):
    """
    Generate synthetic abnormal ECG signal
    
    Parameters:
    - file_path: Path to save the generated ECG data
    - abnormality: Type of abnormality ('tachycardia', 'bradycardia', 'arrhythmia', 'afib', 'st_elevation')
    - params: Additional parameters specific to each abnormality
    """
    if abnormality == 'tachycardia':
        heart_rate = params.get('heart_rate', 120)
        return generate_normal_ecg(file_path, heart_rate=heart_rate, noise_level=0.05)
    
    elif abnormality == 'bradycardia':
        heart_rate = params.get('heart_rate', 45)
        return generate_normal_ecg(file_path, heart_rate=heart_rate, noise_level=0.05)
    
    elif abnormality == 'arrhythmia':
        # Generate irregular heartbeats
        duration_sec = params.get('duration_sec', 32)
        sampling_rate = params.get('sampling_rate', 250)
        
        # First create a baseline normal ECG
        time, ecg = generate_normal_ecg(file_path + '.temp', duration_sec=duration_sec, 
                                        sampling_rate=sampling_rate, heart_rate=75)
        
        # Add irregularity to RR intervals
        num_samples = len(time)
        irregularity = params.get('irregularity', 0.3)
        
        # Detect R peaks in base ECG
        base_period = 60000.0 / 75  # ms (for 75 BPM)
        expected_peaks = np.arange(base_period/2, duration_sec*1000, base_period)
        
        # Add irregularity to peak positions
        irregular_peaks = []
        for peak in expected_peaks:
            # Skip some beats or add extra beats randomly
            if random.random() < irregularity * 0.3:
                if random.random() < 0.5:
                    # Skip beat
                    continue
                else:
                    # Add extra beat
                    extra_peak = peak - base_period * random.uniform(0.3, 0.5)
                    if extra_peak > 0:
                        irregular_peaks.append(extra_peak)
            
            # Add normal beat with some timing variation
            jitter = base_period * random.uniform(-irregularity, irregularity)
            irregular_peaks.append(peak + jitter)
        
        # Regenerate ECG with irregular peaks
        ecg_new = np.zeros_like(ecg)
        
        for peak in irregular_peaks:
            if peak < 0 or peak >= duration_sec*1000:
                continue
                
            idx = int(peak * sampling_rate / 1000)
            # Generate QRS complex around this peak
            if idx - 25 >= 0 and idx + 25 < num_samples:
                # Create a QRS template - Fixed to length 25 to match the array slices
                qrs = np.concatenate([
                    -0.2 * np.ones(5),  # Q wave
                    np.linspace(-0.2, 1.0, 5),  # Q-R transition
                    1.0 * np.ones(5),  # R peak
                    np.linspace(1.0, -0.3, 5),  # R-S transition
                    -0.3 * np.ones(5)  # S wave
                ])
                
                # P wave
                if random.random() < 0.7:  # 30% chance of missing P wave
                    if idx - 25 >= 0:
                        p_wave = 0.2 * np.sin(np.linspace(0, np.pi, 15))
                        ecg_new[idx-25:idx-10] += p_wave
                
                # QRS - ensure the array slice and QRS array have the same length
                if idx - 10 >= 0 and idx + 15 < num_samples:
                    ecg_new[idx-10:idx+15] += qrs
                
                # T wave
                if idx + 15 < num_samples - 20:
                    t_wave = 0.3 * np.sin(np.linspace(0, np.pi, 20))
                    ecg_new[idx+15:idx+35] += t_wave
        
        # Add noise
        noise_level = params.get('noise_level', 0.05)
        noise = noise_level * np.random.normal(0, 1, len(ecg_new))
        ecg_new += noise
        
        # Save to CSV
        data = np.column_stack((time, ecg_new))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savetxt(file_path, data, delimiter=',', header='time_ms,ecg_mv', comments='')
        
        # Remove temp file
        if os.path.exists(file_path + '.temp'):
            os.remove(file_path + '.temp')
            
        return time, ecg_new
        
    elif abnormality == 'afib':
        # Atrial fibrillation: irregular rhythm, absence of P waves, rapid atrial activity
        duration_sec = params.get('duration_sec', 32)
        sampling_rate = params.get('sampling_rate', 250)
        
        # First generate irregular rhythm like arrhythmia
        time, ecg = generate_abnormal_ecg(file_path + '.temp', abnormality='arrhythmia', 
                                        irregularity=0.4, duration_sec=duration_sec, 
                                        sampling_rate=sampling_rate)
        
        # Add rapid atrial fibrillatory waves (f-waves) instead of P waves
        f_wave_freq = params.get('f_wave_freq', 6)  # Hz
        f_wave_amp = params.get('f_wave_amp', 0.1)
        
        f_waves = f_wave_amp * np.sin(2 * np.pi * f_wave_freq * time / 1000)
        
        # Remove existing P waves by high-pass filtering
        b, a = signal.butter(3, 0.5/(sampling_rate/2), 'highpass')
        ecg_filtered = signal.filtfilt(b, a, ecg)
        
        # Add f-waves
        ecg_new = ecg_filtered + f_waves
        
        # Save to CSV
        data = np.column_stack((time, ecg_new))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savetxt(file_path, data, delimiter=',', header='time_ms,ecg_mv', comments='')
        
        # Remove temp file
        if os.path.exists(file_path + '.temp'):
            os.remove(file_path + '.temp')
            
        return time, ecg_new
        
    elif abnormality == 'st_elevation':
        # Generate normal ECG first
        heart_rate = params.get('heart_rate', 80)
        time, ecg = generate_normal_ecg(file_path + '.temp', heart_rate=heart_rate)
        
        # Add ST segment elevation
        elevation = params.get('elevation', 0.2)  # in mV
        duration_sec = len(time) / (params.get('sampling_rate', 250))
        
        # Detect approximate locations of QRS complexes
        period = 60000.0 / heart_rate  # in ms
        r_peaks = np.arange(period/2, duration_sec*1000, period)
        
        # Add elevation to ST segments
        for peak in r_peaks:
            peak_idx = int(peak * params.get('sampling_rate', 250) / 1000)
            if peak_idx + 50 < len(ecg):
                # Apply elevation to the ST segment (after QRS)
                st_start = peak_idx + 15
                st_end = peak_idx + 50
                
                # Create smooth elevation profile
                elevation_profile = np.zeros(st_end - st_start)
                elevation_profile[:10] = np.linspace(0, elevation, 10)  # Ramp up
                elevation_profile[10:-10] = elevation  # Plateau
                elevation_profile[-10:] = np.linspace(elevation, 0, 10)  # Ramp down
                
                ecg[st_start:st_end] += elevation_profile
        
        # Save to CSV
        data = np.column_stack((time, ecg))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savetxt(file_path, data, delimiter=',', header='time_ms,ecg_mv', comments='')
        
        # Remove temp file
        if os.path.exists(file_path + '.temp'):
            os.remove(file_path + '.temp')
            
        return time, ecg
    
    else:
        print(f"Unknown abnormality: {abnormality}")
        return None, None

def generate_large_test_dataset(output_dir='large_test_dataset', count=100):
    """Generate a large number of ECG samples for testing
    
    Parameters:
    - output_dir: Directory to save generated samples
    - count: Number of samples to generate
    """
    normal_dir = os.path.join(output_dir, 'normal')
    abnormal_dir = os.path.join(output_dir, 'abnormal')
    
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(abnormal_dir, exist_ok=True)
    
    print(f"Generating {count} ECG samples...")
    
    # Create normal samples with varied heart rates and noise levels
    for i in range(count//2):
        hr = random.randint(60, 100)  # Normal heart rate range
        noise_level = random.uniform(0.01, 0.1)
        
        # Add respiration effect to some samples
        respiration = random.choice([True, False])
        respiration_str = "_resp" if respiration else ""
        
        filepath = os.path.join(normal_dir, f"normal_hr{hr}_noise{noise_level:.2f}{respiration_str}.csv")
        generate_normal_ecg(filepath, heart_rate=hr, noise_level=noise_level, respiration_effect=respiration)
        
        if i % 10 == 0:
            print(f"Generated {i+1} normal samples")
    
    # Create abnormal samples
    abnormal_types = ['tachycardia', 'bradycardia', 'arrhythmia', 'afib', 'st_elevation']
    for i in range(count//2):
        abnormal_type = abnormal_types[i % len(abnormal_types)]  # Cycle through types
        filepath = os.path.join(abnormal_dir, f"abnormal_{abnormal_type}_{i}.csv")
        
        if abnormal_type == 'tachycardia':
            hr = random.randint(100, 180)
            generate_abnormal_ecg(filepath, abnormality='tachycardia', heart_rate=hr)
        elif abnormal_type == 'bradycardia':
            hr = random.randint(30, 60)
            generate_abnormal_ecg(filepath, abnormality='bradycardia', heart_rate=hr)
        elif abnormal_type == 'arrhythmia':
            irregularity = random.uniform(0.2, 0.4)
            generate_abnormal_ecg(filepath, abnormality='arrhythmia', irregularity=irregularity)
        elif abnormal_type == 'afib':
            f_wave_freq = random.uniform(4, 8)
            generate_abnormal_ecg(filepath, abnormality='afib', f_wave_freq=f_wave_freq)
        elif abnormal_type == 'st_elevation':
            elevation = random.uniform(0.15, 0.5)
            generate_abnormal_ecg(filepath, abnormality='st_elevation', elevation=elevation)
        
        if i % 10 == 0:
            print(f"Generated {i+1} abnormal samples")
    
    print(f"Generated {count} ECG samples in {output_dir}")
    return count

if __name__ == "__main__":
    # Generate a dataset with 100 samples (50 normal, 50 abnormal)
    generate_large_test_dataset(count=100) 