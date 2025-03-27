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
    variability = np.random.normal(0, 0.03 * period, int(num_samples / (period * sampling_rate / 1000)))
    variability_extended = np.interp(time, 
                                    np.linspace(0, duration_sec * 1000, len(variability)), 
                                    variability)
    
    # Create empty ECG signal
    ecg = np.zeros(len(time))
    
    # Define waveform templates with more realistic morphology
    # Use gamma functions for more realistic P and T waves
    def p_wave_template(x):
        # Use gamma function for more realistic P wave
        amplitude = 0.15
        p_width = 0.08 * period  # P wave width (in ms)
        beta = 4.0  # Shape parameter
        # Normalized time from 0 to 1 within the P wave duration
        x_norm = x / p_width
        return amplitude * (x_norm ** (beta-1)) * np.exp(-x_norm * beta) * (x_norm <= 1) * 4
    
    def qrs_complex_template(x):
        # More realistic QRS complex with distinct Q, R, and S waves
        q_width = 0.02 * period  # Q wave width
        r_width = 0.03 * period  # R wave width
        s_width = 0.02 * period  # S wave width
        
        q_amp = -0.2  # Q wave depth
        r_amp = 1.0   # R wave height
        s_amp = -0.3  # S wave depth
        
        # Q wave (initial downward deflection)
        q_wave = q_amp * np.exp(-(x ** 2) / (2 * (q_width/5) ** 2)) * (x <= q_width)
        
        # R wave (upward deflection)
        r_wave = r_amp * np.exp(-((x - q_width) ** 2) / (2 * (r_width/3) ** 2)) * ((x > q_width) & (x <= q_width + r_width))
        
        # S wave (final downward deflection)
        s_wave = s_amp * np.exp(-((x - q_width - r_width) ** 2) / (2 * (s_width/3) ** 2)) * (x > q_width + r_width)
        
        return q_wave + r_wave + s_wave
    
    def t_wave_template(x):
        # Use gamma function for more realistic T wave
        amplitude = 0.3
        t_width = 0.16 * period  # T wave width (in ms)
        beta = 5.5  # Shape parameter - 5.5 gives a nice asymmetric rise and fall
        # Normalized time from 0 to 1 within the T wave duration
        x_norm = x / t_width
        return amplitude * (x_norm ** (beta-1)) * np.exp(-x_norm * beta) * (x_norm <= 1) * 4
    
    # Generate each beat
    for i, beat_time in enumerate(np.arange(0, duration_sec * 1000, period)):
        # Add heart rate variability to this beat
        if i < len(variability):
            beat_time += variability[i]
        
        # Time offsets for each wave
        p_offset = beat_time - 0.2 * period  # P wave starts before QRS
        qrs_offset = beat_time              # QRS centered at beat time
        t_offset = beat_time + 0.05 * period  # T wave after QRS
        
        # Generate time arrays relative to each wave's start
        beat_indices = np.where((time >= beat_time - 0.3 * period) & (time <= beat_time + 0.5 * period))[0]
        
        for idx in beat_indices:
            # P wave
            if time[idx] >= p_offset and time[idx] < p_offset + 0.11 * period:
                ecg[idx] += p_wave_template(time[idx] - p_offset)
            
            # QRS complex
            if time[idx] >= qrs_offset - 0.03 * period and time[idx] < qrs_offset + 0.08 * period:
                ecg[idx] += qrs_complex_template(time[idx] - (qrs_offset - 0.03 * period))
            
            # T wave
            if time[idx] >= t_offset and time[idx] < t_offset + 0.2 * period:
                ecg[idx] += t_wave_template(time[idx] - t_offset)
    
    # Add baseline wander (low frequency drift)
    if respiration_effect:
        # Simulate respiration at about 15-20 breaths per minute
        resp_freq = random.uniform(0.25, 0.33)  # Hz
        resp_amp = random.uniform(0.05, 0.15)   # Amplitude
        baseline_wander = resp_amp * np.sin(2 * np.pi * resp_freq * time / 1000)
        ecg += baseline_wander
    
    # Add more realistic noise patterns
    # White noise component
    noise_white = noise_level * 0.5 * np.random.normal(0, 1, len(ecg))
    
    # Low frequency noise component (muscle artifact)
    noise_low = np.zeros(len(ecg))
    for _ in range(10):  # Add multiple low frequency components
        freq = random.uniform(0.5, 10)  # Hz
        amp = noise_level * 0.05 * random.random()
        phase = random.uniform(0, 2*np.pi)
        noise_low += amp * np.sin(2*np.pi*freq*time/1000 + phase)
    
    # High frequency noise (EMG-like)
    noise_high = np.zeros(len(ecg))
    envelope = np.random.normal(0, 1, len(ecg)//50)
    envelope = np.interp(np.arange(len(ecg)), 
                          np.linspace(0, len(ecg), len(envelope)), 
                          envelope)
    for _ in range(5):
        freq = random.uniform(40, 100)  # Hz
        amp = noise_level * 0.03
        phase = random.uniform(0, 2*np.pi)
        noise_high += amp * envelope * np.sin(2*np.pi*freq*time/1000 + phase)
    
    # Combine all noise components
    ecg += noise_white + noise_low + noise_high
    
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
        # Tachycardia often has shortened T-P interval but preserves QRS morphology
        return generate_normal_ecg(file_path, heart_rate=heart_rate, noise_level=0.05)
    
    elif abnormality == 'bradycardia':
        heart_rate = params.get('heart_rate', 45)
        # Bradycardia often has prolonged T-P interval but normal QRS morphology
        return generate_normal_ecg(file_path, heart_rate=heart_rate, noise_level=0.05)
    
    elif abnormality == 'arrhythmia':
        # Generate irregular heartbeats
        duration_sec = params.get('duration_sec', 32)
        sampling_rate = params.get('sampling_rate', 250)
        num_samples = duration_sec * sampling_rate
        time = np.linspace(0, duration_sec * 1000, num_samples)  # Time in ms
        ecg = np.zeros(len(time))
        
        # Base heart rate with high variability
        base_hr = params.get('base_hr', 75)
        base_period = 60000.0 / base_hr  # ms
        irregularity = params.get('irregularity', 0.3)
        
        # Generate variable RR intervals
        rr_intervals = []
        current_time = 0
        while current_time < duration_sec * 1000:
            # Generate next RR interval with irregularity
            if random.random() < 0.1:  # 10% chance of very irregular beat
                rr = base_period * random.uniform(0.6, 1.5)
            else:
                rr = base_period * random.uniform(1-irregularity, 1+irregularity)
            
            rr_intervals.append(rr)
            current_time += rr
        
        beat_times = np.cumsum(rr_intervals)
        beat_times = beat_times[beat_times < duration_sec * 1000]
        
        # Define wave templates similar to normal ECG but with potential changes
        def p_wave_template(x, amplitude=0.15):
            p_width = 0.08 * base_period
            beta = 4.0
            x_norm = x / p_width
            return amplitude * (x_norm ** (beta-1)) * np.exp(-x_norm * beta) * (x_norm <= 1) * 4
        
        def qrs_complex_template(x, morphology='normal'):
            q_width = 0.02 * base_period
            r_width = 0.03 * base_period
            s_width = 0.02 * base_period
            
            if morphology == 'normal':
                q_amp = -0.2
                r_amp = 1.0
                s_amp = -0.3
            elif morphology == 'wide':
                # Wide QRS complex (bundle branch block-like)
                q_width *= 1.5
                r_width *= 1.5
                s_width *= 1.5
                q_amp = -0.15
                r_amp = 0.9
                s_amp = -0.4
            elif morphology == 'small':
                # Low voltage QRS
                q_amp = -0.1
                r_amp = 0.6
                s_amp = -0.2
            
            q_wave = q_amp * np.exp(-(x ** 2) / (2 * (q_width/5) ** 2)) * (x <= q_width)
            r_wave = r_amp * np.exp(-((x - q_width) ** 2) / (2 * (r_width/3) ** 2)) * ((x > q_width) & (x <= q_width + r_width))
            s_wave = s_amp * np.exp(-((x - q_width - r_width) ** 2) / (2 * (s_width/3) ** 2)) * (x > q_width + r_width)
            
            return q_wave + r_wave + s_wave
        
        def t_wave_template(x, amplitude=0.3, inverted=False):
            t_width = 0.16 * base_period
            beta = 5.5
            x_norm = x / t_width
            if inverted:
                amplitude = -amplitude  # Inverted T wave
            return amplitude * (x_norm ** (beta-1)) * np.exp(-x_norm * beta) * (x_norm <= 1) * 4
        
        # Generate each beat with varying morphologies
        for i, beat_time in enumerate(beat_times):
            # Decide beat morphology (some abnormal beats)
            has_p_wave = random.random() > 0.2  # 80% chance of having P wave
            qrs_morphology = random.choices(
                ['normal', 'wide', 'small'], 
                weights=[0.7, 0.2, 0.1], 
                k=1
            )[0]
            t_inverted = random.random() < 0.15  # 15% chance of inverted T wave
            
            # Time offsets for each wave
            p_offset = beat_time - 0.2 * base_period
            qrs_offset = beat_time
            t_offset = beat_time + 0.05 * base_period
            
            # Generate time arrays relative to each wave's start
            beat_indices = np.where((time >= beat_time - 0.3 * base_period) & 
                                   (time <= beat_time + 0.5 * base_period))[0]
            
            for idx in beat_indices:
                # P wave (if present)
                if has_p_wave and time[idx] >= p_offset and time[idx] < p_offset + 0.11 * base_period:
                    ecg[idx] += p_wave_template(time[idx] - p_offset)
                
                # QRS complex
                if time[idx] >= qrs_offset - 0.03 * base_period and time[idx] < qrs_offset + 0.08 * base_period:
                    ecg[idx] += qrs_complex_template(time[idx] - (qrs_offset - 0.03 * base_period), 
                                                   morphology=qrs_morphology)
                
                # T wave
                if time[idx] >= t_offset and time[idx] < t_offset + 0.2 * base_period:
                    ecg[idx] += t_wave_template(time[idx] - t_offset, inverted=t_inverted)
        
        # Add noise
        noise_level = params.get('noise_level', 0.05)
        noise = noise_level * np.random.normal(0, 1, len(ecg))
        ecg += noise
        
        # Save to CSV
        data = np.column_stack((time, ecg))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savetxt(file_path, data, delimiter=',', header='time_ms,ecg_mv', comments='')
            
        return time, ecg
        
    elif abnormality == 'afib':
        # Atrial fibrillation: irregular rhythm, absence of P waves, rapid atrial activity
        duration_sec = params.get('duration_sec', 32)
        sampling_rate = params.get('sampling_rate', 250)
        num_samples = duration_sec * sampling_rate
        time = np.linspace(0, duration_sec * 1000, num_samples)
        ecg = np.zeros(len(time))
        
        # Highly irregular RR intervals (characteristic of AFib)
        base_hr = params.get('base_hr', 100)  # Often higher in AFib
        base_period = 60000.0 / base_hr
        
        # Generate very irregular RR intervals
        rr_intervals = []
        current_time = 0
        while current_time < duration_sec * 1000:
            rr = base_period * random.uniform(0.6, 1.4)  # High variability in AFib
            rr_intervals.append(rr)
            current_time += rr
        
        beat_times = np.cumsum(rr_intervals)
        beat_times = beat_times[beat_times < duration_sec * 1000]
        
        # Add fine fibrillatory waves (f-waves) - typical of AFib
        f_wave_freq = params.get('f_wave_freq', random.uniform(4, 8))  # Hz
        f_wave_amp = params.get('f_wave_amp', random.uniform(0.05, 0.12))
        f_waves = f_wave_amp * np.sin(2 * np.pi * f_wave_freq * time / 1000)
        
        # Add baseline undulation to f-waves (makes them more realistic)
        for i in range(3):
            mod_freq = random.uniform(0.3, 1.2)
            mod_amp = random.uniform(0.02, 0.06)
            f_waves *= 1 + mod_amp * np.sin(2 * np.pi * mod_freq * time / 1000)
        
        ecg += f_waves
        
        # Define wave templates (no P waves in AFib)
        def qrs_complex_template(x):
            q_width = 0.02 * base_period
            r_width = 0.03 * base_period
            s_width = 0.02 * base_period
            
            q_amp = -0.2
            r_amp = 1.0
            s_amp = -0.3
            
            q_wave = q_amp * np.exp(-(x ** 2) / (2 * (q_width/5) ** 2)) * (x <= q_width)
            r_wave = r_amp * np.exp(-((x - q_width) ** 2) / (2 * (r_width/3) ** 2)) * ((x > q_width) & (x <= q_width + r_width))
            s_wave = s_amp * np.exp(-((x - q_width - r_width) ** 2) / (2 * (s_width/3) ** 2)) * (x > q_width + r_width)
            
            return q_wave + r_wave + s_wave
        
        def t_wave_template(x):
            t_width = 0.16 * base_period
            beta = 5.5
            x_norm = x / t_width
            return 0.3 * (x_norm ** (beta-1)) * np.exp(-x_norm * beta) * (x_norm <= 1) * 4
        
        # Generate each beat with no P waves (typical of AFib)
        for beat_time in beat_times:
            qrs_offset = beat_time
            t_offset = beat_time + 0.05 * base_period
            
            beat_indices = np.where((time >= beat_time - 0.03 * base_period) & 
                                  (time <= beat_time + 0.5 * base_period))[0]
            
            for idx in beat_indices:
                # QRS complex
                if time[idx] >= qrs_offset - 0.03 * base_period and time[idx] < qrs_offset + 0.08 * base_period:
                    ecg[idx] += qrs_complex_template(time[idx] - (qrs_offset - 0.03 * base_period))
                
                # T wave
                if time[idx] >= t_offset and time[idx] < t_offset + 0.2 * base_period:
                    ecg[idx] += t_wave_template(time[idx] - t_offset)
        
        # Add noise
        noise_level = params.get('noise_level', 0.05)
        noise = noise_level * np.random.normal(0, 1, len(ecg))
        ecg += noise
        
        # Save to CSV
        data = np.column_stack((time, ecg))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savetxt(file_path, data, delimiter=',', header='time_ms,ecg_mv', comments='')
            
        return time, ecg
        
    elif abnormality == 'st_elevation':
        # Generate normal ECG first
        heart_rate = params.get('heart_rate', 80)
        time, ecg = generate_normal_ecg(file_path + '.temp', heart_rate=heart_rate, 
                                      noise_level=0.03)  # Less noise to see ST changes clearly
        
        # Define ST segment elevation more clearly
        elevation = params.get('elevation', random.uniform(0.2, 0.4))  # in mV
        duration_sec = len(time) / (params.get('sampling_rate', 250))
        
        # Calculate beat locations
        period = 60000.0 / heart_rate  # in ms
        beat_times = np.arange(period/2, duration_sec*1000, period)
        
        # Apply ST elevation to all beats
        for beat_time in beat_times:
            peak_idx = int(beat_time * params.get('sampling_rate', 250) / 1000)
            
            # Create window for ST segment (starts after S wave, ends before T wave)
            if peak_idx + 60 < len(ecg):
                # Identify start and end of ST segment
                st_start = peak_idx + 20  # ~80ms after R peak (after S wave)
                st_end = peak_idx + 60    # ~240ms after R peak (before T wave peak)
                
                # Create smooth elevation profile with realistic morphology
                st_length = st_end - st_start
                
                # Create a concave/convex ST elevation pattern (characteristic of STEMI)
                x = np.linspace(0, 1, st_length)
                
                # Choose between different ST morphologies
                st_morphology = random.choice(['convex', 'straight', 'concave'])
                
                if st_morphology == 'convex':
                    # Convex upward (classic STEMI pattern)
                    st_shape = elevation * (1 - (2*x-1)**2)
                elif st_morphology == 'straight':
                    # Straight elevation
                    st_shape = np.ones(st_length) * elevation
                else:  # concave
                    # Concave upward
                    st_shape = elevation * np.sqrt(1 - (2*x-1)**2)
                
                # Apply the elevation with smooth transitions
                st_shape[:5] = st_shape[0] * np.linspace(0, 1, 5)  # Smooth start
                st_shape[-5:] = st_shape[-1] * np.linspace(1, 0.8, 5)  # Smooth end
                
                ecg[st_start:st_end] += st_shape
                
                # Optionally make the T wave more prominent and possibly inverted
                # as these patterns can accompany ST elevation in myocardial infarction
                if random.random() < 0.5:  # 50% chance of T wave changes
                    t_start = st_end
                    t_end = min(t_start + 40, len(ecg))
                    t_peak = t_start + (t_end - t_start) // 2
                    
                    if random.random() < 0.3:  # 30% chance of T inversion
                        # Create inverted T wave
                        t_amp = -0.3
                    else:
                        # Create tall T wave
                        t_amp = 0.5
                    
                    # Create T wave
                    x = np.linspace(-1, 1, t_end - t_start)
                    t_wave = t_amp * (1 - x**2)
                    ecg[t_start:t_end] = t_wave  # Replace the existing T wave
        
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