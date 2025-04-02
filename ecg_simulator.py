import numpy as np
import random

class ECGGenerator:
    """Generates simulated ECG signals"""
    def __init__(self, sampling_rate=250):
        self.fs = sampling_rate
        self.time_step = 1.0 / self.fs
        self.heart_rate = 75  # BPM
        self.noise_level = 0.05 # Noise amplitude
        self.ecg_type = 'normal' # Default type
        self.current_time_in_beat = 0.0
        self.last_rr_interval = 60.0 / self.heart_rate

        # Load or create ECG templates
        self.templates = self._load_ecg_templates()

    def _load_ecg_templates(self):
        """Load or create ECG templates for different types"""
        templates = {}
        
        # Basic Normal ECG template (simplified P-QRS-T)
        t_beat = np.arange(0, 1.0, self.time_step) # Time vector for one beat (normalized to 1 second)
        # --- Widen the Gaussian components --- 
        p_std_dev = 0.025 # Wider P wave
        qrs_std_dev = 0.02 # Wider QRS 
        t_std_dev = 0.05 # Wider T wave

        # Adjust centers slightly for better spacing with wider shapes
        p_center = 0.15
        qrs_center = 0.30
        # QRS negative deflection (Q or S wave component) - make it slightly before main peak
        qs_center = 0.28 
        t_center = 0.55

        p_wave = 0.1 * np.exp(-((t_beat - p_center)**2) / (2 * p_std_dev**2))
        # Make QRS sharper positive, slight negative dip before
        qrs = 1.0 * np.exp(-((t_beat - qrs_center)**2) / (2 * qrs_std_dev**2)) - \
              0.15 * np.exp(-((t_beat - qs_center)**2) / (2 * (qrs_std_dev * 0.8)**2)) # Narrower Q/S dip
        t_wave = 0.2 * np.exp(-((t_beat - t_center)**2) / (2 * t_std_dev**2))
        # -----------------------------------
        normal_template = p_wave + qrs + t_wave
        templates['normal'] = normal_template

        # Atrial Fibrillation: Irregular rhythm, no distinct P waves, fibrillatory waves
        afib_template = qrs + t_wave # No P-wave
        # Add some baseline noise representing fibrillatory waves
        fibrillatory_waves = np.random.normal(0, 0.03, size=afib_template.shape)
        templates['afib'] = afib_template + fibrillatory_waves

        # ST Elevation: Elevate the segment after QRS
        st_elevation_template = normal_template.copy()
        st_start_idx = int(0.3 / self.time_step)
        st_end_idx = int(0.4 / self.time_step)
        st_elevation_template[st_start_idx:st_end_idx] += 0.2 # Elevate ST segment
        templates['st_elevation'] = st_elevation_template
        
        # Tachycardia and Bradycardia use the normal template but adjusted timing
        templates['tachycardia'] = normal_template
        templates['bradycardia'] = normal_template

        return templates

    def get_available_types(self):
        """Return list of available ECG types"""
        return list(self.templates.keys())

    def set_heart_rate(self, bpm):
        self.heart_rate = max(20, min(bpm, 250))

    def set_noise_level(self, level):
        self.noise_level = max(0, min(level, 1.0))
    
    def set_ecg_type(self, ecg_type):
        type_key = ecg_type.lower().replace(" ", "_")
        if type_key in self.templates:
            self.ecg_type = type_key
        else:
            print(f"Warning: Unknown ECG type '{ecg_type}'. Using normal.")
            self.ecg_type = 'normal'

    def reset(self):
        """Reset the generator state"""
        self.current_time_in_beat = 0.0
        self.last_rr_interval = 60.0 / self.heart_rate

    def generate(self, num_samples):
        """Generate a segment of ECG signal"""
        segment = np.zeros(num_samples)
        
        template = self.templates.get(self.ecg_type, self.templates['normal'])
        template_len = len(template)

        # Use the RR interval consistent with the current HR setting for indexing within this segment
        current_target_rr_interval = 60.0 / self.heart_rate
        # For Afib, we still need variability, but let's apply it to the next interval setting
        next_beat_rr_interval = current_target_rr_interval
        if self.ecg_type == 'afib':
             # Add variability for the *next* beat interval calculation
            variability = np.random.normal(0, 0.1 * current_target_rr_interval)
            next_beat_rr_interval = max(0.2, min(current_target_rr_interval + variability, 2.0))

        for i in range(num_samples):
            # Determine position in the current beat cycle using the interval of the beat we are IN
            # Ensure last_rr_interval is not zero to avoid division issues
            beat_duration = self.last_rr_interval if self.last_rr_interval > 0 else current_target_rr_interval
            pos_in_beat = self.current_time_in_beat / beat_duration
            pos_in_beat = min(max(pos_in_beat, 0.0), 1.0) # Clamp position just in case

            # Map position to template index
            template_idx = int(pos_in_beat * template_len)
            template_idx = min(template_idx, template_len - 1) # Ensure index is within bounds

            # Get value from template
            base_value = template[template_idx]
            
            # Add noise
            noise = np.random.normal(0, self.noise_level)
            segment[i] = base_value + noise

            # Advance time within the beat
            self.current_time_in_beat += self.time_step

            # Check if beat completed
            if self.current_time_in_beat >= beat_duration:
                self.current_time_in_beat -= beat_duration # Reset time for next beat using the duration we just finished
                # Set the duration for the NEXT beat based on current HR / afib logic
                self.last_rr_interval = next_beat_rr_interval 
                # Recalculate next_beat_rr_interval for the subsequent beat
                current_target_rr_interval = 60.0 / self.heart_rate
                next_beat_rr_interval = current_target_rr_interval
                if self.ecg_type == 'afib':
                    variability = np.random.normal(0, 0.1 * current_target_rr_interval)
                    next_beat_rr_interval = max(0.2, min(current_target_rr_interval + variability, 2.0))

        return segment 