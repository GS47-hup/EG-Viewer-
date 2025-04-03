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

    def _gaussian(self, t, center, sigma, amplitude):
        """Generates a Gaussian curve."""
        return amplitude * np.exp(-((t - center)**2) / (2 * sigma**2))

    def _load_ecg_templates(self):
        """Generate ECG templates using Gaussian functions for more realistic shapes."""
        templates = {}
        
        # --- Parameters for a more realistic Normal Sinus Rhythm (adjust as needed) ---
        # Timings are approximate fractions of a 1-second cycle for easier scaling if needed
        # Assuming a beat duration reference of ~0.8s for parameter setting
        base_duration = 0.8 # Reference duration in seconds
        p_center, p_sigma, p_amp = 0.12 * base_duration, 0.020 * base_duration, 0.15  # P wave
        q_center, q_sigma, q_amp = 0.20 * base_duration, 0.008 * base_duration, -0.12 # Q wave
        r_center, r_sigma, r_amp = 0.22 * base_duration, 0.012 * base_duration, 1.40  # R wave (Taller, sharper)
        s_center, s_sigma, s_amp = 0.25 * base_duration, 0.015 * base_duration, -0.25 # S wave
        t_center, t_sigma, t_amp = 0.40 * base_duration, 0.040 * base_duration, 0.35  # T wave

        # Template duration slightly longer than typical beat to allow for variability/noise at end
        template_duration = 1.2 # seconds
        num_points = int(template_duration * self.fs)
        t_beat = np.linspace(0, template_duration, num_points, endpoint=False)

        # Initialize template
        normal_template = np.zeros(num_points)

        # Generate individual waves using the helper
        p_wave = self._gaussian(t_beat, p_center, p_sigma, p_amp)
        q_wave = self._gaussian(t_beat, q_center, q_sigma, q_amp)
        r_wave = self._gaussian(t_beat, r_center, r_sigma, r_amp)
        s_wave = self._gaussian(t_beat, s_center, s_sigma, s_amp)
        t_wave = self._gaussian(t_beat, t_center, t_sigma, t_amp)

        # Sum waves to create the final template
        normal_template = p_wave + q_wave + r_wave + s_wave + t_wave

        # --- Store template ---
        templates['normal'] = normal_template

        # --- Recreate other templates based on the new normal --- 
        # For AFib, remove the P wave and add fibrillatory baseline
        afib_template = q_wave + r_wave + s_wave + t_wave # No P wave
        fibrillatory_waves = np.random.normal(0, 0.04, size=num_points) # Slightly more noise
        templates['afib'] = afib_template + fibrillatory_waves

        st_elevation_template = normal_template.copy()
        # Elevate the ST segment (time between QRS end and T start)
        # Approximate ST segment start/end based on S and T wave timings
        st_start_time = s_center + 2*s_sigma # Approx end of S wave
        st_end_time = t_center - 2*t_sigma   # Approx start of T wave
        st_start_idx = int(st_start_time / self.time_step)
        st_end_idx = int(st_end_time / self.time_step)
        if st_start_idx < st_end_idx: # Ensure indices are valid
            st_elevation_template[st_start_idx:st_end_idx] += 0.2 # Elevate ST segment
        templates['st_elevation'] = st_elevation_template
        
        # Tachycardia and Bradycardia use the normal template but adjusted timing in generate()
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
        if not hasattr(self, '_gen_debug_counter'): self._gen_debug_counter = 0
        if self._gen_debug_counter == 0: # Print once per generate call
            print(f"[Sim Generate Debug] Using type: {self.ecg_type}, Template length: {len(template)}")
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

            # <<< Debug Print >>>
            if self._gen_debug_counter % (self.fs // 4) == 0: # Print ~4 times per second
                print(f"[Sim Generate Debug] i={i}, time_in_beat={self.current_time_in_beat:.4f}, beat_dur={beat_duration:.4f}, pos={pos_in_beat:.4f}, idx={template_idx}, base={base_value:.3f}, noise={noise:.3f}, final={segment[i]:.3f}")
            self._gen_debug_counter += 1
            # <<< End Debug Print >>>

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

# --- Add plotting code for direct execution ---
if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not found. Cannot plot template.")
        print("Install it using: pip install matplotlib")
        plt = None

    if plt:
        fs_plot = 250 # Use a typical sampling rate for plotting
        generator = ECGGenerator(sampling_rate=fs_plot)
        normal_template = generator.templates['normal']
        time_vector = np.arange(len(normal_template)) / fs_plot

        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, normal_template)
        plt.title("Generated 'Normal' ECG Template (One Beat)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude (mV)")
        plt.grid(True)
        plt.show() 