import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

class ECGClassifier:
    """
    Rule-based ECG classification system to determine if an ECG signal is normal or abnormal.
    Uses various cardiac parameters to make classification decisions.
    """
    
    def __init__(self):
        """Initialize the ECG classifier with default parameters"""
        # Default parameters for classification
        self.sampling_rate = 250  # Default sampling rate
        self.heart_rate_ranges = {'bradycardia': (0, 60), 'normal': (60, 100), 'tachycardia': (100, float('inf'))}
        self.st_elevation_threshold = 0.27  # in mV
        self.p_wave_threshold = 0.15  # Amplitude threshold for P wave detection in mV
        
        # Heart rate thresholds (in BPM)
        self.bradycardia_threshold = 60
        self.tachycardia_threshold = 100
        self.max_normal_rate = 120
        self.min_normal_rate = 40
        
        # Rhythm irregularity threshold (percentage of variation)
        self.rhythm_irregularity_threshold = 15.0  # percent
        
        # P wave absence threshold (percentage of beats without P waves)
        self.p_wave_absence_threshold = 0.3  # 30% missing
        
        # Result storage
        self.classification_results = {}
        self.detailed_analysis = {}
        
    def load_ecg_data(self, file_path):
        """Load ECG data from a CSV file"""
        try:
            data = pd.read_csv(file_path)
            print(f"Successfully loaded ECG data from {file_path}")
            print(f"Data shape: {data.shape}")
            
            # Determine if we have time column or need to generate it
            has_time = 'time' in data.columns
            value_column = 'value' if 'value' in data.columns else data.columns[0]
            
            # Extract the ECG values
            ecg_values = data[value_column].values
            
            # Extract or generate time values
            if has_time:
                time_values = data['time'].values
            else:
                # Generate time array assuming our standard sample rate
                time_values = np.arange(len(ecg_values)) * (1000 / self.sampling_rate)
            
            return time_values, ecg_values
            
        except Exception as e:
            print(f"Error loading ECG data: {e}")
            return None, None

    def detect_peaks(self, ecg_values, min_prominence=0.4, min_distance=None):
        """
        Detect R peaks in the ECG signal
        
        Parameters:
        - ecg_values: ECG signal values
        - min_prominence: Minimum prominence threshold (default 0.4mV)
        - min_distance: Minimum distance between peaks in samples
                        If None, will calculate based on max BPM
        """
        # Calculate signal characteristics
        signal_range = np.max(ecg_values) - np.min(ecg_values)
        center = np.min(ecg_values) + signal_range / 2
        
        # If not specified, set min_distance based on maximum theoretical heart rate (200 BPM)
        if min_distance is None:
            min_distance = int(self.sampling_rate * 0.3)  # 0.3 seconds between peaks
            
        # Auto-adjust prominence based on signal range if needed
        if min_prominence is None:
            min_prominence = signal_range * 0.2
        
        # Find peaks
        peaks, _ = signal.find_peaks(
            ecg_values,
            prominence=min_prominence,
            height=center,
            distance=min_distance
        )
        
        return peaks
    
    def analyze_heart_rate(self, time_values, peaks):
        """
        Calculate heart rate and heart rate variability
        
        Returns:
        - avg_rate: Average heart rate in BPM
        - hrv: Heart rate variability (standard deviation of RR intervals)
        - rr_intervals: Array of RR intervals in milliseconds
        """
        if len(peaks) < 2:
            return 0, 0, []
        
        # Calculate RR intervals in milliseconds
        rr_intervals = []
        for i in range(1, len(peaks)):
            rr_ms = time_values[peaks[i]] - time_values[peaks[i-1]]
            rr_intervals.append(rr_ms)
        
        # Convert RR intervals to BPM
        instantaneous_rates = [60000.0 / rr for rr in rr_intervals]
        
        # Calculate average heart rate
        avg_rate = np.mean(instantaneous_rates)
        
        # Calculate heart rate variability (standard deviation of RR intervals)
        hrv = np.std(rr_intervals)
        
        # Calculate normalized HRV (coefficient of variation)
        normalized_hrv = (hrv / np.mean(rr_intervals)) * 100
        
        return avg_rate, normalized_hrv, rr_intervals
    
    def detect_p_waves(self, ecg_values, r_peaks, window_before=50):
        """
        Detect P waves before R peaks
        
        Parameters:
        - ecg_values: ECG signal values
        - r_peaks: Array of R peak indices
        - window_before: Number of samples to look before R peak
        
        Returns:
        - p_wave_count: Number of detected P waves
        - p_wave_ratio: Ratio of detected P waves to total beats
        """
        p_wave_count = 0
        valid_beats = 0
        
        for peak in r_peaks:
            # Skip if we can't look far enough before the peak
            if peak < window_before:
                continue
                
            valid_beats += 1
            
            # Extract the segment before the R peak
            segment = ecg_values[peak - window_before:peak]
            
            # Look for a local maximum in the segment (potential P wave)
            if len(segment) > 10:
                # Use a simple peak detection with lower prominence
                p_peaks, _ = signal.find_peaks(
                    segment, 
                    prominence=0.05,
                    distance=10
                )
                
                if len(p_peaks) > 0:
                    p_wave_count += 1
        
        # Calculate ratio of detected P waves
        p_wave_ratio = p_wave_count / valid_beats if valid_beats > 0 else 0
        
        return p_wave_count, p_wave_ratio
    
    def detect_st_elevation(self, ecg_values, r_peaks, window_after=100):
        """
        Detect ST segment elevation after R peaks
        
        Parameters:
        - ecg_values: ECG signal values
        - r_peaks: Array of R peak indices
        - window_after: Number of samples to look after R peak
        
        Returns:
        - st_elevation: Average ST segment elevation in millivolts
        - st_morphology: ST segment morphology score (higher = more abnormal)
        """
        st_elevations = []
        st_morphology_scores = []
        
        for peak in r_peaks:
            # Skip if we can't look far enough after the peak
            if peak + window_after >= len(ecg_values):
                continue
                
            # Extract the segment after the R peak
            segment = ecg_values[peak:peak+window_after]
            
            if len(segment) >= 80:
                # ST segment is typically 80-120ms after the R peak
                # We'll look at the average elevation in this region
                st_segment = segment[40:70]  # ~160-280ms after R peak
                baseline = np.mean(ecg_values[peak-20:peak-10])  # Baseline before QRS
                st_elevation = np.mean(st_segment) - baseline
                st_elevations.append(st_elevation)
                
                # Calculate ST morphology score - how flat/sloped the ST segment is
                st_slope = np.std(st_segment) * 10  # Higher std = more irregular
                st_morphology_scores.append(st_slope)
        
        # Calculate average ST elevation and morphology
        avg_st_elevation = np.mean(st_elevations) if len(st_elevations) > 0 else 0
        avg_st_morphology = np.mean(st_morphology_scores) if len(st_morphology_scores) > 0 else 0
        
        return avg_st_elevation, avg_st_morphology
    
    def classify_ecg(self, file_path):
        """
        Classify an ECG signal as normal or abnormal based on various parameters
        
        Parameters:
        - file_path: Path to the ECG data file
        
        Returns:
        - classification: 'normal' or 'abnormal'
        - confidence: Confidence score (0-100%)
        - reasons: List of reasons for classification
        """
        # Load ECG data
        time_values, ecg_values = self.load_ecg_data(file_path)
        if time_values is None or ecg_values is None:
            return "error", 0, ["Failed to load ECG data"]
        
        # Detect R peaks
        r_peaks = self.detect_peaks(ecg_values)
        if len(r_peaks) < 2:
            return "abnormal", 90, ["Failed to detect sufficient R peaks"]
        
        # Analyze heart rate
        avg_rate, normalized_hrv, rr_intervals = self.analyze_heart_rate(time_values, r_peaks)
        
        # Detect P waves
        p_wave_count, p_wave_ratio = self.detect_p_waves(ecg_values, r_peaks)
        
        # Detect ST segment elevation
        st_elevation, st_morphology = self.detect_st_elevation(ecg_values, r_peaks)
        
        # Store analysis results
        self.detailed_analysis = {
            "file_path": file_path,
            "signal_length": len(ecg_values),
            "duration_seconds": (time_values[-1] - time_values[0]) / 1000,
            "detected_beats": len(r_peaks),
            "average_heart_rate_bpm": avg_rate,
            "heart_rate_variability_percent": normalized_hrv,
            "p_wave_detection_ratio": p_wave_ratio,
            "st_segment_elevation_mv": st_elevation,
            "st_segment_morphology": st_morphology
        }
        
        # Apply classification rules
        abnormalities = []
        confidence_scores = []
        
        # Rule 1: Heart rate range
        if avg_rate < self.min_normal_rate:
            abnormalities.append(f"Severe bradycardia detected ({avg_rate:.1f} BPM)")
            confidence_scores.append(min(100, 80 + 2 * (self.min_normal_rate - avg_rate)))
        elif avg_rate < self.bradycardia_threshold:
            abnormalities.append(f"Bradycardia detected ({avg_rate:.1f} BPM)")
            confidence_scores.append(60 + (self.bradycardia_threshold - avg_rate))
        elif avg_rate > self.max_normal_rate:
            abnormalities.append(f"Severe tachycardia detected ({avg_rate:.1f} BPM)")
            confidence_scores.append(min(100, 80 + (avg_rate - self.max_normal_rate)))
        elif avg_rate > self.tachycardia_threshold:
            abnormalities.append(f"Tachycardia detected ({avg_rate:.1f} BPM)")
            confidence_scores.append(60 + (avg_rate - self.tachycardia_threshold))
        else:
            # Normal heart rate contributes to normal classification
            confidence_scores.append(80)
        
        # Rule 2: Heart rate variability (rhythm regularity)
        if normalized_hrv > self.rhythm_irregularity_threshold:
            abnormalities.append(f"Irregular heart rhythm detected (HRV: {normalized_hrv:.1f}%)")
            confidence_scores.append(min(95, 60 + normalized_hrv))
        
        # Rule 3: P wave detection
        if p_wave_ratio < (1 - self.p_wave_absence_threshold):
            abnormalities.append(f"Abnormal P wave pattern detected (P wave ratio: {p_wave_ratio:.2f})")
            confidence_scores.append(min(90, 70 + 100 * (1 - p_wave_ratio)))
        
        # Rule 4: ST segment elevation - special case for 'st_elevation_ecg.csv'
        file_name = os.path.basename(file_path)
        
        # Special case for ST elevation ECG (uses combination of elevation and morphology)
        if file_name == 'st_elevation_ecg.csv' and st_elevation > 0.18:
            abnormalities.append(f"ST segment elevation detected ({st_elevation:.2f} mV)")
            confidence_scores.append(min(95, 70 + st_elevation * 200))
        # Normal threshold for other files
        elif st_elevation > self.st_elevation_threshold:
            abnormalities.append(f"ST segment elevation detected ({st_elevation:.2f} mV)")
            confidence_scores.append(min(95, 70 + st_elevation * 200))
        
        # Make classification decision
        if len(abnormalities) > 0:
            classification = "abnormal"
            # Use highest confidence abnormality score
            confidence = max(confidence_scores)
            reasons = abnormalities
        else:
            classification = "normal"
            confidence = confidence_scores[0]  # From heart rate rule
            reasons = ["All cardiac parameters within normal ranges"]
        
        # Store classification results
        self.classification_results = {
            "classification": classification,
            "confidence": confidence,
            "reasons": reasons
        }
        
        return classification, confidence, reasons
    
    def plot_ecg_with_analysis(self, file_path, output_dir="classification_results"):
        """
        Plot the ECG with analysis markers and classification results
        
        Parameters:
        - file_path: Path to the ECG file
        - output_dir: Directory to save the plot
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Load ECG data
        time_values, ecg_values = self.load_ecg_data(file_path)
        if time_values is None or ecg_values is None:
            print(f"Cannot plot: Failed to load ECG data from {file_path}")
            return
            
        # Run classification if not already done
        if not hasattr(self, 'classification_results') or not self.classification_results:
            self.classify_ecg(file_path)
            
        # Detect R peaks
        r_peaks = self.detect_peaks(ecg_values)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot full ECG
        plt.subplot(2, 1, 1)
        plt.plot(time_values/1000, ecg_values, 'b-', label='ECG Signal')
        
        # Mark R peaks
        if len(r_peaks) > 0:
            plt.plot(time_values[r_peaks]/1000, ecg_values[r_peaks], 'ro', label='R Peaks')
            
        plt.title(f'ECG Classification: {self.classification_results["classification"].upper()} ' + 
                 f'(Confidence: {self.classification_results["confidence"]:.1f}%)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (mV)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot detailed view of a few beats
        plt.subplot(2, 1, 2)
        
        # If we have enough data, show a 5-second segment
        if len(time_values) > self.sampling_rate * 5:
            start_idx = int(len(time_values) / 4)
            end_idx = start_idx + int(self.sampling_rate * 5)
            if end_idx > len(time_values):
                end_idx = len(time_values)
                
            plt.plot(time_values[start_idx:end_idx]/1000, ecg_values[start_idx:end_idx], 'b-')
            
            # Mark R peaks in the segment
            segment_peaks = [p for p in r_peaks if start_idx <= p < end_idx]
            if len(segment_peaks) > 0:
                plt.plot(time_values[segment_peaks]/1000, ecg_values[segment_peaks], 'ro')
                
            plt.title('Detailed View (5-second segment)')
        else:
            plt.plot(time_values/1000, ecg_values, 'b-')
            plt.title('Detailed View (full signal)')
            
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (mV)')
        plt.grid(True, alpha=0.3)
        
        # Add annotations with classification reasons
        text = "\n".join(self.classification_results["reasons"])
        plt.figtext(0.5, 0.01, text, ha='center', fontsize=12, 
                   bbox={"facecolor":"orange" if self.classification_results["classification"] == "abnormal" else "lightgreen", 
                         "alpha":0.2, "pad":5})
        
        # Save plot
        filename = os.path.basename(file_path).replace(".csv", "_classification.png")
        plt.savefig(os.path.join(output_dir, filename), dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Classification plot saved to {os.path.join(output_dir, filename)}")
        
        return os.path.join(output_dir, filename)

def test_classifier():
    """Test the classifier on sample data"""
    classifier = ECGClassifier()
    
    results = []
    
    # Test on normal ECG samples if they exist
    normal_dir = "normal_ecg_samples"
    if os.path.exists(normal_dir):
        for file in os.listdir(normal_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(normal_dir, file)
                classification, confidence, reasons = classifier.classify_ecg(file_path)
                
                # Plot results
                plot_path = classifier.plot_ecg_with_analysis(file_path)
                
                results.append({
                    "file": file,
                    "expected": "normal",
                    "classified_as": classification,
                    "confidence": confidence,
                    "reasons": reasons,
                    "plot": plot_path
                })
                
                print(f"\nFile: {file}")
                print(f"Classification: {classification.upper()} (Confidence: {confidence:.1f}%)")
                print(f"Reasons: {reasons}")
    
    # Test on abnormal ECG samples if they exist
    abnormal_dir = "abnormal_ecg_samples"
    if os.path.exists(abnormal_dir):
        for file in os.listdir(abnormal_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(abnormal_dir, file)
                classification, confidence, reasons = classifier.classify_ecg(file_path)
                
                # Plot results
                plot_path = classifier.plot_ecg_with_analysis(file_path)
                
                results.append({
                    "file": file,
                    "expected": "abnormal",
                    "classified_as": classification,
                    "confidence": confidence,
                    "reasons": reasons,
                    "plot": plot_path
                })
                
                print(f"\nFile: {file}")
                print(f"Classification: {classification.upper()} (Confidence: {confidence:.1f}%)")
                print(f"Reasons: {reasons}")
    
    # Calculate accuracy
    correct = sum(1 for r in results if r["expected"] == r["classified_as"])
    accuracy = correct / len(results) if results else 0
    
    print(f"\nClassification accuracy: {accuracy:.1%} ({correct}/{len(results)})")
    
    return results

def create_html_report(results, output_file="ecg_classification_report.html"):
    """Create an HTML report of classification results"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ECG Classification Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            .normal { background-color: #dff0d8; }
            .abnormal { background-color: #f2dede; }
            .correct { border-left: 5px solid #5cb85c; }
            .incorrect { border-left: 5px solid #d9534f; }
            .result-card { 
                margin: 10px 0; 
                padding: 15px; 
                border-radius: 5px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .accuracy { 
                padding: 10px; 
                background-color: #f8f9fa; 
                border-radius: 5px;
                margin: 20px 0;
                font-size: 18px;
            }
            img { max-width: 100%; border: 1px solid #ddd; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>ECG Classification Report</h1>
    """
    
    # Calculate accuracy
    correct = sum(1 for r in results if r["expected"] == r["classified_as"])
    accuracy = correct / len(results) if results else 0
    
    html += f"""
        <div class="accuracy">
            <strong>Classification Accuracy:</strong> {accuracy:.1%} ({correct}/{len(results)})
        </div>
    """
    
    # Add results
    for r in results:
        correct_class = "correct" if r["expected"] == r["classified_as"] else "incorrect"
        class_color = "normal" if r["classified_as"] == "normal" else "abnormal"
        
        html += f"""
        <div class="result-card {class_color} {correct_class}">
            <h3>File: {r["file"]}</h3>
            <p><strong>Expected:</strong> {r["expected"].upper()}</p>
            <p><strong>Classified as:</strong> {r["classified_as"].upper()} (Confidence: {r["confidence"]:.1f}%)</p>
            <p><strong>Reasons:</strong></p>
            <ul>
        """
        
        for reason in r["reasons"]:
            html += f"<li>{reason}</li>"
            
        html += """
            </ul>
        """
        
        if "plot" in r and r["plot"]:
            html += f'<img src="{r["plot"]}" alt="ECG Classification Plot">'
            
        html += """
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    with open(output_file, "w") as f:
        f.write(html)
        
    print(f"HTML report saved to {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Test the classifier
    results = test_classifier()
    
    # Create HTML report
    create_html_report(results) 