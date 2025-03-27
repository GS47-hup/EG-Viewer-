import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
import argparse

def convert_mitbih_sample(input_file, output_dir, sample_index=0):
    """
    Convert a sample from the MIT-BIH dataset to a format our classifier can use
    
    Args:
        input_file: Path to the MIT-BIH dataset file
        output_dir: Directory to save the converted sample
        sample_index: Index of the sample to convert (default: 0)
    """
    print(f"Loading MIT-BIH dataset from {input_file}...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Load the MIT-BIH dataset
        data = pd.read_csv(input_file, header=None)
        
        # Check if sample_index is valid
        if sample_index >= len(data):
            print(f"Error: Sample index {sample_index} out of range. Dataset has {len(data)} samples.")
            return
        
        # Extract the sample
        sample = data.iloc[sample_index].values
        
        # Remove trailing zeros if any (MIT-BIH data sometimes has padding)
        non_zero_indices = np.where(sample != 0)[0]
        if len(non_zero_indices) > 0:
            last_non_zero = non_zero_indices[-1]
            sample = sample[:last_non_zero + 1]
        
        # Determine if this is likely a normal or abnormal ECG
        # This is a simple heuristic - in real applications, a more sophisticated 
        # analysis would be used
        
        # Compute the standard deviation to detect irregularity
        std_val = np.std(sample)
        max_val = np.max(sample)
        min_val = np.min(sample)
        range_val = max_val - min_val
        
        # Simple peak detection using scipy.signal.find_peaks
        try:
            # Use scipy's find_peaks with appropriate parameters
            peaks, _ = signal.find_peaks(sample, height=0.5, distance=20)
            num_peaks = len(peaks)
            # Estimate heart rate: num_peaks / recording_length_in_sec * 60
            approx_heart_rate = num_peaks / (len(sample) / 250) * 60
            
            print(f"Detected {num_peaks} peaks, approximate heart rate: {approx_heart_rate:.1f} BPM")
            is_tachycardia = approx_heart_rate > 100
            is_bradycardia = approx_heart_rate < 60
            
            # Simple classification rule
            if is_tachycardia or is_bradycardia or std_val > 0.3:
                is_likely_normal = False
            else:
                is_likely_normal = True
                
        except Exception as e:
            print(f"Warning: Peak detection failed: {str(e)}")
            # Fallback to simpler heuristic if peak detection fails
            is_likely_normal = (std_val < 0.3) and (range_val < 0.8)
        
        # Create time values that simulate a realistic ECG
        # Our classifier expects time in ms with typical ECG recording
        # The MIT-BIH data is typically 1-second strips at 125Hz
        # We'll scale it to a realistic duration (8 seconds at 250Hz)
        length = len(sample)
        
        # Make time values suitable for our classifier
        # Simulate a complete ECG recording with multiple beats
        # Use 4ms intervals (250 Hz sampling rate)
        time_ms = np.arange(length) * 4  
        
        # Create a dataframe with time and ECG values
        ecg_df = pd.DataFrame({
            'time': time_ms,
            'value': sample
        })
        
        # Scale the ECG values to a suitable range for our classifier
        # Most ECG values should be in mV range
        if np.max(np.abs(sample)) > 2.0:
            scaling_factor = 1.0 / np.max(np.abs(sample))
            ecg_df['value'] = ecg_df['value'] * scaling_factor
        
        # Generate longer ECG by repeating the pattern
        # This helps the classifier, which expects longer recordings
        # Only do this if the sample is short
        if length < 500:
            repetitions = max(1, int(2000 / length))
            parts = []
            for i in range(repetitions):
                part = ecg_df.copy()
                part['time'] = part['time'] + i * time_ms[-1]
                parts.append(part)
            ecg_df = pd.concat(parts, ignore_index=True)
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"mitbih_sample_{sample_index}.csv")
        ecg_df.to_csv(output_file, index=False)
        
        # Plot the sample for visualization
        plt.figure(figsize=(12, 6))
        plt.plot(ecg_df['time'], ecg_df['value'])
        
        # Plot peaks if detected
        if 'peaks' in locals() and len(peaks) > 0:
            plt.plot(time_ms[peaks], sample[peaks], 'ro', label='Detected Peaks')
            plt.legend()
            
        # Add classification info to title
        if 'approx_heart_rate' in locals():
            plt.title(f"MIT-BIH ECG Sample {sample_index} - Likely {'Normal' if is_likely_normal else 'Abnormal'} - HR: {approx_heart_rate:.1f} BPM")
        else:
            plt.title(f"MIT-BIH ECG Sample {sample_index} - Likely {'Normal' if is_likely_normal else 'Abnormal'}")
            
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (mV)")
        plt.grid(True)
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f"mitbih_sample_{sample_index}.png"))
        plt.close()
        
        print(f"Successfully converted sample {sample_index}")
        print(f"Sample saved to {output_file}")
        print(f"Plot saved to {os.path.join(output_dir, f'mitbih_sample_{sample_index}.png')}")
        print(f"Likely classification: {'Normal' if is_likely_normal else 'Abnormal'}")
        
        return output_file
    
    except Exception as e:
        print(f"Error converting sample: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def convert_multiple_samples(input_file, output_dir, num_samples=5):
    """
    Convert multiple samples from the MIT-BIH dataset
    
    Args:
        input_file: Path to the MIT-BIH dataset file
        output_dir: Directory to save the converted samples
        num_samples: Number of samples to convert
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get total number of samples
    try:
        data = pd.read_csv(input_file, header=None)
        total_samples = len(data)
        
        # Adjust num_samples if it exceeds total_samples
        if num_samples > total_samples:
            print(f"Warning: Requested {num_samples} samples, but dataset only has {total_samples} samples.")
            num_samples = total_samples
        
        # Randomly select sample indices
        indices = np.random.choice(total_samples, num_samples, replace=False)
        
        # Convert each sample
        converted_files = []
        for i, idx in enumerate(indices):
            print(f"\nConverting sample {i+1}/{num_samples} (index {idx})...")
            output_file = convert_mitbih_sample(input_file, output_dir, idx)
            if output_file:
                converted_files.append(output_file)
        
        print(f"\nConverted {len(converted_files)} samples successfully")
        return converted_files
    
    except Exception as e:
        print(f"Error converting multiple samples: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert MIT-BIH ECG samples for classifier testing')
    parser.add_argument('--input', type=str, default='mitbih_test.csv',
                        help='Path to the MIT-BIH dataset file')
    parser.add_argument('--output', type=str, default='mitbih_samples',
                        help='Directory to save converted samples')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of samples to convert')
    parser.add_argument('--index', type=int, default=None,
                        help='Specific sample index to convert (overrides --samples)')
    args = parser.parse_args()
    
    # Convert samples
    if args.index is not None:
        convert_mitbih_sample(args.input, args.output, args.index)
    else:
        convert_multiple_samples(args.input, args.output, args.samples)

if __name__ == "__main__":
    main() 