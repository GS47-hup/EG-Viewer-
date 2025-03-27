import pandas as pd
import numpy as np
import os
import sys

def convert_single_row_ecg_to_time_series(input_file, output_file, sampling_rate_hz=250):
    """
    Convert a single-row ECG file (with one value per column) to a two-column time series format.
    
    Parameters:
    - input_file: Path to the input CSV file with a single row of ECG values
    - output_file: Path to save the output CSV file with time and value columns
    - sampling_rate_hz: Sampling rate in Hz (samples per second), default is 250 Hz
    """
    print(f"Converting {input_file} to time-series format...")
    
    try:
        # Read the single-row CSV file
        # Note: header=None assumes there are no column headers
        data = pd.read_csv(input_file, header=None)
        
        # Extract all values into a single list
        if data.shape[0] == 1:
            # File has one row with multiple columns
            ecg_values = data.iloc[0, :].tolist()
        else:
            # File might have one column with multiple rows
            ecg_values = data.iloc[:, 0].tolist()
        
        # Generate time values based on sampling rate
        # Time interval between samples in milliseconds
        time_interval_ms = 1000 / sampling_rate_hz
        time_values = [i * time_interval_ms for i in range(len(ecg_values))]
        
        # Create a new DataFrame with time and value columns
        result_df = pd.DataFrame({
            'time': time_values,
            'value': ecg_values
        })
        
        # Save to the output file
        result_df.to_csv(output_file, index=False)
        print(f"Conversion complete! Saved to {output_file}")
        print(f"Total samples: {len(ecg_values)}")
        print(f"Duration: {len(ecg_values)/sampling_rate_hz:.2f} seconds at {sampling_rate_hz} Hz")
        
        return True
    
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_ecg.py <input_file> [output_file] [sampling_rate_hz]")
        print("Example: python convert_ecg.py my_ecg_export.csv sample_ecg.csv 250")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Default output filename is sample_ecg.csv (to match what the app looks for)
    output_file = "sample_ecg.csv"
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    
    # Default sampling rate is 250 Hz
    sampling_rate = 250
    if len(sys.argv) >= 4:
        sampling_rate = int(sys.argv[3])
    
    success = convert_single_row_ecg_to_time_series(input_file, output_file, sampling_rate)
    
    if success:
        print("\nYou can now run the ECG Viewer in simulation mode! It will automatically use this file.")
    else:
        print("\nConversion failed. Please check your input file format.") 