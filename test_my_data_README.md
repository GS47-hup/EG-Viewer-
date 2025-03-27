# Testing Your ECG Data with the Classifier

This guide explains how to use the ECG classifier we've built to analyze your own ECG data.

## Requirements

- Python 3.6+
- The required packages (install them using `pip install -r requirements.txt`)
- ECG data in CSV format

## Data Format

Your ECG data should be in CSV format with:
- Two columns: time (ms) and ECG signal value (mV)
- First row as header (optional)
- Sampling rate around 250 Hz (samples per second) is ideal

Example of expected CSV format:
```
time,value
0,0.123
4,0.145
8,0.167
...
```

## Running the Classifier

### Testing a Single ECG File

To analyze a single ECG file:

```bash
python test_my_ecg_data.py path/to/your/ecg_file.csv
```

If your data is synthetic (computer-generated):

```bash
python test_my_ecg_data.py path/to/your/ecg_file.csv --synthetic
```

### Testing Multiple ECG Files

To analyze all ECG files in a directory:

```bash
python test_my_ecg_data.py path/to/your/ecg_folder
```

With synthetic data:

```bash
python test_my_ecg_data.py path/to/your/ecg_folder --synthetic
```

## Understanding the Results

The script will:

1. Analyze each ECG file
2. Print classification results to the console
3. Save ECG plots with analysis markers to the `my_analysis_results` folder
4. Create a summary CSV file with all results

### Classification Output

For each file, you'll see:
- Whether the ECG is classified as NORMAL or ABNORMAL
- The confidence level of the classification (0-100%)
- Detailed analysis explaining the classification
- Path to a saved plot file visualizing the ECG with markers

### Summary Statistics

After analyzing all files in a directory, you'll see summary statistics showing:
- Total number of files analyzed
- Number and percentage of normal ECGs
- Number and percentage of abnormal ECGs

## Understanding the Plots

The generated plots show:
- The full ECG signal
- Markers for detected R-peaks (heartbeats)
- Classification result and confidence
- A detailed view of a few heartbeats
- Annotations for any abnormalities detected

## Troubleshooting

If you encounter issues:

1. Ensure your CSV files have the correct format (time, value columns)
2. Check that the sampling rate is appropriate (around 250 Hz)
3. For very noisy signals, try preprocessing your data to remove noise
4. If your ECG data has a different format, you may need to modify the script

## Medical Disclaimer

This classifier is for educational and research purposes only. It is NOT intended for medical diagnosis. Always consult healthcare professionals for medical advice. 