import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Input and output files
input_file = "sample_ecg.csv"  # Original ECG data file
output_file = "sample_ecg_mv_original.csv"  # Converted millivolt ECG data

# Arduino ADC reference points
ADC_MAX = 1023.0  # Maximum ADC value (10-bit ADC on Arduino)
ADC_REFERENCE = 5.0  # Arduino reference voltage in volts

# ECG signal parameters
ECG_MV_MAX = 1.0  # Maximum ECG amplitude in millivolts (typical R peak)

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file {input_file} does not exist")
    print("Please provide the correct path to your original ECG sample file")
    exit(1)

# Read the original ECG data
try:
    data = pd.read_csv(input_file)
    print(f"Successfully read {len(data)} samples from {input_file}")
except Exception as e:
    print(f"Error reading {input_file}: {e}")
    exit(1)

# Display columns in the file
print(f"Columns in the original file: {', '.join(data.columns)}")

# Determine if we need to convert time column
has_time = 'time' in data.columns
value_column = 'value' if 'value' in data.columns else data.columns[0]

# Extract the ECG data values
ecg_values = data[value_column].values

# Get the data range and statistics
adc_min = np.min(ecg_values)
adc_max = np.max(ecg_values)
adc_range = adc_max - adc_min
adc_mean = np.mean(ecg_values)

print(f"Original ADC value statistics:")
print(f"- Min: {adc_min:.1f}")
print(f"- Max: {adc_max:.1f}")
print(f"- Range: {adc_range:.1f}")
print(f"- Mean: {adc_mean:.1f}")

# Convert ADC values to millivolts
# Standard conversion: ADC->Volts->Millivolts
# First normalize to 0-1 range based on actual min/max
normalized = (ecg_values - adc_min) / adc_range
# Then scale to millivolts range with baseline at 0
millivolts = normalized * ECG_MV_MAX

print(f"Converted millivolt values:")
print(f"- Min: {np.min(millivolts):.3f}mV")
print(f"- Max: {np.max(millivolts):.3f}mV")
print(f"- Range: {np.max(millivolts) - np.min(millivolts):.3f}mV")
print(f"- Mean: {np.mean(millivolts):.3f}mV")

# Create new dataframe with millivolt values
if has_time:
    df_mv = pd.DataFrame({
        'time': data['time'],
        'value': millivolts
    })
else:
    # Generate time column assuming 250Hz sample rate
    sample_rate = 250
    time = np.arange(len(millivolts)) / sample_rate
    df_mv = pd.DataFrame({
        'time': time,
        'value': millivolts
    })

# Save the converted data
df_mv.to_csv(output_file, index=False)
print(f"Converted data saved to {output_file}")

# Plot the original and converted signals for comparison
plt.figure(figsize=(15, 10))

# Plot original ADC values
plt.subplot(2, 1, 1)
if has_time:
    plt.plot(data['time'], ecg_values)
else:
    plt.plot(ecg_values)
plt.title('Original ECG (ADC values)')
plt.ylabel('ADC Value')
plt.grid(True, alpha=0.3)

# Plot converted millivolt values
plt.subplot(2, 1, 2)
plt.plot(df_mv['time'], df_mv['value'])
plt.title('Converted ECG (millivolts)')
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (mV)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ecg_conversion.png')
print("Conversion comparison plot saved to ecg_conversion.png")

# Update the ECG_serial_handler to prioritize this file
print("\nNext steps:")
print("1. The ECG viewer will automatically use this converted file in millivolts")
print("2. Run the ECG viewer with: python ecg_viewer.py") 