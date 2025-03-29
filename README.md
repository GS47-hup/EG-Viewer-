# ECG Viewer Project

## Overview
The ECG Viewer is a comprehensive electrocardiogram (ECG) visualization, simulation, and analysis tool for healthcare professionals and researchers.

## Features

### Visualization
- Real-time ECG signal display with dynamic waveform rendering
- Automatic R-peak detection and heart rate calculation
- Multiple visualization modes (static, scrolling, continuous)

### Simulation
- Generation of synthetic ECG patterns for different cardiac conditions
- Adjustable parameters (heart rate, noise level)
- Customizable waveform morphology

### Real-world Data Integration
- Import and visualization of real ECG recordings
- Continuous playback of static ECG samples
- Sample selection interface for navigating multiple records

### Analysis and Classification
- Automated ECG interpretation with abnormality detection
- Classification of cardiac conditions (AFib, ST elevation, etc.)
- Detailed analysis reports

## Getting Started

### Installation
1. Clone the repository
2. Ensure dependencies are installed (PyQt5, numpy, pandas, matplotlib, scipy)
3. Run the application:
   ```
   python standalone_ecg_simulator.py
   ```

### Using the Application
- Use "Start Monitoring" to begin real-time ECG simulation
- Select ECG type from the dropdown menu
- Adjust heart rate and noise levels using sliders
- Click "Classify ECG" to perform automated interpretation
- Use "Load Real-World ECG" to import and visualize real ECG data
- With real-world data loaded, use "Play as Continuous" for real-time scrolling display

## Evaluating Classifier Performance

To evaluate the performance of our current ECG classifier, use the evaluation script:

```
python evaluate_current_classifier.py
```

This script provides:
- Performance metrics (accuracy, precision, recall, F1 score)
- Confusion matrix visualization
- Classification report
- Distribution of confidence scores and classification reasons

The evaluation can be performed on:
1. Real-world ECG data with manual labeling
2. Pre-labeled datasets
3. Synthetic test data

## Future Enhancements

### Machine Learning Integration
We are working on integrating machine learning-based classification using pre-trained models from repositories like [ECG-Arrhythmia-Classifier](https://github.com/Tobai24/ECG-Arrhythmia-Classifier). This will enhance our classification accuracy and provide more advanced arrhythmia detection capabilities.

### Multi-lead ECG Support
Future versions will support 12-lead ECG visualization and analysis.

### Clinical Decision Support
Enhanced integration with medical knowledge bases and treatment recommendation systems.

## ECG Simulator

A standalone ECG (Electrocardiogram) visualization and simulation tool that allows viewing and analysis of ECG signals.

## Features

- Load and visualize real or synthetic ECG data
- Interactive ECG signal display with R-peak detection
- Real-time ECG signal simulation with adjustable speed
- Classification of ECG signals as normal or abnormal
- Generate synthetic ECG data for testing (normal ECG, atrial fibrillation, ST elevation)

## Requirements

- Python 3.7+
- Dependencies:
  - NumPy
  - Pandas
  - Matplotlib
  - PyQt5
  - SciPy

## Installation

1. Clone this repository or download the source code
2. Install required dependencies:

```
pip install numpy pandas matplotlib PyQt5 scipy
```

## Usage

### Generating Sample ECG Data

The `sample_data_generator.py` script creates synthetic ECG data for testing the simulator:

```
python sample_data_generator.py
```

This will:
- Generate a full dataset with 140 normal and 50 abnormal ECG samples in `data/synthetic_ecg.csv`
- Generate a small test dataset with 20 normal and 10 abnormal samples in `Real ECG.csv`
- Create visual examples of the different ECG types in `sample_plots/sample_ecgs.png`

### Running the ECG Simulator

To launch the ECG simulator:

```
python standalone_ecg_simulator.py
```

The simulator will:
- Look for the `Real ECG.csv` file by default to load ECG samples
- If the file is not found, it will prompt you to select an ECG file
- Allow you to load normal, abnormal, or random ECG samples
- Enable real-time simulation of the ECG signal with adjustable speed
- Provide an option to classify the current ECG signal

## Simulator Controls

- **Load Normal ECG**: Loads a random normal ECG sample
- **Load Abnormal ECG**: Loads a random abnormal ECG sample
- **Load Random ECG**: Loads any random ECG sample
- **Start Simulation**: Begins real-time simulation of the loaded ECG
- **Stop Simulation**: Stops the current simulation
- **Simulation Speed**: Adjusts the playback speed of the simulation
- **Classify ECG**: Performs classification on the current ECG sample

## ECG Data Format

The ECG data is expected to be in CSV format:
- Multiple columns of ECG data points
- The last column contains the class label (0 for normal, 1 for abnormal)

## Synthetic ECG Types

The sample data generator can create three types of ECG patterns:

1. **Normal ECG**: Regular heart rhythm with normal P, QRS, and T waves
2. **Atrial Fibrillation ECG**: Irregular heart rhythm, absence of P waves, and fibrillatory waves
3. **ST Elevation ECG**: Normal rhythm with elevated ST segment, indicative of myocardial infarction

## Customization

You can customize the sample data generation by modifying the parameters in `sample_data_generator.py`:
- Number of samples to generate
- Duration of each ECG recording
- Sampling rate
- Heart rate ranges
- Noise levels
- ST elevation amount (for abnormal samples)

## License

This project is open-source and free to use for educational and research purposes.

## Acknowledgments

This simulator was designed for educational purposes to help understand ECG signal processing and analysis. 