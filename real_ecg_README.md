# Real ECG Data Classifier

This project implements a machine learning classifier for real ECG data, achieving 97% accuracy on test data. The classifier can distinguish between normal and abnormal ECG patterns using advanced feature extraction techniques.

## Dataset

The classifier uses a real ECG dataset with the following characteristics:
- Each ECG sample consists of 140 data points
- Labels indicate normal (0) or abnormal (1) heart conditions
- Total of 4998 samples (2079 normal, 2919 abnormal)

## Features

The classifier extracts the following features from ECG signals:
- Statistical features (mean, standard deviation, min, max, range)
- Frequency domain features (dominant frequency, energy in different frequency bands)
- Wavelet transform coefficients for multi-scale analysis
- R-peak related features (heart rate, RR intervals, RR variability)
- ST segment elevation measurement

## Performance

The classifier achieves excellent performance metrics:
- Accuracy: 97.3%
- Sensitivity: 98.5% (ability to detect abnormal ECGs)
- Specificity: 95.7% (ability to correctly identify normal ECGs)
- ROC AUC: 0.995

## Model Comparison

Multiple machine learning models were evaluated:
| Model | Accuracy | Sensitivity | Specificity | ROC AUC |
|-------|----------|-------------|-------------|---------|
| Random Forest | 97.4% | 98.3% | 96.2% | 0.995 |
| Gradient Boosting | 97.5% | 99.0% | 95.4% | 0.996 |
| SVM | 97.4% | 99.3% | 94.7% | 0.993 |
| Neural Network | 97.5% | 97.3% | 97.8% | 0.996 |
| K-Nearest Neighbors | 97.1% | 99.3% | 94.0% | 0.985 |

## Scripts

### 1. real_ecg_classifier.py
- Main classifier implementation
- Feature extraction and model training
- Evaluation on validation and test sets

```
python real_ecg_classifier.py
```

### 2. evaluate_real_ecg_models.py
- Evaluates multiple machine learning models
- Compares performance metrics
- Generates visualization of model comparison

```
python evaluate_real_ecg_models.py
```

### 3. test_real_ecg_model.py
- Tests a trained model on new data
- Provides predictions for individual ECG samples
- Visualizes classification results

```
python test_real_ecg_model.py --data Real_ECG.csv --model real_ecg_data/best_model.joblib
python test_real_ecg_model.py --sample 123  # Predict a specific sample
```

### 4. analyze_misclassifications.py
- Analyzes misclassified samples
- Visualizes feature distributions
- Provides suggestions for model improvement

```
python analyze_misclassifications.py --model real_ecg_data/real_ecg_classifier.joblib
```

## Key Findings

1. The most important features for classification were:
   - Maximum amplitude (0.271)
   - Signal range (0.224)
   - Wavelet energy at different scales (0.103)
   - Heart rate and RR interval variability features

2. Common misclassification patterns:
   - False positives (normal ECGs classified as abnormal) were often associated with higher amplitude values
   - False negatives (abnormal ECGs classified as normal) often had subtle ST segment abnormalities

3. Improvement suggestions:
   - Adjust feature thresholds for the most problematic features
   - Add more specialized features for detecting subtle abnormalities
   - Consider ensemble methods to combine multiple classifiers

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy
- PyWavelets
- Seaborn (for visualization)

## Future Work

- Incorporate additional ECG lead data for more comprehensive analysis
- Implement deep learning approaches (CNN, LSTM) for direct signal processing
- Develop explanations for abnormal classifications to aid clinical interpretation
- Create a web interface for interactive ECG analysis 