# Machine Learning-Based ECG Classifier

This branch contains the machine learning approach to ECG signal classification. This classifier uses advanced ML algorithms trained on labeled ECG data to identify normal and abnormal patterns.

## Key Files

- `real_ecg_classifier.py` - The main ML classifier implementation
- `evaluate_real_ecg_models.py` - Script to evaluate various ML models
- `test_real_ecg_model.py` - Script for testing the classifier on new data
- `analyze_misclassifications.py` - Script to analyze misclassified samples
- `real_ecg_data/` - Directory containing trained models and analysis results

## Features

The ML-based classifier extracts various features from ECG signals:
- Statistical features (mean, standard deviation, min, max, range)
- Frequency domain features (dominant frequency, energy in different frequency bands)
- Wavelet transform coefficients for multi-scale analysis
- R-peak related features (heart rate, RR intervals, RR variability)
- ST segment elevation measurement

## Advantages

- High accuracy (97.3% reported on test data)
- Excellent sensitivity (98.5%) and specificity (95.7%)
- Can detect subtle patterns that might be missed by rule-based approaches
- Performance improves as more training data becomes available

## Performance

The classifier achieves excellent performance metrics:
- Accuracy: 97.3%
- Sensitivity: 98.5% (ability to detect abnormal ECGs)
- Specificity: 95.7% (ability to correctly identify normal ECGs)
- ROC AUC: 0.995

## Usage

```python
from real_ecg_classifier import RealEcgClassifier

# Load a trained model
classifier = RealEcgClassifier.load_model('real_ecg_data/real_ecg_classifier.joblib')

# Predict on new data
prediction = classifier.predict(ecg_data)
```

## Training New Models

You can train new models with:

```
python real_ecg_classifier.py
```

## Evaluating Multiple Models

To compare different ML algorithms:

```
python evaluate_real_ecg_models.py
```

This will generate comparison plots and metrics for various classifiers.

## Dataset

The classifier uses real ECG data with labels (normal/abnormal) for training and evaluation. 