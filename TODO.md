# ECG Viewer Project - TODO List

## Project Goal
Develop an ECG signal classification system that can determine if a heart's ECG signal is "good" or "bad" by analyzing the waveform patterns.

## Implementation Approach
1. Start with rule-based classification as proof of concept
2. Create simulations of good and bad ECG signals for testing
3. Later expand to feature-based classification with machine learning

## To Do
- Test the rule-based classification system with real ECG data from Arduino
- Refine detection thresholds based on real-world testing
- Implement logging of classification results for later analysis
- Expand classification to detect more specific cardiac abnormalities
- Prepare for feature-based classification:
  - Research ECG feature extraction techniques
  - Identify machine learning approach appropriate for ECG classification
  - Plan data collection strategy for training set

## Completed
- Connected Arduino with ECG sensor to laptop
- Verified ECG signal visualization in application
- Tested reading real ECG signals from body
- Implemented basic heart rate calculation
- Created simulation data for "good" ECG signals with normal characteristics
- Created simulation data for "bad" ECG signals with various abnormalities
- Implemented rule-based classification system:
  - Defined rules for normal vs. abnormal ECG characteristics
  - Implemented detection for heart rate abnormalities (bradycardia, tachycardia)
  - Implemented detection for rhythm irregularities
  - Implemented detection for P wave abnormalities
  - Implemented detection for ST segment elevation
- Added classification output display to ECG viewer UI
- Created integrated classification reporting with visual feedback
- Added HTML report generation for classification results

## Notes
- Rule-based classification is simpler but less comprehensive than ML-based approaches
- Common ECG abnormalities detected: arrhythmia, tachycardia, bradycardia, ST elevation, missing P waves
- Key testing files created:
  - create_normal_ecg.py: Generates simulation data for normal ECGs
  - create_abnormal_ecg.py: Generates simulation data for abnormal ECGs
  - ecg_classifier.py: Main classification system implementation
  - ecg_viewer_with_classification.py: Integration with the ECG viewer UI
- When testing with real people eventually, ensure proper informed consent and privacy
- Future work could include adding more sophisticated morphology analysis of ECG waveforms