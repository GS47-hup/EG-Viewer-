# ECG-Viewer Branch Notes

This document provides information about the various branches in the ECG-Viewer repository, their purpose, and key features.

## Table of Contents
- [Main Branch](#main-branch)
- [Stable Releases](#stable-releases)
- [ML Integration Branches](#ml-integration-branches)
- [Classifier Implementation Branches](#classifier-implementation-branches)
- [Feature Branches](#feature-branches)
- [Experimental Branches](#experimental-branches)

## Main Branch

### `main`
- **Purpose**: Primary development branch
- **Features**: Core ECG viewer functionality
- **Status**: Stable, base application

## Stable Releases

### `v0.1.10-stable`
- **Purpose**: Stable release version 0.1.10
- **Features**: Core ECG viewer functionality with stable UI
- **Status**: Production-ready version

## ML Integration Branches

### `ml-integration-improvements`
- **Purpose**: Ongoing improvements to ML integration
- **Features**:
  - Improved demo mode classifier (accuracy: ~61%)
  - Better heart rate and RR variability calculations
  - Enhanced UI indicators for demo mode
  - Proper fallback when ML model can't be loaded
  - Integration with standalone ECG simulator
- **Status**: Active development, currently stable
- **Key Commits**:
  - "Improve ML Model 2.0 integration with demo mode clarity and better UI"
  - "Fix statusBar access in ML toggle method"
  - "Fix ML Model demo mode to ensure it works reliably"
  - "Improve demo classifier accuracy to 80.97% on real-world ECG data"
  - "Fix heart rate and RR variability calculation in demo classifier"

### `ml-integration-working-state`
- **Purpose**: Preserved working state of ML integration
- **Features**:
  - Basic ML model integration
  - Simple demo mode
  - Working UI components
- **Status**: Stable reference point

### `integrate-arrhythmia-classifier`
- **Purpose**: Initial integration of arrhythmia classifier
- **Features**:
  - Basic integration of ECG-Arrhythmia-Classifier
  - Initial UI for ML features
- **Status**: Merged into ml-integration branches

## Classifier Implementation Branches

### `ml-based-classifier`
- **Purpose**: Implementation of ML-based classification
- **Features**:
  - Trained ML model for ECG classification
  - Model loading and inference
- **Status**: Complete

### `rule-based-classifier`
- **Purpose**: Implementation of rule-based classification
- **Features**:
  - Signal processing based detection
  - Heart rate and arrhythmia detection rules
- **Status**: Complete, used as fallback in demo mode

### `ecg-classifier-improvements`
- **Purpose**: Improvements to ECG classifier
- **Features**:
  - Enhanced classification accuracy
  - Additional detection methods
- **Status**: Complete

## Feature Branches

### `feature/ml-classification`
- **Purpose**: Development of ML classification features
- **Features**:
  - Classification UI
  - Result visualization
- **Status**: Merged into integration branches

## Experimental Branches

### `experiment`
- **Purpose**: Experimental features and concepts
- **Features**:
  - Various experimental UI elements
  - Test implementations
- **Status**: Ongoing, not for production

### `future-improvements`
- **Purpose**: Planned future improvements
- **Features**:
  - Draft implementations of future features
  - Advanced visualization concepts
- **Status**: In planning/development

## Working with Branches

### Switching Branches
```bash
git checkout branch-name
```

### Creating a New Branch from Current Branch
```bash
git checkout -b new-branch-name
```

### Merging Changes from Another Branch
```bash
git merge other-branch-name
```

## Testing Branches

To test a particular branch's functionality:

1. Checkout the branch: `git checkout branch-name`
2. Run the application: `python standalone_ecg_simulator.py`
3. For ML classifier testing: `python test_demo_classifier_real.py`

## Notes for Contributors

When working on the ML integration:

1. The ML classifier is implemented in `integrate_ml_classifier.py`
2. The UI components are in `ml_classifier_ui.py`
3. Demo mode is activated when the real model cannot be loaded
4. Test thoroughly with various ECG types before committing

## Current Development Focus

The current focus is on improving the ML integration, particularly:

1. Enhancing the demo classifier's accuracy
2. Improving the UI for ML results
3. Better error handling and user guidance
4. Integrating with additional ECG data sources 