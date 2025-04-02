#!/usr/bin/env python
"""
Integration Script for ECG-Viewer

This script shows how to integrate the ML Model 2.0 classifier into the
existing ECG-Viewer application.
"""

import sys
import os

# Import the ML classifier UI module
from ml_classifier_ui import MLClassifierUI

def integrate_ml_classifier():
    """
    Main function to integrate the ML Model 2.0 classifier with the ECG-Viewer.
    
    Returns:
        str: Status message
    """
    # Check if the ML model exists
    model_paths = [
        os.path.join('ECG-Arrhythmia-Classifier-main', 'notebooks', 'model.pkl'),
        os.path.join('ECG-Arrhythmia-Classifier-main', 'deployment', 'web_deployment', 'model.pkl')
    ]
    
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            model_found = True
            break
    
    if not model_found:
        return "Error: ML Model 2.0 not found. Please ensure the ECG-Arrhythmia-Classifier-main directory contains model.pkl"
    
    print("ML Model 2.0 found!")
    
    # To integrate with the existing ECG-Viewer, you need to:
    #
    # 1. Import the MLClassifierUI class in the main ECG-Viewer file
    # 2. Initialize the UI in the ECG-Viewer's __init__ method
    # 3. Call the classify_current_ecg method when needed
    
    # Here's an example code snippet to add to your ECG-Viewer:
    
    sample_code = """
    # Add this to your imports
    from ml_classifier_ui import MLClassifierUI
    
    # Add this to your ECGViewer class __init__ method
    self.ml_classifier_ui = MLClassifierUI(self)
    
    # Modify your existing classify method to use the ML Model 2.0 when enabled
    def classify_ecg(self):
        # Get current ECG data
        ecg_values = self.value_history[:self.capture_index]
        time_values = self.time_history[:self.capture_index]
        
        # If ML Model 2.0 is enabled, use it
        if self.ml_classifier_ui.ml_enabled:
            # Use the advanced ML Model 2.0 (98.8% accuracy)
            result = self.ml_classifier_ui.classify_current_ecg(ecg_values, time_values)
            # Handle result as needed
            if result['success']:
                print(f"ML Model 2.0 Classification: {result['class']} (Confidence: {result['confidence']:.2f})")
            else:
                print(f"ML Model 2.0 classification failed: {result.get('error', 'Unknown error')}")
        else:
            # Use the original rule-based classifier
            # ... your existing classification code ...
            pass
    """
    
    print("\nTo integrate the ML Model 2.0 classifier into your ECG-Viewer:")
    print("1. Make sure the following files are in your project:")
    print("   - integrate_ml_classifier.py")
    print("   - ml_classifier_ui.py")
    print("   - ECG-Arrhythmia-Classifier-main/ directory with model.pkl")
    print("\n2. Add the following code to your main ECG-Viewer file:")
    print(sample_code)
    
    return "ML Model 2.0 integration code generated successfully!"

# Example of how to modify standalone_ecg_simulator.py to use ML classifier
def modify_simulator_example():
    """
    Return example code for modifying the standalone_ecg_simulator.py
    to include the ML Model 2.0 classifier.
    """
    code = """
# In standalone_ecg_simulator.py, add the following:

# 1. Add import at the top
from ml_classifier_ui import MLClassifierUI

# 2. In the ECGSimulator class __init__ method, add:
self.ml_classifier_ui = MLClassifierUI(self)

# 3. Add a new ML classify button to the layout
self.button_ml_classify = QtWidgets.QPushButton("ML Model 2.0 Classify")
self.button_ml_classify.clicked.connect(self.ml_classify_ecg)
self.layout.addWidget(self.button_ml_classify, row, col)

# 4. Add the ML classification method
def ml_classify_ecg(self):
    # Get current ECG data
    if self.current_ecg is not None:
        # Extract the signal data (excluding the label)
        ecg_signal = self.current_ecg[:, :-1] if self.current_ecg.shape[1] > 1 else self.current_ecg
        
        # Use the advanced ML Model 2.0 classifier
        result = self.ml_classifier_ui.classify_current_ecg(ecg_signal.flatten())
        
        # Show result in a message box
        if result['success']:
            msg = f"ML Model 2.0 Classification Result:\\n\\n"
            msg += f"Class: {result['class']}\\n"
            msg += f"Confidence: {result['confidence']:.4f}\\n\\n"
            msg += f"Note: This uses the advanced ML model with 98.8% accuracy"
            
            QtWidgets.QMessageBox.information(self, "ML Model 2.0 Classification", msg)
        else:
            QtWidgets.QMessageBox.warning(
                self, 
                "ML Model 2.0 Classification Failed", 
                f"Error: {result.get('error', 'Unknown error')}"
            )
    else:
        QtWidgets.QMessageBox.warning(
            self, 
            "No ECG Data", 
            "Please load an ECG sample before classifying."
        )
"""
    return code

if __name__ == "__main__":
    status = integrate_ml_classifier()
    print("\n" + status)
    
    print("\nExample code for modifying standalone_ecg_simulator.py:")
    print(modify_simulator_example()) 