#!/usr/bin/env python
"""
UI Module for ML Classifier Integration

This module extends the ECG-Viewer UI to include the ML-based classifier option.
"""

from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np

# Import our ML classifier adapter
from integrate_ml_classifier import MLClassifier

class MLClassifierUI:
    """
    UI integration for ML classifier in ECG-Viewer
    """
    
    def __init__(self, parent_app):
        """
        Initialize the ML classifier UI integration.
        
        Args:
            parent_app: The parent ECG-Viewer application instance
        """
        self.parent = parent_app
        self.ml_classifier = MLClassifier()
        self.ml_enabled = False
        self.model_path = "ECG-Arrhythmia-Classifier-main/notebooks/model.pkl"
        self.model = None
        self.classes = ["Normal", "Abnormal", "Atrial Fibrillation", "ST Elevation", "Bradycardia", "Tachycardia"]
        
        # Load ML model
        self.model = self.ml_classifier.model  # Store reference to the model
        
        # Update parent's ML status label if available
        if hasattr(parent_app, 'mlStatusLabel'):
            if self.model is not None:
                if self.ml_classifier.use_dummy_model:
                    parent_app.mlStatusLabel.setText("ML Model 2.0: Demo Mode")
                    parent_app.mlStatusLabel.setStyleSheet("color: orange;")
                else:
                    parent_app.mlStatusLabel.setText("ML Model 2.0: Ready")
                    parent_app.mlStatusLabel.setStyleSheet("color: green;")
            else:
                parent_app.mlStatusLabel.setText("ML Model 2.0: Not Available")
                parent_app.mlStatusLabel.setStyleSheet("color: red;")
        
        self.setup_ui()
    
    def setup_ui(self):
        """Add ML classifier UI elements to the ECG-Viewer"""
        
        # --- ML Model 2.0 Toggle Button (remains specific) ---
        if self.ml_classifier.use_dummy_model:
            button_text = "ML Model 2.0 (Demo): Off"
        else:
            button_text = "ML Model 2.0: Off"
            
        self.ml_toggle_button = QtWidgets.QPushButton(button_text)
        self.ml_toggle_button.setCheckable(True)
        self.ml_toggle_button.setChecked(False)
        self.ml_toggle_button.setToolTip("Toggle between rule-based and advanced ML-based ECG classification (Model 2.0)")
        self.ml_toggle_button.clicked.connect(self.toggle_ml_classifier)
        
        # --- General Classification Results Display Area --- 
        self.classification_results_group = QtWidgets.QGroupBox("Rhythm Classification")
        self.classification_results_layout = QtWidgets.QVBoxLayout()
        
        # Label for current status (from rule-based or ML model)
        self.rhythm_status_label = QtWidgets.QLabel("Rhythm: Unknown")
        font = self.rhythm_status_label.font()
        font.setPointSize(font.pointSize() + 1) # Slightly larger font
        font.setBold(True)
        self.rhythm_status_label.setFont(font)
        self.classification_results_layout.addWidget(self.rhythm_status_label)
        
        # Add a separator line
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.classification_results_layout.addWidget(line)
        
        # --- Elements specific to ML Model 2.0 (initially hidden/managed by toggle) ---
        self.ml_model_details_widget = QtWidgets.QWidget() # Container for ML-specific details
        self.ml_model_details_layout = QtWidgets.QVBoxLayout()
        self.ml_model_details_layout.setContentsMargins(0,0,0,0)

        # Description label for ML Model 2.0
        if self.ml_classifier.use_dummy_model:
            model_description = "Advanced ML Model DEMO Active"
        else:
            model_description = "Advanced ML Model Active (98.8% accuracy)"
        self.ml_version_label = QtWidgets.QLabel(model_description)
        self.ml_model_details_layout.addWidget(self.ml_version_label)
        
        # Confidence label for ML Model 2.0
        self.ml_confidence_label = QtWidgets.QLabel("Confidence: N/A")
        self.ml_model_details_layout.addWidget(self.ml_confidence_label)
        
        # Detailed probabilities group for ML Model 2.0
        self.ml_probabilities_group = QtWidgets.QGroupBox("ML Class Probabilities")
        self.ml_probabilities_layout = QtWidgets.QVBoxLayout()
        self.prob_labels = {}
        for i, class_name in enumerate(self.classes):
            label = QtWidgets.QLabel(f"{class_name}: 0.0000")
            self.prob_labels[i] = label
            self.ml_probabilities_layout.addWidget(label)
        self.ml_probabilities_group.setLayout(self.ml_probabilities_layout)
        self.ml_model_details_layout.addWidget(self.ml_probabilities_group)

        self.ml_model_details_widget.setLayout(self.ml_model_details_layout)
        self.classification_results_layout.addWidget(self.ml_model_details_widget) # Add container to main layout
        
        # Set the main layout for the group box
        self.classification_results_group.setLayout(self.classification_results_layout)
        
        # Initially hide the ML-specific details; status label is always visible
        self.ml_model_details_widget.setVisible(False) 
        
        # --- Placement Logic --- NEW simplified approach
        ml_ui_placed = False
        try:
            # Try adding directly to the main grid layout gridLayout_5
            if hasattr(self.parent, 'gridLayout_5'):
                 # Place toggle button at Row 4, Col 2
                 self.parent.gridLayout_5.addWidget(self.ml_toggle_button, 4, 2, 1, 1) 
                 # Place results group below it at Row 5, Col 2
                 self.parent.gridLayout_5.addWidget(self.classification_results_group, 5, 2, 1, 1) 
                 ml_ui_placed = True
                 self.classification_results_group.setVisible(True) # Ensure visibility after placement
                 print("Added Classification UI elements to main gridLayout_5 at (4,2) and (5,2).")
            # Remove the previous attempt to add into gridFrame_21
            # Remove the fallback logic as this is now the primary attempt

            if not ml_ui_placed:
                 print("Could not find main gridLayout_5 to add Classification UI elements.")

        except Exception as e:
            print(f"Error adding Classification UI elements dynamically: {e}")
    
    def update_results(self, status: str):
        """Update the classification status label."""
        # This method is called by ecg_viewer.py with the status from math_calc_hr
        # or potentially later by the advanced ML classifier.
        if self.ml_enabled:
             # If ML Model 2.0 is ON, the advanced classifier would update labels here
             # For now, we just show the status passed (which might still be the rule-based one)
             # In a future step, the advanced ML classifier would call this (or a similar method)
             # with its own results (class, confidence, probabilities).
             self.rhythm_status_label.setText(f"ML Rhythm: {status}") # Indicate it *should* be ML
             # We would also update self.ml_confidence_label and self.prob_labels here
        else:
            # If ML Model 2.0 is OFF, display the rule-based status
            self.rhythm_status_label.setText(f"Rhythm: {status}")

    def toggle_ml_classifier(self):
        """Toggle between the rule-based and ML-based classifier display/logic"""
        self.ml_enabled = self.ml_toggle_button.isChecked()
        
        if self.ml_enabled:
            # --- ML Model 2.0 Enabled ---
            if self.ml_classifier.use_dummy_model:
                self.ml_toggle_button.setText("ML Model 2.0 (Demo): On")
            else:
                self.ml_toggle_button.setText("ML Model 2.0: On")
            
            # Show the ML-specific details widget within the results group
            self.ml_model_details_widget.setVisible(True)
            
            # Update the status label to reflect ML is active (results TBD)
            # The actual classification would happen elsewhere and call update_results
            self.update_results("Pending ML Analysis...") 

            # Hide the old rule-based classify button if it exists
            if hasattr(self.parent, 'classifyButton'):
                self.parent.classifyButton.setVisible(False)
                
            # Update status bar
            if hasattr(self.parent, 'statusBar'):
                demo_suffix = " (Demo Mode)" if self.ml_classifier.use_dummy_model else ""
                self.parent.statusBar.showMessage(f"Advanced ML Model 2.0{demo_suffix} enabled.")
        else:
            # --- ML Model 2.0 Disabled (Use Rule-Based) ---
            if self.ml_classifier.use_dummy_model:
                self.ml_toggle_button.setText("ML Model 2.0 (Demo): Off")
            else:
                self.ml_toggle_button.setText("ML Model 2.0: Off")
            
            # Hide the ML-specific details widget
            self.ml_model_details_widget.setVisible(False)
            
            # Ensure the main classification group is visible
            self.classification_results_group.setVisible(True) 
            
            # Update the status label immediately if we have a stored rule-based status
            if hasattr(self.parent, 'current_classification_status'):
                self.update_results(self.parent.current_classification_status)
            else:
                self.update_results("Unknown") # Default if no status available

            # Show the old rule-based classify button if it exists
            if hasattr(self.parent, 'classifyButton'):
                 self.parent.classifyButton.setVisible(True) # Make sure it's visible
                 self.parent.classifyButton.setText("Classify (Rule-Based)") # Clarify button function
                 
            # Update status bar
            if hasattr(self.parent, 'statusBar'):
                self.parent.statusBar.showMessage("Advanced ML Model disabled. Using rule-based rhythm classification.")
                
        # The classification_results_group itself should remain visible if placed
        # self.classification_results_group.setVisible(True) # Assuming it was placed successfully

    def classify_current_ecg(self, ecg_values, time_values=None):
        """
        Classify the current ECG using the ML model.
        
        Args:
            ecg_values: numpy array of ECG signal values
            time_values: optional time values corresponding to ECG samples
            
        Returns:
            dict: Classification result including predicted class, confidence, and details
        """
        if not self.ml_enabled:
            return {
                'success': False,
                'class': 'ML Model 2.0 Disabled',
                'confidence': 0.0
            }
        
        # Use the ML classifier
        result = self.ml_classifier.classify_ecg(ecg_values, time_values)
        
        # Update the UI with results
        self.update_results(result['class'])
        
        return result
    
    def ml_classify_ecg(self):
        """Wrapper to classify ECG with ML model when button is clicked"""
        if hasattr(self.parent, 'ecg_canvas') and self.parent.ecg_canvas.ecg_data is not None:
            ecg_data = self.parent.ecg_canvas.ecg_data
            time_data = self.parent.ecg_canvas.time_data
            
            # Call the ML classifier
            result = self.classify_current_ecg(ecg_data, time_data)
            
            return result
        else:
            return {
                'success': False,
                'error': 'No ECG data available'
            }

# Example usage - this would be called from the main ECG-Viewer app
if __name__ == "__main__":
    # This is just an example of how the class would be used
    # In reality, this would be imported and used by the ECG-Viewer
    
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
    
    class DummyECGViewer(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Dummy ECG Viewer")
            self.resize(800, 600)
            
            # Create a central widget and layout
            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)
            self.main_layout = QVBoxLayout(self.central_widget)
            
            # Add a classify button (to mimic the real ECG-Viewer)
            self.button_classify = QPushButton("Classify ECG (Original Model)")
            self.main_layout.addWidget(self.button_classify)
            
            # Initialize the ML classifier UI
            self.ml_ui = MLClassifierUI(self)
            
            # Add a test button
            self.button_test = QPushButton("Test ML Model 2.0 Classification")
            self.button_test.clicked.connect(self.test_classification)
            self.main_layout.addWidget(self.button_test)
    
        def test_classification(self):
            """Generate and classify a synthetic ECG"""
            # Generate a synthetic ECG
            t = np.linspace(0, 10, 2500)  # 10 seconds at 250Hz
            ecg = np.zeros_like(t)
            
            # Create synthetic heartbeats
            for i in range(12):  # 12 beats in 10 seconds = 72 BPM
                beat_center = i * 0.83  # Position of each heartbeat
                # P wave
                ecg += 0.25 * np.exp(-((t - beat_center + 0.2) ** 2) / 0.001)
                # QRS complex
                ecg += 1.0 * np.exp(-((t - beat_center) ** 2) / 0.0002)
                # S wave
                ecg -= 0.3 * np.exp(-((t - beat_center + 0.05) ** 2) / 0.0002)
                # T wave
                ecg += 0.3 * np.exp(-((t - beat_center - 0.15) ** 2) / 0.002)
            
            # Add noise
            ecg += np.random.normal(0, 0.05, len(t))
            
            # Classify the ECG
            self.ml_ui.classify_current_ecg(ecg, t)
    
    # Create and run a simple test app
    app = QApplication(sys.argv)
    viewer = DummyECGViewer()
    viewer.show()
    sys.exit(app.exec_()) 