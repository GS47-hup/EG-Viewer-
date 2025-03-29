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
        self.load_model()
        
        # Update parent's ML status label if available
        if hasattr(parent_app, 'mlStatusLabel'):
            if self.model is not None:
                parent_app.mlStatusLabel.setText("ML Model 2.0: Ready")
                parent_app.mlStatusLabel.setStyleSheet("color: green;")
            else:
                parent_app.mlStatusLabel.setText("ML Model 2.0: Not Available")
                parent_app.mlStatusLabel.setStyleSheet("color: red;")
        
        self.setup_ui()
    
    def setup_ui(self):
        """Add ML classifier UI elements to the ECG-Viewer"""
        
        # Create a toggle button for ML classifier
        self.ml_toggle_button = QtWidgets.QPushButton("ML Model 2.0: Off")
        self.ml_toggle_button.setCheckable(True)
        self.ml_toggle_button.setChecked(False)
        self.ml_toggle_button.setToolTip("Toggle between rule-based and advanced ML-based ECG classification (Model 2.0)")
        self.ml_toggle_button.clicked.connect(self.toggle_ml_classifier)
        
        # Create a results display area
        self.ml_results_group = QtWidgets.QGroupBox("ML Model 2.0 Classification Results")
        self.ml_results_layout = QtWidgets.QVBoxLayout()
        
        # Add a label to indicate this is the advanced ML model
        self.ml_version_label = QtWidgets.QLabel("Advanced Machine Learning Model (98.8% accuracy)")
        font = self.ml_version_label.font()
        font.setBold(True)
        self.ml_version_label.setFont(font)
        self.ml_results_layout.addWidget(self.ml_version_label)
        
        # Add a separator line
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.ml_results_layout.addWidget(line)
        
        # Main results layout
        self.ml_class_label = QtWidgets.QLabel("Class: Not classified")
        self.ml_confidence_label = QtWidgets.QLabel("Confidence: N/A")
        self.ml_results_layout.addWidget(self.ml_class_label)
        self.ml_results_layout.addWidget(self.ml_confidence_label)
        
        # Detailed probabilities
        self.ml_probabilities_group = QtWidgets.QGroupBox("Class Probabilities")
        self.ml_probabilities_layout = QtWidgets.QVBoxLayout()
        
        # Add probability labels for each class
        self.prob_labels = {}
        for i, class_name in [(0, 'Normal beats'), 
                             (1, 'Supraventricular ectopic beats'),
                             (2, 'Ventricular ectopic beats'), 
                             (3, 'Fusion beats'),
                             (4, 'Unknown beats')]:
            label = QtWidgets.QLabel(f"{class_name}: 0.0000")
            self.prob_labels[i] = label
            self.ml_probabilities_layout.addWidget(label)
        
        self.ml_probabilities_group.setLayout(self.ml_probabilities_layout)
        self.ml_results_layout.addWidget(self.ml_probabilities_group)
        
        # Set the layout
        self.ml_results_group.setLayout(self.ml_results_layout)
        
        # Initially hide the ML results
        self.ml_results_group.setVisible(False)
        
        # Add to the parent's appropriate layout
        try:
            if hasattr(self.parent, 'leftPanelLayout'):
                # Add to the left panel if it exists in the new layout
                self.parent.leftPanelLayout.addWidget(self.ml_toggle_button)
                self.parent.leftPanelLayout.addWidget(self.ml_results_group)
            elif hasattr(self.parent, 'button_ml_classify'):
                # If we have the ML classify button, add next to it
                parent_layout = self.parent.button_ml_classify.parent().layout()
                if parent_layout is not None:
                    parent_layout.addWidget(self.ml_toggle_button)
                    parent_layout.addWidget(self.ml_results_group)
            else:
                # Fallback to adding to the main control layout
                self.parent.controlLayout.addWidget(self.ml_toggle_button)
                self.parent.controlLayout.addWidget(self.ml_results_group)
        except Exception as e:
            print(f"Error adding ML UI elements: {e}")
            print("You may need to manually add these UI elements to your ECG-Viewer.")
    
    def toggle_ml_classifier(self, checked):
        """Toggle between rule-based and ML-based classifier"""
        self.ml_enabled = checked
        
        if checked:
            self.ml_toggle_button.setText("ML Model 2.0: On")
            self.ml_results_group.setVisible(True)
            print("Advanced ML Model 2.0 enabled")
            
            # Update status label if available
            if hasattr(self.parent, 'mlStatusLabel'):
                self.parent.mlStatusLabel.setText("ML Model 2.0: Active")
                self.parent.mlStatusLabel.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.ml_toggle_button.setText("ML Model 2.0: Off")
            self.ml_results_group.setVisible(False)
            print("Advanced ML Model 2.0 disabled")
            
            # Update status label if available
            if hasattr(self.parent, 'mlStatusLabel'):
                self.parent.mlStatusLabel.setText("ML Model 2.0: Ready")
                self.parent.mlStatusLabel.setStyleSheet("color: black;")
    
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
        self.update_ui_with_results(result)
        
        return result
    
    def update_ui_with_results(self, result):
        """
        Update the UI with classification results.
        
        Args:
            result: dict containing classification results
        """
        if result['success']:
            # Update main class and confidence
            self.ml_class_label.setText(f"Class: {result['class']}")
            self.ml_confidence_label.setText(f"Confidence: {result.get('confidence', 0):.4f}")
            
            # Update detailed probabilities if available
            if 'class_probabilities' in result:
                for i, class_name in enumerate(['Normal beats', 
                                                'Supraventricular ectopic beats',
                                                'Ventricular ectopic beats', 
                                                'Fusion beats',
                                                'Unknown beats']):
                    if i in self.prob_labels:
                        prob = result['class_probabilities'].get(class_name, 0)
                        self.prob_labels[i].setText(f"{class_name}: {prob:.4f}")
        else:
            # Show error
            self.ml_class_label.setText(f"Class: Error")
            self.ml_confidence_label.setText(f"Error: {result.get('error', 'Unknown error')}")

    def load_model(self):
        """Load the ML model"""
        self.ml_classifier.load_model()
        self.model = self.ml_classifier.model
    

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