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
        
        # Create a toggle button for ML classifier
        if self.ml_classifier.use_dummy_model:
            button_text = "ML Model 2.0 (Demo): Off"
        else:
            button_text = "ML Model 2.0: Off"
            
        self.ml_toggle_button = QtWidgets.QPushButton(button_text)
        self.ml_toggle_button.setCheckable(True)
        self.ml_toggle_button.setChecked(False)
        self.ml_toggle_button.setToolTip("Toggle between rule-based and advanced ML-based ECG classification (Model 2.0)")
        self.ml_toggle_button.clicked.connect(self.toggle_ml_classifier)
        
        # Create a results display area
        self.ml_results_group = QtWidgets.QGroupBox("ML Model 2.0 Classification Results")
        self.ml_results_layout = QtWidgets.QVBoxLayout()
        
        # Add a label to indicate this is the advanced ML model
        if self.ml_classifier.use_dummy_model:
            model_description = "Advanced Machine Learning Model DEMO (for demonstration purposes)"
        else:
            model_description = "Advanced Machine Learning Model (98.8% accuracy)"
            
        self.ml_version_label = QtWidgets.QLabel(model_description)
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
        for i, class_name in enumerate(['Normal', 'Abnormal', 'Atrial Fibrillation', 'ST Elevation', 'Bradycardia', 'Tachycardia']):
            label = QtWidgets.QLabel(f"{class_name}: 0.0000")
            self.prob_labels[i] = label
            self.ml_probabilities_layout.addWidget(label)
        
        self.ml_probabilities_group.setLayout(self.ml_probabilities_layout)
        self.ml_results_layout.addWidget(self.ml_probabilities_group)
        
        # Set the layout
        self.ml_results_group.setLayout(self.ml_results_layout)
        
        # Initially hide the ML results
        self.ml_results_group.setVisible(False)
        
        # Try to place the toggle button directly in the control panel
        try:
            # Try to add to ECG Controls panel in a prominent position
            if hasattr(self.parent, 'controlLayout'):
                # Create a dedicated section for ML Model in the control panel
                ml_section = QtWidgets.QGroupBox("ML Model 2.0 Controls")
                ml_section_layout = QtWidgets.QVBoxLayout()
                
                # Add the toggle button with a descriptive label
                ml_toggle_layout = QtWidgets.QHBoxLayout()
                toggle_label = QtWidgets.QLabel("Enable ML Model:")
                toggle_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                ml_toggle_layout.addWidget(toggle_label)
                ml_toggle_layout.addWidget(self.ml_toggle_button)
                
                ml_section_layout.addLayout(ml_toggle_layout)
                
                # Add a descriptive label
                ml_description = QtWidgets.QLabel("Advanced ECG analysis with machine learning")
                ml_description.setAlignment(QtCore.Qt.AlignCenter)
                font = ml_description.font()
                font.setItalic(True)
                ml_description.setFont(font)
                ml_section_layout.addWidget(ml_description)
                
                # Set the layout for the section
                ml_section.setLayout(ml_section_layout)
                
                # Add it to the parent control layout in a good position
                # Insert after main controls but before real-world data
                self.parent.controlLayout.insertWidget(2, ml_section)
                
                # Add the results group to the left panel
                if hasattr(self.parent, 'leftPanelLayout'):
                    self.parent.leftPanelLayout.addWidget(self.ml_results_group)
                else:
                    # Fallback to the main control layout
                    self.parent.controlLayout.addWidget(self.ml_results_group)
            else:
                # Fallback for other layouts
                if hasattr(self.parent, 'leftPanelLayout'):
                    self.parent.leftPanelLayout.addWidget(self.ml_toggle_button)
                    self.parent.leftPanelLayout.addWidget(self.ml_results_group)
                elif hasattr(self.parent, 'button_ml_classify'):
                    parent_layout = self.parent.button_ml_classify.parent().layout()
                    if parent_layout is not None:
                        parent_layout.addWidget(self.ml_toggle_button)
                        parent_layout.addWidget(self.ml_results_group)
                else:
                    print("Could not find appropriate layout for ML UI elements")
        except Exception as e:
            print(f"Error adding ML UI elements: {e}")
            print("You may need to manually add these UI elements to your ECG-Viewer.")
    
    def toggle_ml_classifier(self):
        """Toggle between the rule-based and ML-based classifier"""
        # Get the current state of the button
        is_ml_enabled = self.ml_toggle_button.isChecked()
        
        # Set the appropriate text based on the toggle state and model type
        if is_ml_enabled:
            if self.ml_classifier.use_dummy_model:
                self.ml_toggle_button.setText("ML Model 2.0 (Demo): On")
            else:
                self.ml_toggle_button.setText("ML Model 2.0: On")
            
            # Show the results UI
            self.ml_results_group.setVisible(True)
            
            # Make the ML classify button in the parent visible but disabled
            if hasattr(self.parent, 'button_ml_classify'):
                self.parent.button_ml_classify.setVisible(False)
                
            # Make the original Classify button have a different label
            if hasattr(self.parent, 'classifyButton'):
                self.parent.classifyButton.setText("Rule-Based Classify")
        else:
            if self.ml_classifier.use_dummy_model:
                self.ml_toggle_button.setText("ML Model 2.0 (Demo): Off")
            else:
                self.ml_toggle_button.setText("ML Model 2.0: Off")
            
            # Hide the results UI
            self.ml_results_group.setVisible(False)
            
            # Restore the ML classify button
            if hasattr(self.parent, 'button_ml_classify'):
                self.parent.button_ml_classify.setVisible(True)
                
            # Restore the original Classify button text
            if hasattr(self.parent, 'classifyButton'):
                self.parent.classifyButton.setText("Classify")
                
        # Auto-classify if monitoring is active
        if hasattr(self.parent, 'is_monitoring') and self.parent.is_monitoring and is_ml_enabled:
            # If we have current ECG data, classify it
            if hasattr(self.parent, 'ecg_values') and self.parent.ecg_values is not None:
                self.ml_classify_ecg()
                
        # Update status bar
        if hasattr(self.parent, 'statusBar'):
            if is_ml_enabled:
                demo_suffix = " (Demo Mode)" if self.ml_classifier.use_dummy_model else ""
                self.parent.statusBar.showMessage(f"ML Model 2.0{demo_suffix} enabled. Real-time classification active.")
            else:
                self.parent.statusBar.showMessage("ML Model 2.0 disabled. Using rule-based classification.")
                
        print(f"ML classifier {'enabled' if is_ml_enabled else 'disabled'}")
    
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
        self.update_results(result)
        
        return result
    
    def update_results(self, result):
        """Update the ML classifier UI with new classification results"""
        if not result or not result.get('success', False):
            self.ml_class_label.setText("Class: Classification failed")
            self.ml_confidence_label.setText("Confidence: N/A")
            return
            
        # Show the UI components
        self.ml_results_group.setVisible(True)
        
        # Update class and confidence
        self.ml_class_label.setText(f"Class: {result['class']}")
        
        # Format confidence to 2 decimal places
        confidence = result.get('confidence', 0) * 100
        self.ml_confidence_label.setText(f"Confidence: {confidence:.1f}%")
        
        # Update model version information if demo mode
        if 'Demo' in result.get('model_version', ''):
            demo_text = result.get('model_version', 'ML Model 2.0 (Demo Mode)')
            self.ml_version_label.setText(demo_text)
            
            # Set a yellow background to indicate demo mode
            self.ml_results_group.setStyleSheet("QGroupBox { background-color: #FFFFD0; border: 1px solid #E0E0A0; }")
            
            # Add a note about demo mode at the bottom if not already added
            if not hasattr(self, 'demo_note_label'):
                self.demo_note_label = QtWidgets.QLabel("Note: Running in demo mode - simulated results")
                self.demo_note_label.setStyleSheet("color: #707040; font-style: italic;")
                self.ml_results_layout.addWidget(self.demo_note_label)
                
            # Make demo note visible
            self.demo_note_label.setVisible(True)
        else:
            # Normal mode - reset any demo styling
            self.ml_version_label.setText("Advanced Machine Learning Model (98.8% accuracy)")
            self.ml_results_group.setStyleSheet("")
            
            # Hide demo note if it exists
            if hasattr(self, 'demo_note_label'):
                self.demo_note_label.setVisible(False)
        
        # Add extra information if available
        heart_rate = result.get('heart_rate', None)
        rr_variability = result.get('rr_variability', None)
        
        # Create or update additional metrics display
        if not hasattr(self, 'metrics_group'):
            # Create metrics display if it doesn't exist
            self.metrics_group = QtWidgets.QGroupBox("Additional Metrics")
            self.metrics_layout = QtWidgets.QVBoxLayout()
            self.heart_rate_label = QtWidgets.QLabel("Heart Rate: N/A")
            self.rr_variability_label = QtWidgets.QLabel("RR Variability: N/A")
            
            self.metrics_layout.addWidget(self.heart_rate_label)
            self.metrics_layout.addWidget(self.rr_variability_label)
            self.metrics_group.setLayout(self.metrics_layout)
            
            # Add to main results layout
            self.ml_results_layout.addWidget(self.metrics_group)
        
        # Update metrics if available
        if heart_rate is not None:
            self.heart_rate_label.setText(f"Heart Rate: {heart_rate} BPM")
            
            # Color-code heart rate
            if heart_rate < 60:
                self.heart_rate_label.setStyleSheet("color: blue;")  # Bradycardia
            elif heart_rate > 100:
                self.heart_rate_label.setStyleSheet("color: red;")   # Tachycardia
            else:
                self.heart_rate_label.setStyleSheet("color: green;") # Normal
        
        if rr_variability is not None:
            self.rr_variability_label.setText(f"RR Variability: {rr_variability:.3f}")
            
            # Color-code RR variability
            if rr_variability > 0.2:
                self.rr_variability_label.setStyleSheet("color: red;")   # High variability (AF)
            else:
                self.rr_variability_label.setStyleSheet("color: green;") # Normal variability
        
        # Update detailed probabilities if available
        if 'class_probabilities' in result:
            # Make the probabilities group visible
            self.ml_probabilities_group.setVisible(True)
            
            # Update each probability label
            for i, class_name in enumerate(['Normal', 'Abnormal', 'Atrial Fibrillation', 'ST Elevation', 'Bradycardia', 'Tachycardia']):
                if i in self.prob_labels:
                    prob = result['class_probabilities'].get(class_name, 0)
                    
                    # Format with percentage
                    label_text = f"{class_name}: {prob*100:.1f}%"
                    
                    # Bold the highest probability
                    if class_name == result['class']:
                        label_text = f"<b>{label_text}</b>"
                        # Add an indicator arrow
                        label_text = f"► {label_text}"
                        # Set label color for the selected class
                        self.prob_labels[i].setStyleSheet("color: blue;")
                    else:
                        self.prob_labels[i].setStyleSheet("")
                        
                    self.prob_labels[i].setText(label_text)
        else:
            # Hide probabilities if not available
            self.ml_probabilities_group.setVisible(False)

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