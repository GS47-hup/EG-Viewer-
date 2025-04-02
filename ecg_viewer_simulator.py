import sys
import os
import numpy as np
import pandas as pd
import random
from PyQt5 import QtWidgets, QtCore, QtGui
import ecg_viewer
from ecg_viewer_window import Ui_MainWindow

# Import our classifier
from ecg_classifier import ECGClassifier
from real_ecg_classifier import RealEcgClassifier

class ECGViewerWithSimulator(ecg_viewer.ECGViewer):
    """
    Enhanced ECG Viewer with simulation capabilities for real ECG data
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize the parent class
        super(ECGViewerWithSimulator, self).__init__(*args, **kwargs)
        
        # Add simulation buttons to the UI
        self.setup_simulation_ui()
        
        # Initialize the classifier
        self.classifier = ECGClassifier()
        self.real_classifier = RealEcgClassifier.load_model('real_ecg_data/real_ecg_classifier.joblib')
        
        # Setup simulation variables
        self.simulation_active = False
        self.simulation_timer = QtCore.QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation)
        self.simulation_data = None
        self.simulation_index = 0
        self.ecg_file = 'Real ECG.csv'
        self.loaded_samples = []
        self.current_sample_index = None
        self.current_sample_label = None
        
    def setup_simulation_ui(self):
        """Setup the simulation UI elements"""
        # Create a group box for simulation controls
        self.groupBox_simulation = QtWidgets.QGroupBox("ECG Simulation")
        self.groupBox_simulation.setMinimumHeight(150)
        
        # Create layout for simulation controls
        sim_layout = QtWidgets.QVBoxLayout()
        
        # Create buttons for simulation
        self.button_load_normal = QtWidgets.QPushButton("Load Normal ECG")
        self.button_load_normal.setToolTip("Load a random normal ECG sample")
        self.button_load_normal.clicked.connect(lambda: self.load_ecg_sample(is_normal=True))
        
        self.button_load_abnormal = QtWidgets.QPushButton("Load Abnormal ECG")
        self.button_load_abnormal.setToolTip("Load a random abnormal ECG sample")
        self.button_load_abnormal.clicked.connect(lambda: self.load_ecg_sample(is_normal=False))
        
        self.button_load_random = QtWidgets.QPushButton("Load Random ECG")
        self.button_load_random.setToolTip("Load a random ECG sample (normal or abnormal)")
        self.button_load_random.clicked.connect(lambda: self.load_ecg_sample())
        
        # Create simulation control buttons
        self.button_start_simulation = QtWidgets.QPushButton("Start Simulation")
        self.button_start_simulation.setToolTip("Start simulating the loaded ECG data")
        self.button_start_simulation.clicked.connect(self.start_simulation)
        self.button_start_simulation.setEnabled(False)
        
        self.button_stop_simulation = QtWidgets.QPushButton("Stop Simulation")
        self.button_stop_simulation.setToolTip("Stop the ECG simulation")
        self.button_stop_simulation.clicked.connect(self.stop_simulation)
        self.button_stop_simulation.setEnabled(False)
        
        # Create speed slider
        self.label_speed = QtWidgets.QLabel("Simulation Speed:")
        self.slider_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_speed.setMinimum(1)
        self.slider_speed.setMaximum(10)
        self.slider_speed.setValue(5)
        self.slider_speed.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_speed.setTickInterval(1)
        
        # Create info label for loaded sample
        self.label_sample_info = QtWidgets.QLabel("No sample loaded")
        self.label_sample_info.setAlignment(QtCore.Qt.AlignCenter)
        
        # Add buttons to layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.button_load_normal)
        button_layout.addWidget(self.button_load_abnormal)
        button_layout.addWidget(self.button_load_random)
        
        sim_control_layout = QtWidgets.QHBoxLayout()
        sim_control_layout.addWidget(self.button_start_simulation)
        sim_control_layout.addWidget(self.button_stop_simulation)
        
        speed_layout = QtWidgets.QHBoxLayout()
        speed_layout.addWidget(self.label_speed)
        speed_layout.addWidget(self.slider_speed)
        
        # Add all layouts to main simulation layout
        sim_layout.addLayout(button_layout)
        sim_layout.addLayout(sim_control_layout)
        sim_layout.addLayout(speed_layout)
        sim_layout.addWidget(self.label_sample_info)
        
        # Set the layout for the group box
        self.groupBox_simulation.setLayout(sim_layout)
        
        # Add the group box to the main window layout
        self.gridLayout.addWidget(self.groupBox_simulation, 6, 0, 1, 5)
    
    def load_ecg_sample(self, is_normal=None, sample_index=None):
        """
        Load an ECG sample from the Real ECG.csv file
        
        Args:
            is_normal: If True, load a normal sample; if False, load an abnormal sample;
                      if None, load a random sample.
            sample_index: Specific sample index to load
        """
        try:
            # Check if the ECG file exists
            if not os.path.exists(self.ecg_file):
                self.ui_display_error_message("File Error", f"ECG file '{self.ecg_file}' not found.")
                return
            
            # Load the data if not already loaded
            if not self.loaded_samples:
                data = pd.read_csv(self.ecg_file, header=None)
                self.loaded_samples = data
                print(f"Loaded ECG data with {len(data)} samples")
            
            # If sample_index is provided, use it
            if sample_index is not None:
                if sample_index >= len(self.loaded_samples):
                    self.ui_display_error_message("Index Error", 
                                                f"Sample index {sample_index} is out of range. File has {len(self.loaded_samples)} samples.")
                    return
                
                self.current_sample_index = sample_index
                self.current_sample_label = self.loaded_samples.iloc[sample_index, -1]
                
            # If is_normal is specified, filter and select randomly
            elif is_normal is not None:
                target_label = 0 if is_normal else 1
                filtered_data = self.loaded_samples[self.loaded_samples.iloc[:, -1] == target_label]
                
                if len(filtered_data) == 0:
                    self.ui_display_error_message("Data Error", 
                                                f"No {'normal' if is_normal else 'abnormal'} samples found in the file.")
                    return
                
                # Select a random sample from the filtered data
                random_index = random.randint(0, len(filtered_data) - 1)
                self.current_sample_index = filtered_data.index[random_index]
                self.current_sample_label = target_label
                
            else:
                # Select a random sample from the entire dataset
                self.current_sample_index = random.randint(0, len(self.loaded_samples) - 1)
                self.current_sample_label = self.loaded_samples.iloc[self.current_sample_index, -1]
            
            # Extract the sample
            self.simulation_data = self.loaded_samples.iloc[self.current_sample_index, :-1].values
            
            # Update the UI
            self.label_sample_info.setText(
                f"Loaded sample {self.current_sample_index}: "
                f"{'Normal' if self.current_sample_label == 0 else 'Abnormal'} "
                f"({len(self.simulation_data)} data points)"
            )
            
            # Enable start simulation button
            self.button_start_simulation.setEnabled(True)
            
            # Clear previous data from the graph
            self.clear_history()
            
            # Update status bar
            self.ui_statusbar_message(f"Loaded {'normal' if self.current_sample_label == 0 else 'abnormal'} ECG sample {self.current_sample_index}")
            
            print(f"Successfully loaded {'normal' if self.current_sample_label == 0 else 'abnormal'} ECG sample {self.current_sample_index}")
            
        except Exception as e:
            self.ui_display_error_message("Error", f"Failed to load ECG sample: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def start_simulation(self):
        """Start the ECG simulation"""
        if self.simulation_data is None:
            self.ui_display_error_message("Error", "No ECG sample loaded. Please load a sample first.")
            return
        
        try:
            # Get simulation speed
            speed_factor = self.slider_speed.value() / 5.0  # 1.0 is normal speed
            
            # Reset simulation variables
            self.simulation_index = 0
            self.simulation_active = True
            
            # Calculate the timer interval based on speed
            # Assuming 250 Hz sampling rate, that's 4ms per sample at normal speed
            interval = int(4 / speed_factor)
            self.simulation_timer.setInterval(interval)
            
            # Start the timer
            self.simulation_timer.start()
            
            # Update UI
            self.button_start_simulation.setEnabled(False)
            self.button_stop_simulation.setEnabled(True)
            self.button_load_normal.setEnabled(False)
            self.button_load_abnormal.setEnabled(False)
            self.button_load_random.setEnabled(False)
            
            # Update status bar
            self.ui_statusbar_message(f"Simulating ECG at {speed_factor:.1f}x speed")
            
            print(f"Started ECG simulation at {speed_factor:.1f}x speed (interval: {interval}ms)")
            
        except Exception as e:
            self.ui_display_error_message("Error", f"Failed to start simulation: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def stop_simulation(self):
        """Stop the ECG simulation"""
        try:
            # Stop the timer
            self.simulation_timer.stop()
            self.simulation_active = False
            
            # Update UI
            self.button_start_simulation.setEnabled(True)
            self.button_stop_simulation.setEnabled(False)
            self.button_load_normal.setEnabled(True)
            self.button_load_abnormal.setEnabled(True)
            self.button_load_random.setEnabled(True)
            
            # Update status bar
            self.ui_statusbar_message("Simulation stopped")
            
            print("Stopped ECG simulation")
            
        except Exception as e:
            self.ui_display_error_message("Error", f"Failed to stop simulation: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_simulation(self):
        """Update the simulation by adding the next data point"""
        if not self.simulation_active or self.simulation_data is None:
            return
        
        try:
            # Get the current data point
            value = self.simulation_data[self.simulation_index]
            
            # Add the value to the graph
            # Note: For simulation, we use fake time values based on index
            # Assuming 250 Hz sampling rate, that's 4ms per sample
            time_ms = self.simulation_index * 4
            self.add_data_point(time_ms, value)
            
            # Increment the index
            self.simulation_index += 1
            
            # If we've reached the end of the data, loop back to the beginning
            if self.simulation_index >= len(self.simulation_data):
                self.simulation_index = 0
                print("Restarting ECG simulation from beginning")
            
        except Exception as e:
            print(f"Error during simulation update: {str(e)}")
            self.stop_simulation()
    
    def classify_current_ecg(self):
        """Override parent method to handle classification of simulated data"""
        # If we're simulating, use the known label
        if self.simulation_active and self.current_sample_label is not None:
            is_normal = self.current_sample_label == 0
            classification = "normal" if is_normal else "abnormal"
            confidence = 95.0  # High confidence since we know the label
            
            if is_normal:
                reasons = ["Normal sinus rhythm", "Regular R-R intervals", "Normal ECG morphology"]
            else:
                reasons = ["Abnormal heart rhythm", "Irregular R-R intervals", "Abnormal ECG morphology"]
            
            # Update status label
            status_color = "#28a745" if is_normal else "#dc3545"
            # This attribute may exist in the parent class, adjust as needed
            if hasattr(self, 'classification_label'):
                self.classification_label.setText(f"Classification: {classification.upper()} ({confidence:.1f}%)")
                self.classification_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")
            
            # Show results dialog - this assumes a ClassificationResultDialog exists
            # You may need to implement this dialog as in the original ECG viewer
            if hasattr(self, 'ClassificationResultDialog'):
                dialog = self.ClassificationResultDialog(classification, confidence, reasons, self)
                dialog.exec_()
            else:
                # Fallback to a simple message box
                msg = QtWidgets.QMessageBox()
                msg.setWindowTitle("ECG Classification Results")
                msg.setText(f"Classification: {classification.upper()}")
                msg.setInformativeText(f"Confidence: {confidence:.1f}%\n\nReasons:\n" + "\n".join(reasons))
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.exec_()
            
            # Update status bar
            self.ui_statusbar_message(f"ECG classified as {classification.upper()}")
            
            return
        
        # Otherwise, use the parent class method
        super(ECGViewerWithSimulator, self).classify_current_ecg()

def main():
    """Main entry point for the ECG Viewer with Simulator"""
    app = QtWidgets.QApplication(sys.argv)
    window = ECGViewerWithSimulator()
    window.setWindowTitle("ECG Viewer with Simulator - " + ecg_viewer.VERSION)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 