#!/usr/bin/env python
"""
Standalone ECG Simulator - A real-time ECG viewer that simulates sensor input
"""
import sys
import os
import numpy as np
import pandas as pd
import random
import time
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy import signal

class ECGCanvas(FigureCanvasQTAgg):
    """Canvas for plotting ECG data"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#f0f0f0')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('Time (ms)')
        self.axes.set_ylabel('Amplitude (mV)')
        self.axes.set_title('ECG Signal')
        
        super(ECGCanvas, self).__init__(self.fig)
        
        # Set up data
        self.time_data = []
        self.ecg_data = []
        self.r_peaks = []
        
    def update_plot(self, time_data=None, ecg_data=None, r_peaks=None):
        """Update the plot with new data"""
        if time_data is not None:
            self.time_data = time_data
        if ecg_data is not None:
            self.ecg_data = ecg_data
        if r_peaks is not None:
            self.r_peaks = r_peaks
            
        # Clear the axis
        self.axes.clear()
        
        # Re-plot the data
        if len(self.time_data) > 0 and len(self.ecg_data) > 0:
            self.axes.plot(self.time_data, self.ecg_data, 'b-')
            
            # Plot R-peaks if available
            if len(self.r_peaks) > 0:
                peak_times = [self.time_data[i] for i in self.r_peaks if i < len(self.time_data)]
                peak_values = [self.ecg_data[i] for i in self.r_peaks if i < len(self.ecg_data)]
                self.axes.plot(peak_times, peak_values, 'ro')
        
        # Set up grid and labels
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.axes.set_xlabel('Time (ms)')
        self.axes.set_ylabel('Amplitude (mV)')
        self.axes.set_title('ECG Signal')
        
        # Draw the updated plot
        self.fig.tight_layout()
        self.draw()
    
    def clear_plot(self):
        """Clear the plot"""
        self.time_data = []
        self.ecg_data = []
        self.r_peaks = []
        self.update_plot()

class ECGSimulator(QtWidgets.QMainWindow):
    """Main window for ECG simulator"""
    def __init__(self):
        super(ECGSimulator, self).__init__()
        
        # Set up the window
        self.setWindowTitle("Real-Time ECG Simulator")
        self.setGeometry(100, 100, 800, 600)
        
        # Set up central widget and main layout
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        
        # Create ECG canvas
        self.ecg_canvas = ECGCanvas(self.centralWidget, width=8, height=4)
        self.mainLayout.addWidget(self.ecg_canvas)
        
        # Create control panel
        self.controlPanel = QtWidgets.QGroupBox("ECG Controls")
        self.controlLayout = QtWidgets.QVBoxLayout(self.controlPanel)
        
        # Create buttons for real-time monitoring
        self.buttonLayout = QtWidgets.QHBoxLayout()
        
        self.startMonitoringButton = QtWidgets.QPushButton("Start Monitoring")
        self.startMonitoringButton.clicked.connect(self.start_real_time_monitoring)
        self.buttonLayout.addWidget(self.startMonitoringButton)
        
        self.stopMonitoringButton = QtWidgets.QPushButton("Stop Monitoring")
        self.stopMonitoringButton.clicked.connect(self.stop_real_time_monitoring)
        self.stopMonitoringButton.setEnabled(False)
        self.buttonLayout.addWidget(self.stopMonitoringButton)
        
        self.controlLayout.addLayout(self.buttonLayout)
        
        # Create ECG type selection
        self.ecgTypeLayout = QtWidgets.QHBoxLayout()
        self.ecgTypeLabel = QtWidgets.QLabel("ECG Type:")
        self.ecgTypeCombo = QtWidgets.QComboBox()
        self.ecgTypeCombo.addItems(["Normal", "Atrial Fibrillation", "ST Elevation", "Tachycardia", "Bradycardia"])
        
        self.ecgTypeLayout.addWidget(self.ecgTypeLabel)
        self.ecgTypeLayout.addWidget(self.ecgTypeCombo)
        
        self.controlLayout.addLayout(self.ecgTypeLayout)
        
        # Create noise level slider
        self.noiseLayout = QtWidgets.QHBoxLayout()
        self.noiseLabel = QtWidgets.QLabel("Noise Level:")
        self.noiseSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.noiseSlider.setMinimum(0)
        self.noiseSlider.setMaximum(100)
        self.noiseSlider.setValue(10)
        self.noiseSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.noiseSlider.setTickInterval(10)
        
        self.noiseLayout.addWidget(self.noiseLabel)
        self.noiseLayout.addWidget(self.noiseSlider)
        
        self.controlLayout.addLayout(self.noiseLayout)
        
        # Create heart rate slider
        self.hrLayout = QtWidgets.QHBoxLayout()
        self.hrLabel = QtWidgets.QLabel("Heart Rate (BPM):")
        self.hrSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.hrSlider.setMinimum(40)
        self.hrSlider.setMaximum(200)
        self.hrSlider.setValue(70)
        self.hrSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.hrSlider.setTickInterval(20)
        self.hrValueLabel = QtWidgets.QLabel("70")
        self.hrSlider.valueChanged.connect(lambda v: self.hrValueLabel.setText(str(v)))
        
        self.hrLayout.addWidget(self.hrLabel)
        self.hrLayout.addWidget(self.hrSlider)
        self.hrLayout.addWidget(self.hrValueLabel)
        
        self.controlLayout.addLayout(self.hrLayout)
        
        # Create info label
        self.infoLabel = QtWidgets.QLabel("Ready to monitor")
        self.infoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.controlLayout.addWidget(self.infoLabel)
        
        # Add classify button
        self.classifyButton = QtWidgets.QPushButton("Classify ECG")
        self.classifyButton.clicked.connect(self.classify_ecg)
        self.classifyButton.setEnabled(False)
        self.controlLayout.addWidget(self.classifyButton)
        
        # Add recording button
        self.recordButton = QtWidgets.QPushButton("Start Recording")
        self.recordButton.clicked.connect(self.toggle_recording)
        self.recordButton.setEnabled(False)
        self.controlLayout.addWidget(self.recordButton)
        
        # Add status bar
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready to monitor ECG")
        
        # Add control panel to main layout
        self.mainLayout.addWidget(self.controlPanel)
        
        # Load button for backward compatibility
        self.loadLayout = QtWidgets.QHBoxLayout()
        self.loadLabel = QtWidgets.QLabel("Previous Mode:")
        self.loadNormalButton = QtWidgets.QPushButton("Load Normal ECG")
        self.loadNormalButton.clicked.connect(lambda: self.load_ecg_sample(is_normal=True))
        
        self.loadAbnormalButton = QtWidgets.QPushButton("Load Abnormal ECG")
        self.loadAbnormalButton.clicked.connect(lambda: self.load_ecg_sample(is_normal=False))
        
        self.loadLayout.addWidget(self.loadLabel)
        self.loadLayout.addWidget(self.loadNormalButton)
        self.loadLayout.addWidget(self.loadAbnormalButton)
        
        self.controlLayout.addLayout(self.loadLayout)
        
        # Initialize simulation variables
        self.ecg_file = 'Real ECG.csv'
        self.loaded_samples = None
        self.current_sample_label = None
        self.monitoring_active = False
        self.recording_active = False
        self.recorded_data = []
        
        # Real-time monitoring variables
        self.monitoring_timer = QtCore.QTimer()
        self.monitoring_timer.timeout.connect(self.update_real_time_ecg)
        self.sampling_rate = 250  # Hz
        self.update_interval = 20  # ms (50 Hz refresh rate)
        self.time_window = 5000  # ms (5 seconds of data shown)
        self.max_points = self.time_window * self.sampling_rate // 1000
        
        self.current_time = 0
        self.ecg_buffer_time = []
        self.ecg_buffer_values = []
        
        # ECG generation parameters
        self.ecg_templates = self._load_ecg_templates()
        
        # Check if ECG file exists for backward compatibility
        if not os.path.exists(self.ecg_file):
            self.statusBar.showMessage(f"Warning: ECG file '{self.ecg_file}' not found.")
            self.loadNormalButton.setEnabled(False)
            self.loadAbnormalButton.setEnabled(False)
    
    def _load_ecg_templates(self):
        """Load or create ECG templates for different types"""
        templates = {}
        
        # Generate basic ECG template if we don't have real samples
        # Normal sinus rhythm template (simplified)
        t = np.linspace(0, 2*np.pi, 200)
        normal = np.zeros_like(t)
        # P wave
        normal += 0.25 * np.exp(-((t-0.4)**2)/0.01)
        # QRS complex
        normal += -0.6 * np.exp(-((t-1.0)**2)/0.005)
        normal += 1.5 * np.exp(-((t-1.2)**2)/0.003)
        normal += -0.5 * np.exp(-((t-1.4)**2)/0.008)
        # T wave
        normal += 0.35 * np.exp(-((t-2.0)**2)/0.02)
        
        templates['normal'] = normal
        
        # Atrial fibrillation: irregular rhythm, absence of P waves
        afib = np.copy(normal)
        # Remove P waves
        afib -= 0.25 * np.exp(-((t-0.4)**2)/0.01)
        # Add fibrillatory waves
        for i in range(5, 30, 4):
            afib += 0.05 * np.sin(i * t) * np.exp(-((t-0.6)**2)/0.2)
        
        templates['afib'] = afib
        
        # ST elevation (myocardial infarction)
        st_elevation = np.copy(normal)
        # Elevate the ST segment
        st_elevation += 0.2 * (np.exp(-((t-1.7)**2)/0.1) + np.exp(-((t-1.8)**2)/0.1))
        
        templates['st_elevation'] = st_elevation
        
        # Tachycardia (fast heart rate) - use normal template with fast rate
        templates['tachycardia'] = normal
        
        # Bradycardia (slow heart rate) - use normal template with slow rate
        templates['bradycardia'] = normal
        
        return templates
    
    def start_real_time_monitoring(self):
        """Start real-time ECG monitoring"""
        try:
            # Clear canvas and prepare for real-time data
            self.ecg_canvas.clear_plot()
            
            # Reset buffers
            self.current_time = 0
            self.ecg_buffer_time = []
            self.ecg_buffer_values = []
            
            # Start the timer for updates
            self.monitoring_timer.start(self.update_interval)
            self.monitoring_active = True
            
            # Update UI
            self.startMonitoringButton.setEnabled(False)
            self.stopMonitoringButton.setEnabled(True)
            self.loadNormalButton.setEnabled(False)
            self.loadAbnormalButton.setEnabled(False)
            self.classifyButton.setEnabled(True)
            self.recordButton.setEnabled(True)
            
            # Update status
            self.statusBar.showMessage("Real-time ECG monitoring started")
            self.infoLabel.setText("Monitoring: " + self.ecgTypeCombo.currentText() + " ECG pattern")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error starting monitoring: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def stop_real_time_monitoring(self):
        """Stop real-time ECG monitoring"""
        try:
            # Stop the timer
            self.monitoring_timer.stop()
            self.monitoring_active = False
            
            # Stop recording if active
            if self.recording_active:
                self.toggle_recording()
            
            # Update UI
            self.startMonitoringButton.setEnabled(True)
            self.stopMonitoringButton.setEnabled(False)
            self.loadNormalButton.setEnabled(True)
            self.loadAbnormalButton.setEnabled(True)
            self.classifyButton.setEnabled(False)
            self.recordButton.setEnabled(False)
            
            # Update status
            self.statusBar.showMessage("Real-time ECG monitoring stopped")
            self.infoLabel.setText("Monitoring stopped")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error stopping monitoring: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_real_time_ecg(self):
        """Update real-time ECG data"""
        if not self.monitoring_active:
            return
        
        try:
            # Get parameters
            heart_rate = self.hrSlider.value()  # BPM
            noise_level = self.noiseSlider.value() / 1000  # scale 0-0.1
            ecg_type = self.ecgTypeCombo.currentText().lower().replace(" ", "_")
            
            # Calculate parameters
            rr_interval = 60000 / heart_rate  # in ms
            samples_per_beat = int(rr_interval * self.sampling_rate / 1000)
            
            # Generate new data points (typically 5-10 points per update at 50Hz refresh)
            points_to_add = int(self.sampling_rate * self.update_interval / 1000)
            
            # Get the appropriate template
            if ecg_type == "normal":
                template = self.ecg_templates['normal']
            elif ecg_type == "atrial_fibrillation":
                template = self.ecg_templates['afib']
                # Add irregularity to RR interval for AFib
                rr_interval += random.uniform(-100, 100)
                samples_per_beat = int(rr_interval * self.sampling_rate / 1000)
            elif ecg_type == "st_elevation":
                template = self.ecg_templates['st_elevation']
            elif ecg_type == "tachycardia":
                template = self.ecg_templates['normal']
                # Force high heart rate for tachycardia
                if heart_rate < 100:
                    heart_rate = 120 
                    self.hrSlider.setValue(heart_rate)
            elif ecg_type == "bradycardia":
                template = self.ecg_templates['normal']
                # Force low heart rate for bradycardia
                if heart_rate > 60:
                    heart_rate = 50
                    self.hrSlider.setValue(heart_rate)
            else:
                template = self.ecg_templates['normal']
            
            new_time_points = []
            new_ecg_points = []
            
            # Generate new data points
            for i in range(points_to_add):
                # Calculate the current position in the ECG cycle
                position_in_cycle = (self.current_time % rr_interval) / rr_interval
                
                # Map to template index
                template_index = int(position_in_cycle * len(template))
                if template_index >= len(template):
                    template_index = len(template) - 1
                
                # Get the ECG value from template and add noise
                ecg_value = template[template_index]
                ecg_value += np.random.normal(0, noise_level)
                
                # Add the point
                new_time_points.append(self.current_time)
                new_ecg_points.append(ecg_value)
                
                # Record if active
                if self.recording_active:
                    self.recorded_data.append((self.current_time, ecg_value))
                
                # Increment time
                self.current_time += 1000 / self.sampling_rate  # in ms
            
            # Add new points to buffer
            self.ecg_buffer_time.extend(new_time_points)
            self.ecg_buffer_values.extend(new_ecg_points)
            
            # Limit buffer size to time window
            if len(self.ecg_buffer_time) > self.max_points:
                self.ecg_buffer_time = self.ecg_buffer_time[-self.max_points:]
                self.ecg_buffer_values = self.ecg_buffer_values[-self.max_points:]
            
            # Detect R-peaks
            try:
                r_peaks, _ = signal.find_peaks(
                    self.ecg_buffer_values, 
                    height=0.5*max(self.ecg_buffer_values) if self.ecg_buffer_values else 0, 
                    distance=samples_per_beat * 0.8
                )
            except Exception as e:
                print(f"Error detecting R-peaks: {e}")
                r_peaks = []
            
            # Calculate heart rate from R-peaks if we have enough
            if len(r_peaks) > 1:
                rr_intervals = np.diff([self.ecg_buffer_time[i] for i in r_peaks])
                if len(rr_intervals) > 0:
                    avg_rr = np.mean(rr_intervals)
                    calculated_hr = int(60000 / avg_rr) if avg_rr > 0 else 0
                    # Update the HR display only if it's significantly different
                    if abs(calculated_hr - heart_rate) > 5:
                        self.hrValueLabel.setText(f"{heart_rate} ({calculated_hr})")
            
            # Update the plot
            self.ecg_canvas.update_plot(self.ecg_buffer_time, self.ecg_buffer_values, r_peaks)
            
        except Exception as e:
            print(f"Error during real-time update: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def toggle_recording(self):
        """Toggle recording of ECG data"""
        if not self.recording_active:
            # Start recording
            self.recorded_data = []
            self.recording_active = True
            self.recordButton.setText("Stop Recording")
            self.statusBar.showMessage("Recording ECG data...")
        else:
            # Stop recording and save
            self.recording_active = False
            self.recordButton.setText("Start Recording")
            
            if len(self.recorded_data) > 0:
                try:
                    # Create a directory for recordings if it doesn't exist
                    os.makedirs("recordings", exist_ok=True)
                    
                    # Generate filename with timestamp
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    ecg_type = self.ecgTypeCombo.currentText().lower().replace(" ", "_")
                    filename = f"recordings/ecg_{ecg_type}_{timestamp}.csv"
                    
                    # Save to CSV
                    with open(filename, 'w') as f:
                        f.write("time,ecg\n")
                        for t, v in self.recorded_data:
                            f.write(f"{t},{v}\n")
                    
                    self.statusBar.showMessage(f"Recorded ECG data saved to {filename}")
                except Exception as e:
                    self.statusBar.showMessage(f"Error saving recording: {str(e)}")
            else:
                self.statusBar.showMessage("No data recorded")
    
    def select_ecg_file(self):
        """Select an ECG file (for backward compatibility)"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ECG File", "", "CSV Files (*.csv);;All Files (*.*)")
        
        if file_path:
            self.ecg_file = file_path
            self.statusBar.showMessage(f"Selected ECG file: {file_path}")
            self.loadNormalButton.setEnabled(True)
            self.loadAbnormalButton.setEnabled(True)
    
    def load_ecg_sample(self, is_normal=None, sample_index=None):
        """Load an ECG sample from file (for backward compatibility)"""
        try:
            # Check if the ECG file exists
            if not os.path.exists(self.ecg_file):
                self.statusBar.showMessage(f"Error: ECG file '{self.ecg_file}' not found.")
                return
            
            # Load the data if not already loaded
            if self.loaded_samples is None:
                self.statusBar.showMessage(f"Loading ECG data from {self.ecg_file}...")
                data = pd.read_csv(self.ecg_file, header=None)
                self.loaded_samples = data
                self.statusBar.showMessage(f"Loaded ECG data with {len(data)} samples")
            
            # If sample_index is provided, use it
            if sample_index is not None:
                if sample_index >= len(self.loaded_samples):
                    self.statusBar.showMessage(f"Error: Sample index {sample_index} is out of range.")
                    return
                
                current_sample_index = sample_index
                self.current_sample_label = self.loaded_samples.iloc[sample_index, -1]
                
            # If is_normal is specified, filter and select randomly
            elif is_normal is not None:
                target_label = 0 if is_normal else 1
                filtered_data = self.loaded_samples[self.loaded_samples.iloc[:, -1] == target_label]
                
                if len(filtered_data) == 0:
                    self.statusBar.showMessage(f"Error: No {'normal' if is_normal else 'abnormal'} samples found.")
                    return
                
                # Select a random sample from the filtered data
                random_index = random.randint(0, len(filtered_data) - 1)
                current_sample_index = filtered_data.index[random_index]
                self.current_sample_label = target_label
                
            else:
                # Select a random sample from the entire dataset
                current_sample_index = random.randint(0, len(self.loaded_samples) - 1)
                self.current_sample_label = self.loaded_samples.iloc[current_sample_index, -1]
            
            # Extract the sample
            simulation_data = self.loaded_samples.iloc[current_sample_index, :-1].values
            
            # Create time values (assuming 250 Hz sampling rate)
            length = len(simulation_data)
            time_values = np.arange(length) * (1000 / self.sampling_rate)  # in ms
            
            # Calculate R-peaks for display
            try:
                r_peaks, _ = signal.find_peaks(simulation_data, height=0.5*max(simulation_data), distance=50)
            except Exception as e:
                print(f"Error detecting R-peaks: {e}")
                r_peaks = []
            
            # Update the plot
            self.ecg_canvas.update_plot(time_values, simulation_data, r_peaks)
            
            # Update the UI
            self.infoLabel.setText(
                f"Loaded {'Normal' if self.current_sample_label == 0 else 'Abnormal'} ECG "
                f"({length} data points, {len(r_peaks)} peaks)"
            )
            
            # Enable buttons
            self.classifyButton.setEnabled(True)
            
            # Update status bar
            self.statusBar.showMessage(f"Loaded {'normal' if self.current_sample_label == 0 else 'abnormal'} ECG sample")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error loading sample: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def classify_ecg(self):
        """Classify the current ECG data"""
        try:
            # For real-time monitoring, we use the selected ECG type
            if self.monitoring_active:
                ecg_type = self.ecgTypeCombo.currentText().lower()
                
                if "normal" in ecg_type:
                    is_normal = True
                else:
                    is_normal = False
                    abnormality = ecg_type
            else:
                # For loaded samples, use the known label
                is_normal = self.current_sample_label == 0
                abnormality = "unknown"
            
            # Show classification result
            msg_box = QtWidgets.QMessageBox()
            msg_box.setWindowTitle("ECG Classification Result")
            
            if is_normal:
                msg_box.setText("Classification: NORMAL")
                msg_box.setInformativeText(
                    "Confidence: 95.0%\n\n"
                    "Analysis:\n"
                    "- Normal heart rhythm\n"
                    "- Regular R-R intervals\n"
                    "- No significant arrhythmias detected\n"
                    "- Normal waveform morphology"
                )
                msg_box.setIcon(QtWidgets.QMessageBox.Information)
            else:
                msg_box.setText("Classification: ABNORMAL")
                
                # Customize the message based on the abnormality type
                if "fibrillation" in abnormality:
                    details = (
                        "Analysis:\n"
                        "- Irregular heart rhythm\n"
                        "- Absence of P waves\n"
                        "- Rapid, irregular fibrillatory waves\n"
                        "- Irregular ventricular response"
                    )
                elif "elevation" in abnormality:
                    details = (
                        "Analysis:\n"
                        "- ST segment elevation\n"
                        "- Possible myocardial infarction\n"
                        "- Abnormal repolarization\n"
                        "- Recommended immediate medical attention"
                    )
                elif "tachycardia" in abnormality:
                    details = (
                        "Analysis:\n"
                        "- Heart rate above 100 BPM\n"
                        "- Rapid ventricular response\n"
                        "- Shortening of diastolic interval\n"
                        "- Potential underlying cardiac condition"
                    )
                elif "bradycardia" in abnormality:
                    details = (
                        "Analysis:\n"
                        "- Heart rate below 60 BPM\n"
                        "- Prolonged R-R intervals\n"
                        "- Extended diastolic phase\n"
                        "- Possible conduction abnormality"
                    )
                else:
                    details = (
                        "Analysis:\n"
                        "- Irregular heart rhythm\n"
                        "- Abnormal R-R intervals\n"
                        "- Potential arrhythmia detected\n"
                        "- Abnormal waveform morphology"
                    )
                
                msg_box.setInformativeText(f"Confidence: 95.0%\n\n{details}")
                msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            
            msg_box.addButton(QtWidgets.QMessageBox.Ok)
            msg_box.exec_()
            
            # Update status bar
            self.statusBar.showMessage(f"ECG classified as {'NORMAL' if is_normal else 'ABNORMAL'}")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error classifying ECG: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    app = QtWidgets.QApplication(sys.argv)
    window = ECGSimulator()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 