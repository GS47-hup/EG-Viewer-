import sys
import os
import shutil
import logging
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

# Import our classifier
from ecg_classifier import ECGClassifier

# Import original ECG viewer modules
import ecg_viewer
from ecg_viewer_window import Ui_MainWindow

# Configure logging
logging.basicConfig(level=logging.INFO)

class ClassificationResultDialog(QtWidgets.QDialog):
    """Dialog to show ECG classification results"""
    
    def __init__(self, classification, confidence, reasons, parent=None):
        super(ClassificationResultDialog, self).__init__(parent)
        
        # Set dialog properties
        self.setWindowTitle("ECG Classification Results")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        
        # Set window icon if available
        try:
            self.setWindowIcon(QtGui.QIcon(':/icon/icon.png'))
        except:
            pass
        
        # Create layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create result label with large font
        result_font = QtGui.QFont()
        result_font.setPointSize(16)
        result_font.setBold(True)
        
        result_text = "NORMAL" if classification.lower() == "normal" else "ABNORMAL"
        result_color = "#28a745" if classification.lower() == "normal" else "#dc3545"
        
        result_label = QtWidgets.QLabel(result_text)
        result_label.setFont(result_font)
        result_label.setStyleSheet(f"color: {result_color}; padding: 10px;")
        result_label.setAlignment(QtCore.Qt.AlignCenter)
        
        # Create confidence label
        confidence_label = QtWidgets.QLabel(f"Confidence: {confidence:.1f}%")
        confidence_label.setAlignment(QtCore.Qt.AlignCenter)
        
        # Create reasons group box
        reasons_box = QtWidgets.QGroupBox("Analysis Details")
        reasons_layout = QtWidgets.QVBoxLayout()
        
        # Add each reason as a label
        for reason in reasons:
            reason_label = QtWidgets.QLabel(reason)
            reason_label.setWordWrap(True)
            reasons_layout.addWidget(reason_label)
        
        reasons_box.setLayout(reasons_layout)
        
        # Create disclaimer
        disclaimer = QtWidgets.QLabel(
            "<b>DISCLAIMER:</b> This analysis is for educational purposes only. "
            "NOT FOR MEDICAL USE! Please consult a healthcare professional for "
            "actual medical advice."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("color: #856404; background-color: #fff3cd; padding: 8px;")
        
        # Create close button
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.accept)
        
        # Add widgets to layout
        layout.addWidget(result_label)
        layout.addWidget(confidence_label)
        layout.addWidget(reasons_box)
        layout.addStretch()
        layout.addWidget(disclaimer)
        layout.addWidget(close_button)
        
        self.setLayout(layout)


class ECGViewerWithClassification(ecg_viewer.ECGViewer):
    """
    Extended ECG Viewer with classification capabilities
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize the parent class
        super(ECGViewerWithClassification, self).__init__(*args, **kwargs)
        
        # Initialize the classifier
        self.classifier = ECGClassifier()
        
        # Add a classification button to the UI
        self.button_classify = QtWidgets.QPushButton("Classify ECG")
        self.button_classify.setToolTip("Analyze the current ECG signal and classify as normal or abnormal")
        self.button_classify.clicked.connect(self.classify_current_ecg)
        
        # Find a place to add the button in the existing UI
        self.gridLayout_2.addWidget(self.button_classify, 3, 2, 1, 1)
        
        # Add a label to show classification status
        self.classification_label = QtWidgets.QLabel("Classification: Not analyzed")
        self.classification_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout_2.addWidget(self.classification_label, 4, 0, 1, 3)
        
        # Create directory for saving classification results
        self.results_dir = "classification_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def classify_current_ecg(self):
        """Classify the currently displayed ECG signal"""
        # Check if we have data
        if len(self.value_history) == 0 or np.all(self.value_history == 0):
            self.ui_display_error_message("Classification Error", 
                                         "No ECG data available to classify. Please connect a device or load sample data.")
            return
        
        # Check if we have sufficient data (at least a few seconds)
        if len(self.value_history) < self.sample_rate * 3:
            self.ui_display_error_message("Classification Error", 
                                         "Insufficient ECG data for classification. Please capture at least a few seconds of data.")
            return
        
        # Save current ECG to a temporary file
        temp_file = os.path.join(self.results_dir, "current_ecg_for_classification.csv")
        self.save_current_ecg_to_file(temp_file)
        
        # Set status message
        self.ui_statusbar_message("Analyzing ECG signal...")
        
        try:
            # Run classification
            classification, confidence, reasons = self.classifier.classify_ecg(temp_file)
            
            # Update status label
            status_color = "#28a745" if classification.lower() == "normal" else "#dc3545"
            self.classification_label.setText(f"Classification: {classification.upper()} ({confidence:.1f}%)")
            self.classification_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")
            
            # Save classification plot
            plot_file = self.classifier.plot_ecg_with_analysis(temp_file)
            
            # Show results dialog
            dialog = ClassificationResultDialog(classification, confidence, reasons, self)
            dialog.exec_()
            
            # Update status bar
            self.ui_statusbar_message(f"ECG classified as {classification.upper()}")
            
        except Exception as e:
            logging.error(f"Classification error: {e}")
            self.ui_display_error_message("Classification Error", 
                                         f"An error occurred during ECG classification: {str(e)}")
    
    def save_current_ecg_to_file(self, file_path):
        """Save the current ECG data to a CSV file for classification"""
        try:
            import pandas as pd
            
            # Create DataFrame from current data
            df = pd.DataFrame({
                'time': self.time_history,
                'value': self.value_history
            })
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            logging.info(f"Saved current ECG data to {file_path}")
            
            return True
        except Exception as e:
            logging.error(f"Error saving ECG data: {e}")
            return False


def main():
    """Main entry point for the ECG Viewer with Classification"""
    app = QtWidgets.QApplication(sys.argv)
    window = ECGViewerWithClassification()
    window.setWindowTitle("ECG Viewer with Classification - " + ecg_viewer.VERSION)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 