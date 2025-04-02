#!/usr/bin/env python
"""
Test Script for ML Model 2.0 Integration

This script tests the ML Model 2.0 classifier integration without modifying
the existing ECG-Viewer code.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Check if the required files exist
required_files = [
    'integrate_ml_classifier.py', 
    'ml_classifier_ui.py'
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print(f"Error: Missing required files: {', '.join(missing_files)}")
    print("Please run this script from the directory where these files are located.")
    sys.exit(1)

# Import our ML classifier
from integrate_ml_classifier import MLClassifier

def test_ml_classifier():
    """Test the ML Model 2.0 classifier on synthetic ECG data"""
    print("Testing ML Model 2.0 integration...")
    
    # Initialize the classifier
    classifier = MLClassifier()
    
    if classifier.model is None:
        print("\nError: Failed to load the ML Model 2.0.")
        print("Please ensure the ECG-Arrhythmia-Classifier-main directory contains model.pkl")
        return False
    
    print("\nML Model 2.0 loaded successfully!")
    print("This is the advanced machine learning model with 98.8% accuracy")
    
    # Generate synthetic normal ECG
    print("\nGenerating synthetic ECG data...")
    t = np.linspace(0, 10, 2500)  # 10 seconds at 250Hz
    ecg_normal = np.zeros_like(t)
    
    # Create synthetic normal heartbeats
    for i in range(12):  # 12 beats in 10 seconds = 72 BPM
        beat_center = i * 0.83  # Position of each heartbeat
        # P wave
        ecg_normal += 0.25 * np.exp(-((t - beat_center + 0.2) ** 2) / 0.001)
        # QRS complex
        ecg_normal += 1.0 * np.exp(-((t - beat_center) ** 2) / 0.0002)
        # S wave
        ecg_normal -= 0.3 * np.exp(-((t - beat_center + 0.05) ** 2) / 0.0002)
        # T wave
        ecg_normal += 0.3 * np.exp(-((t - beat_center - 0.15) ** 2) / 0.002)
    
    # Add noise
    ecg_normal += np.random.normal(0, 0.05, len(t))
    
    # Generate synthetic abnormal ECG (simulating AFib)
    ecg_afib = np.zeros_like(t)
    
    # Create irregular heartbeat pattern with missing P waves
    beat_intervals = []
    for _ in range(20):  # More beats, but irregular
        beat_intervals.append(np.random.uniform(0.4, 0.9))  # Random intervals
    
    # Normalize to fit in 10 seconds
    beat_intervals = np.array(beat_intervals) * (10 / sum(beat_intervals))
    
    # Create each beat
    beat_positions = np.cumsum(beat_intervals)
    for pos in beat_positions:
        if pos < 10:  # Ensure we're still within the 10 second window
            # No P wave for AFib
            # QRS complex - slightly wider
            ecg_afib += 1.0 * np.exp(-((t - pos) ** 2) / 0.0003)
            # S wave
            ecg_afib -= 0.4 * np.exp(-((t - pos + 0.06) ** 2) / 0.0003)
            # T wave - slightly different
            ecg_afib += 0.25 * np.exp(-((t - pos - 0.2) ** 2) / 0.003)
    
    # Add more baseline noise for AFib
    ecg_afib += np.random.normal(0, 0.08, len(t))
    
    # Test classification
    print("\nClassifying normal ECG with ML Model 2.0...")
    normal_result = classifier.classify_ecg(ecg_normal, t)
    
    print("\nClassifying abnormal ECG (simulated AFib) with ML Model 2.0...")
    afib_result = classifier.classify_ecg(ecg_afib, t)
    
    # Display results
    print("\nML Model 2.0 Classification Results:")
    print(f"Normal ECG classified as: {normal_result['class']} with confidence {normal_result.get('confidence', 0):.4f}")
    print(f"Abnormal ECG classified as: {afib_result['class']} with confidence {afib_result.get('confidence', 0):.4f}")
    
    # Plot the ECGs
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, ecg_normal)
    plt.title(f"Normal ECG - ML Model 2.0 Classification: {normal_result['class']}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, ecg_afib)
    plt.title(f"Abnormal ECG (AFib) - ML Model 2.0 Classification: {afib_result['class']}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle("ML Model 2.0 Classification Results (98.8% Accuracy)", fontsize=14)
    plt.subplots_adjust(top=0.9)
    
    # Save the plot
    plt.savefig("ml_model_2.0_test_results.png")
    print("\nTest plots saved to 'ml_model_2.0_test_results.png'")
    
    # Optionally display the plot
    try:
        plt.show()
    except:
        print("Note: Could not display the plot (headless environment)")
    
    return True

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*50)
    print("NEXT STEPS FOR ML MODEL 2.0 INTEGRATION:")
    print("="*50)
    print("\n1. To fully integrate the ML Model 2.0 into your ECG-Viewer:")
    print("   - Run 'python integrate_to_ecg_viewer.py' for detailed instructions")
    print("   - Follow the instructions to modify your ECG-Viewer code")
    
    print("\n2. The integration consists of these files:")
    print("   - integrate_ml_classifier.py: Core integration with the ML Model 2.0")
    print("   - ml_classifier_ui.py: UI components for the ML Model 2.0 classifier")
    print("   - integrate_to_ecg_viewer.py: Instructions for full integration")
    print("   - ECG-Arrhythmia-Classifier-main/: The ML Model 2.0 and its dependencies")
    
    print("\n3. Once integrated, you will have:")
    print("   - A toggle button to switch between the original model and ML Model 2.0")
    print("   - A display of ML Model 2.0 classification results with confidence scores")
    print("   - Detailed probabilities for each class of arrhythmia")
    
    print("\n4. Comparison with Original Model:")
    print("   - ML Model 2.0 (98.8% accuracy) vs. Original rule-based model (~85-90% estimated)")
    print("   - ML Model 2.0 provides 5 detailed categories vs. basic categories in original model")
    print("   - ML Model 2.0 provides detailed confidence scores for each category")

if __name__ == "__main__":
    success = test_ml_classifier()
    if success:
        print_next_steps()
    else:
        print("\nML Model 2.0 test failed. Please check the error messages above.") 