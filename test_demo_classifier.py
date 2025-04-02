#!/usr/bin/env python
"""
Test script for evaluating the ML Model 2.0 demo classifier performance
using real ECG samples.
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Import the ML classifier implementation
from integrate_ml_classifier import MLClassifier

def load_ecg_from_csv(file_path):
    """Load ECG data from a CSV file."""
    time_values = []
    ecg_values = []
    
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header if exists
            
            for row in reader:
                if len(row) >= 2:
                    try:
                        # First column is usually time, second is ECG value
                        time_values.append(float(row[0]))
                        ecg_values.append(float(row[1]))
                    except (ValueError, IndexError):
                        continue
        
        if not time_values:  # If time values couldn't be parsed
            time_values = list(range(len(ecg_values)))
            
        return np.array(time_values), np.array(ecg_values)
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([]), np.array([])

def get_expected_label(file_path):
    """Get expected label based on filename."""
    filename = os.path.basename(file_path).lower()
    
    if 'normal' in filename:
        return 'Normal'
    elif 'bradycardia' in filename:
        return 'Bradycardia'
    elif 'tachycardia' in filename:
        return 'Tachycardia'
    elif 'afib' in filename or 'fibrillation' in filename:
        return 'Atrial Fibrillation'
    elif 'st' in filename and ('elev' in filename or 'elevation' in filename):
        return 'ST Elevation'
    elif 'abnormal' in filename:
        return 'Abnormal'
    else:
        # Default if we can't determine
        return 'Unknown'

def find_ecg_files(directory):
    """Find all ECG CSV files in a directory and its subdirectories."""
    ecg_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                # Skip files that are clearly not ECG data
                if 'metadata' in file.lower() or 'readme' in file.lower():
                    continue
                    
                file_path = os.path.join(root, file)
                ecg_files.append(file_path)
    
    return ecg_files

def main():
    """Main function to test the demo classifier."""
    print("=== ML Model 2.0 Demo Classifier Performance Test ===\n")
    
    # Create classifier with forced demo mode
    classifier = MLClassifier()
    classifier.use_dummy_model = True
    
    # Find ECG files - check some common directories
    search_dirs = [
        'Samples', 
        'data', 
        'ecg_samples', 
        'normal_ecg_samples', 
        'mitbih_samples',
        'mitbih_samples_improved',
        'large_test_dataset',
        'recordings'
    ]
    all_files = []
    
    for directory in search_dirs:
        if os.path.exists(directory):
            print(f"Looking for ECG files in {directory}...")
            files = find_ecg_files(directory)
            all_files.extend(files)
            print(f"Found {len(files)} files")
    
    if not all_files:
        print("No ECG files found. Please provide a directory with ECG CSV files.")
        return
    
    print(f"\nTesting classifier on {len(all_files)} ECG files...")
    
    # Prepare for results
    results = []
    y_true = []
    y_pred = []
    
    # Process each file
    for file_path in all_files:
        # Get expected label from filename
        expected_label = get_expected_label(file_path)
        
        # If we can't determine expected label, skip this file
        if expected_label == 'Unknown':
            continue
            
        # Load ECG data
        time_values, ecg_values = load_ecg_from_csv(file_path)
        
        if len(ecg_values) < 100:
            print(f"Skipping {file_path} - not enough data points")
            continue
        
        # Classify with demo model
        classification = classifier._dummy_classify_ecg(ecg_values, time_values)
        
        if not classification['success']:
            print(f"Classification failed for {file_path}")
            continue
        
        # Get predicted class
        predicted_label = classification['class']
        confidence = classification['confidence']
        heart_rate = classification.get('heart_rate', 'N/A')
        
        # Save results
        results.append({
            'file': os.path.basename(file_path),
            'expected': expected_label,
            'predicted': predicted_label,
            'confidence': confidence,
            'heart_rate': heart_rate,
            'correct': expected_label == predicted_label
        })
        
        y_true.append(expected_label)
        y_pred.append(predicted_label)
        
        # Print progress
        print(f"Processed: {file_path} - Expected: {expected_label}, Predicted: {predicted_label}, Confidence: {confidence:.2f}")
    
    # Calculate metrics
    if results:
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Print results
        print("\n=== Performance Results ===")
        print(f"Total samples: {len(results)}")
        print(f"Correct classifications: {sum(r['correct'] for r in results)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Create confusion matrix
        labels = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("Predicted →")
        print("Actual ↓")
        
        # Header row with predicted classes
        header = "       " + "  ".join(f"{label[:7]:7}" for label in labels)
        print(header)
        
        # Print each row
        for i, label in enumerate(labels):
            row = f"{label[:7]:7}" + "  ".join(f"{cm[i, j]:7d}" for j in range(len(labels)))
            print(row)
        
        # Per-class performance
        print("\nPer-Class Performance:")
        
        # Calculate per-class metrics
        class_metrics = {}
        for cls in labels:
            cls_true = [1 if t == cls else 0 for t in y_true]
            cls_pred = [1 if p == cls else 0 for p in y_pred]
            
            # Calculate metrics
            tp = sum(1 for t, p in zip(cls_true, cls_pred) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(cls_true, cls_pred) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(cls_true, cls_pred) if t == 1 and p == 0)
            
            # Precision, recall, F1
            cls_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            cls_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
            
            class_metrics[cls] = {
                'precision': cls_precision,
                'recall': cls_recall,
                'f1': cls_f1,
                'count': sum(cls_true)
            }
        
        # Print class metrics
        for cls, metrics in class_metrics.items():
            print(f"{cls} (n={metrics['count']}):")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            
        # Generate report
        print("\nSaving results to 'demo_classifier_report.csv'")
        df = pd.DataFrame(results)
        df.to_csv("demo_classifier_report.csv", index=False)
        
        # Print summary list of incorrect classifications
        wrong_classifications = [r for r in results if not r['correct']]
        if wrong_classifications:
            print("\nIncorrect Classifications:")
            for i, wrong in enumerate(wrong_classifications[:10], 1):  # Show first 10
                print(f"{i}. {wrong['file']}: Expected {wrong['expected']}, Got {wrong['predicted']}")
            
            if len(wrong_classifications) > 10:
                print(f"...and {len(wrong_classifications) - 10} more")
    else:
        print("No classification results to analyze.")

if __name__ == "__main__":
    main() 