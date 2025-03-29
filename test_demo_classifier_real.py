#!/usr/bin/env python
"""
Test script for evaluating the ML Model 2.0 demo classifier performance
using the real-worldecg.csv dataset containing real ECG data with known labels.
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Import the ML classifier implementation
from integrate_ml_classifier import MLClassifier

def load_ecg_from_csv_line(line):
    """Parse a line of CSV containing ECG values with the last value as a label."""
    values = line.strip().split(',')
    
    # Last value should be a label (0 or 1)
    try:
        label = int(float(values[-1]))
        ecg_values = [float(x) for x in values[:-1]]  # All except the last value
        return ecg_values, label
    except (ValueError, IndexError) as e:
        print(f"Error parsing line: {e}")
        return None, None

def main():
    """Main function to test the demo classifier on real-world ECG dataset."""
    print("=== ML Model 2.0 Demo Classifier Performance Test ===")
    print("Using real-worldecg.csv dataset\n")
    
    # Create classifier with forced demo mode
    classifier = MLClassifier()
    classifier.use_dummy_model = True
    
    # Check if the file exists
    file_path = "real-worldecg.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return
    
    # Load the dataset
    try:
        # Count lines first
        with open(file_path, 'r') as f:
            line_count = sum(1 for line in f)
        
        print(f"Found {line_count} samples in the dataset")
        
        # Process the file
        samples = []
        normal_count = 0
        abnormal_count = 0
        
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                ecg_values, label = load_ecg_from_csv_line(line)
                
                if ecg_values is not None:
                    samples.append((ecg_values, label))
                    
                    if label == 0:
                        normal_count += 1
                    else:
                        abnormal_count += 1
                        
                if (i+1) % 1000 == 0:
                    print(f"Loaded {i+1} samples...")
        
        print(f"\nSuccessfully loaded {len(samples)} samples")
        print(f"Normal samples: {normal_count}")
        print(f"Abnormal samples: {abnormal_count}")
        
        # Prepare results storage
        results = []
        y_true = []
        y_pred = []
        
        # Process each sample
        print("\nProcessing samples for classification...")
        
        for i, (ecg_values, true_label) in enumerate(samples):
            # Generate time values
            time_values = list(range(len(ecg_values)))
            
            # Get the true label
            true_class = 'Normal' if true_label == 0 else 'Abnormal'
            
            # Classify with demo model
            classification = classifier._dummy_classify_ecg(ecg_values, time_values)
            
            if not classification['success']:
                print(f"Classification failed for sample {i}")
                continue
            
            # Get predicted class
            predicted_label = classification['class']
            confidence = classification['confidence']
            heart_rate = classification.get('heart_rate', 'N/A')
            
            # Map model output to binary classification
            # Normal = 0, everything else = 1 (abnormal)
            binary_prediction = 0 if predicted_label == 'Normal' else 1
            
            # For binary metrics, use 0/1 format
            y_true.append(true_label)
            y_pred.append(binary_prediction)
            
            # Save detailed results
            results.append({
                'sample_id': i,
                'true_label': true_label,
                'true_class': true_class,
                'predicted_class': predicted_label,
                'detailed_class': predicted_label,
                'binary_prediction': binary_prediction,
                'confidence': confidence,
                'heart_rate': heart_rate,
                'correct': true_label == binary_prediction
            })
            
            # Print progress occasionally
            if (i+1) % 100 == 0:
                correct_so_far = sum(r['correct'] for r in results)
                print(f"Processed {i+1}/{len(samples)} samples... Accuracy so far: {correct_so_far/(i+1):.4f}")
        
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
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            
            # Print confusion matrix
            print("\nConfusion Matrix:")
            print("             Predicted")
            print("             Normal(0)  Abnormal(1)")
            print(f"Actual Normal(0)    {cm[0,0]:6d}     {cm[0,1]:6d}")
            print(f"Actual Abnormal(1)  {cm[1,0]:6d}     {cm[1,1]:6d}")
            
            # Calculate per-class metrics
            tn, fp, fn, tp = cm.ravel()
            
            # Normal class (0)
            normal_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
            normal_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
            normal_f1 = 2 * normal_precision * normal_recall / (normal_precision + normal_recall) if (normal_precision + normal_recall) > 0 else 0
            
            # Abnormal class (1)
            abnormal_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            abnormal_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            abnormal_f1 = 2 * abnormal_precision * abnormal_recall / (abnormal_precision + abnormal_recall) if (abnormal_precision + abnormal_recall) > 0 else 0
            
            # Print per-class metrics
            print("\nPer-Class Performance:")
            print(f"Normal (n={tn+fp}):")
            print(f"  Precision: {normal_precision:.4f}")
            print(f"  Recall: {normal_recall:.4f}")
            print(f"  F1 Score: {normal_f1:.4f}")
            
            print(f"Abnormal (n={fn+tp}):")
            print(f"  Precision: {abnormal_precision:.4f}")
            print(f"  Recall: {abnormal_recall:.4f}")
            print(f"  F1 Score: {abnormal_f1:.4f}")
            
            # Count detailed classifications
            detailed_classes = {}
            for r in results:
                cls = r['detailed_class']
                if cls not in detailed_classes:
                    detailed_classes[cls] = {'total': 0, 'correct': 0}
                
                detailed_classes[cls]['total'] += 1
                if r['correct']:
                    detailed_classes[cls]['correct'] += 1
            
            # Print detailed class breakdowns
            print("\nDetailed Classification Breakdown:")
            for cls, stats in detailed_classes.items():
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                print(f"{cls}: {stats['total']} samples, {stats['correct']} correct, Accuracy: {accuracy:.4f}")
            
            # Generate report
            print("\nSaving results to 'real_world_classifier_report.csv'")
            df_results = pd.DataFrame(results)
            df_results.to_csv("real_world_classifier_report.csv", index=False)
            
        else:
            print("No classification results to analyze.")
            
    except Exception as e:
        print(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 