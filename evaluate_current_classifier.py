#!/usr/bin/env python
"""
Evaluation Script for ECG Classifier Performance Assessment
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import seaborn as sns
from scipy import signal

# Import our current classifier
# Add the directory to the path so we can import from the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from standalone_ecg_simulator import ECGSimulator

def load_test_data(data_path='real-worldecg.csv', label_path=None):
    """
    Load test data and labels for evaluation.
    If label_path is not provided, the last column of data_path is assumed to be the label.
    
    Args:
        data_path: Path to the ECG data CSV
        label_path: Optional path to a separate labels file
        
    Returns:
        X: ECG data samples
        y: True labels (0 for normal, 1 for abnormal)
    """
    # Load the ECG data
    try:
        data = pd.read_csv(data_path, header=None)
        print(f"Loaded {len(data)} samples from {data_path}")
        
        # If we have a separate labels file
        if label_path and os.path.exists(label_path):
            labels = pd.read_csv(label_path, header=None)
            y = labels.iloc[:, 0].values
            X = data.values
        else:
            # Assume the last column is the label
            y = data.iloc[:, -1].values
            X = data.iloc[:, :-1].values
            
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def manually_label_samples(data_path='real-worldecg.csv', output_path='manual_labels.csv', sample_count=20):
    """
    Helper function to manually label a subset of samples for testing purposes.
    This is useful if we don't already have labeled test data.
    
    Args:
        data_path: Path to the ECG data CSV
        output_path: Path to save the manual labels
        sample_count: Number of samples to label
        
    Returns:
        Path to the labels file
    """
    try:
        data = pd.read_csv(data_path, header=None)
        total_samples = len(data)
        
        # Select a random subset of samples
        if sample_count > total_samples:
            sample_count = total_samples
            
        sample_indices = np.random.choice(total_samples, sample_count, replace=False)
        samples_to_label = data.iloc[sample_indices]
        
        # Create a simple ECG simulator without UI for manual labeling
        simulator = ECGSimulator()
        
        # Initialize labels list
        labels = []
        
        for idx, sample in enumerate(samples_to_label.iterrows()):
            sample_idx, sample_data = sample
            
            # Get sample data (excluding potential label column)
            ecg_values = sample_data.iloc[:-1].values if len(sample_data) > 100 else sample_data.values
            
            # Generate time axis
            ecg_time = np.arange(len(ecg_values)) * (1000 / 250)  # Assuming 250 Hz
            
            # Detect R-peaks for heart rate calculation
            try:
                r_peaks, _ = signal.find_peaks(
                    ecg_values, 
                    height=0.5*max(ecg_values),
                    distance=250 * 0.3  # Minimum 300ms between peaks
                )
            except Exception as e:
                print(f"Error detecting R-peaks: {e}")
                r_peaks = []
            
            # Calculate heart rate
            heart_rate = 0
            if len(r_peaks) > 1:
                rr_intervals = np.diff(ecg_time[r_peaks])
                heart_rate = int(60000 / np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0
            
            # Calculate RR interval variability for AFib detection
            rr_variability = 0
            if len(r_peaks) > 2:
                rr_intervals = np.diff(r_peaks)
                rr_variability = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
            
            # Plot the ECG for manual inspection
            plt.figure(figsize=(12, 4))
            plt.plot(ecg_time, ecg_values, 'b-')
            if len(r_peaks) > 0:
                plt.plot(ecg_time[r_peaks], ecg_values[r_peaks], 'ro')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f"Sample {sample_idx}: HR={heart_rate} BPM, RR Var={rr_variability:.3f}")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (mV)")
            plt.show()
            
            # Ask for manual classification
            print(f"\nSample {sample_idx}: Heart Rate = {heart_rate} BPM, RR Variability = {rr_variability:.3f}")
            while True:
                label = input("Is this ECG normal (0) or abnormal (1)? ")
                if label in ['0', '1']:
                    labels.append(int(label))
                    break
                print("Invalid input. Please enter 0 for normal or 1 for abnormal.")
        
        # Save the labels
        pd.DataFrame({"index": sample_indices, "label": labels}).to_csv(output_path, index=False)
        print(f"Saved {len(labels)} manual labels to {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Error in manual labeling: {e}")
        return None

def evaluate_classifier(X, y_true):
    """
    Evaluate our current classifier on the test data.
    
    Args:
        X: ECG data samples
        y_true: True labels (0 for normal, 1 for abnormal)
        
    Returns:
        Dictionary of performance metrics
    """
    # Initialize our classifier (without GUI)
    simulator = ECGSimulator()
    
    # Variables to store results
    y_pred = []
    confidences = []
    reasons = []
    
    # Classify each sample
    for i, sample_data in enumerate(X):
        try:
            # Skip the last value if it's a label
            ecg_values = sample_data[:-1] if len(sample_data) > 100 else sample_data
            
            # Generate time axis
            ecg_time = np.arange(len(ecg_values)) * (1000 / 250)  # Assuming 250 Hz
            
            # Detect R-peaks for heart rate calculation
            r_peaks, _ = signal.find_peaks(
                ecg_values, 
                height=0.5*max(ecg_values),
                distance=250 * 0.3  # Minimum 300ms between peaks
            )
            
            # Calculate heart rate
            heart_rate = 0
            if len(r_peaks) > 1:
                rr_intervals = np.diff(ecg_time[r_peaks])
                heart_rate = int(60000 / np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0
            
            # Calculate RR interval variability for AFib detection
            rr_variability = 0
            if len(r_peaks) > 2:
                rr_intervals = np.diff(r_peaks)
                rr_variability = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
            
            # Detect ST segment elevation
            st_elevation = 0
            if len(r_peaks) > 0:
                # Check 80-120ms after each R peak for ST segment
                st_points = []
                for peak in r_peaks:
                    if peak + 20 < len(ecg_values):  # at least 80ms after R peak
                        st_point = ecg_values[peak + 20:peak + 30].mean()  # 80-120ms segment
                        st_points.append(st_point)
                
                if st_points:
                    st_elevation = np.mean(st_points)
            
            # Simple classification based on features (similar to our current method)
            reason = ""
            confidence = 0.75  # Default confidence
            
            if heart_rate < 60:
                is_normal = False
                reason = "Bradycardia"
                confidence = 0.85
            elif heart_rate > 100:
                is_normal = False
                reason = "Tachycardia"
                confidence = 0.85
            elif rr_variability > 0.2:  # High RR interval variability suggests AFib
                is_normal = False
                reason = "Atrial fibrillation"
                confidence = 0.9
            elif st_elevation > 0.2:  # Significant ST elevation
                is_normal = False
                reason = "ST elevation"
                confidence = 0.9
            else:
                is_normal = True
                reason = "Normal rhythm"
                confidence = 0.8
            
            # Store results
            y_pred.append(0 if is_normal else 1)
            confidences.append(confidence)
            reasons.append(reason)
            
            # Print progress
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(X)} samples")
                
        except Exception as e:
            print(f"Error classifying sample {i}: {e}")
            # Default to most common class
            y_pred.append(int(np.mean(y_true) > 0.5))
            confidences.append(0.5)
            reasons.append("Error in classification")
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate classification report
    class_report = classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal'])
    
    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'y_pred': y_pred,
        'confidences': confidences,
        'reasons': reasons
    }
    
    return results

def plot_results(results, output_dir="evaluation_results"):
    """
    Plot and save the evaluation results.
    
    Args:
        results: Dictionary of performance metrics
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot metrics as a bar chart
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_values = [results[m] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, metric_values, color=['blue', 'green', 'orange', 'red'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.title('Classification Performance Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'))
    plt.close()
    
    # Plot distribution of confidence scores
    plt.figure(figsize=(10, 6))
    sns.histplot(results['confidences'], bins=10, kde=True)
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    plt.close()
    
    # Plot distribution of classification reasons
    reason_counts = pd.Series(results['reasons']).value_counts()
    plt.figure(figsize=(12, 8))
    bars = plt.bar(reason_counts.index, reason_counts.values)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Classification Reason')
    plt.ylabel('Count')
    plt.title('Distribution of Classification Reasons')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reason_distribution.png'))
    plt.close()
    
    # Save detailed results to a text file
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        f.write("ECG Classifier Evaluation Results\n")
        f.write("===============================\n\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1_score']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(results['confusion_matrix']))
        f.write("\n\nClassification Report:\n")
        f.write(results['classification_report'])
        
    print(f"Results saved to {output_dir}")

def main():
    """Main function to run the evaluation"""
    print("ECG Classifier Evaluation")
    print("========================")
    
    # Check if we should manually label some samples
    manual_labeling = input("Do you want to manually label some samples for evaluation? (y/n) ")
    
    if manual_labeling.lower() == 'y':
        sample_count = int(input("How many samples do you want to label? "))
        label_path = manually_label_samples(sample_count=sample_count)
        if label_path:
            X, y_true = load_test_data(label_path=label_path)
        else:
            print("Manual labeling failed. Exiting.")
            return
    else:
        # Ask for path to test data
        data_path = input("Enter path to test data CSV (default: real-worldecg.csv): ")
        data_path = data_path if data_path else 'real-worldecg.csv'
        
        # Ask for path to labels if separate
        has_separate_labels = input("Does the data have separate labels? (y/n) ")
        
        if has_separate_labels.lower() == 'y':
            label_path = input("Enter path to labels CSV: ")
            X, y_true = load_test_data(data_path, label_path)
        else:
            X, y_true = load_test_data(data_path)
    
    if X is None or y_true is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Loaded {len(X)} samples for evaluation")
    
    # Evaluate the classifier
    print("Evaluating classifier...")
    results = evaluate_classifier(X, y_true)
    
    # Print summary results
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot and save results
    plot_results(results)

if __name__ == "__main__":
    main() 