import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import joblib
import argparse
from real_ecg_classifier import RealEcgClassifier

def load_and_plot_sample(file_path, sample_index, save_path=None):
    """
    Load and plot a specific ECG sample
    
    Args:
        file_path: Path to the CSV file
        sample_index: Index of the sample to plot
        save_path: Path to save the plot
    """
    # Load the data
    data = pd.read_csv(file_path, header=None)
    
    # Check if index is valid
    if sample_index >= len(data):
        print(f"Error: Sample index {sample_index} is out of range. File has {len(data)} samples.")
        return None
    
    # Get the sample and label
    sample = data.iloc[sample_index, :-1].values
    label = data.iloc[sample_index, -1]
    
    # Plot the sample
    plt.figure(figsize=(12, 6))
    plt.plot(sample)
    plt.title(f"ECG Sample {sample_index} - {'Abnormal' if label == 1 else 'Normal'}")
    plt.xlabel("Time Point")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Sample plot saved to {save_path}")
    else:
        plt.show()
    
    return sample, label

def predict_sample(classifier, sample):
    """
    Make a prediction for a single ECG sample
    
    Args:
        classifier: Trained RealEcgClassifier
        sample: ECG sample to classify
        
    Returns:
        Prediction and probability
    """
    # Make prediction
    prediction = classifier.predict(sample.reshape(1, -1))[0]
    
    # Get probability
    try:
        probability = classifier.predict_proba(sample.reshape(1, -1))[0]
    except:
        probability = [0, 0]
    
    return prediction, probability

def visualize_feature_importance(classifier, save_path=None):
    """
    Visualize feature importance from the classifier
    
    Args:
        classifier: Trained RealEcgClassifier
        save_path: Path to save the visualization
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(classifier.model, 'feature_importances_'):
        print("Model does not have feature importance information")
        return
    
    # Get feature importances
    importances = classifier.model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Get feature names
    feature_names = classifier.features
    
    # Plot the feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices],
           align="center")
    plt.xticks(range(len(importances)),
              [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, min(20, len(importances))])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()

def visualize_predictions(X_test, y_test, y_pred, indices=None, save_path=None):
    """
    Visualize ECG samples with their true and predicted labels
    
    Args:
        X_test: Test ECG samples
        y_test: True labels
        y_pred: Predicted labels
        indices: Specific sample indices to plot, if None will select random samples
        save_path: Path to save the visualization
    """
    # If indices not specified, select random samples
    if indices is None:
        # Select some correctly classified and misclassified samples
        correct_indices = np.where(y_test == y_pred)[0]
        wrong_indices = np.where(y_test != y_pred)[0]
        
        # Select up to 2 random correct and 2 random wrong if available
        n_correct = min(2, len(correct_indices))
        n_wrong = min(2, len(wrong_indices))
        
        if n_correct > 0:
            correct_samples = np.random.choice(correct_indices, size=n_correct, replace=False)
        else:
            correct_samples = []
            
        if n_wrong > 0:
            wrong_samples = np.random.choice(wrong_indices, size=n_wrong, replace=False)
        else:
            wrong_samples = []
            
        indices = np.concatenate([correct_samples, wrong_samples])
    
    # Number of samples to plot
    n_samples = len(indices)
    
    if n_samples == 0:
        print("No samples to visualize")
        return
    
    # Create plot
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))
    
    # Handle single subplot case
    if n_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Plot ECG signal
        axes[i].plot(X_test[idx])
        axes[i].set_title(f"Sample {idx}: True: {'Abnormal' if y_test[idx] == 1 else 'Normal'}, "
                         f"Predicted: {'Abnormal' if y_pred[idx] == 1 else 'Normal'}")
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Prediction visualization saved to {save_path}")
    else:
        plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test ECG classifier on real data')
    parser.add_argument('--data', type=str, default='Real ECG.csv',
                        help='Path to the CSV file with ECG data')
    parser.add_argument('--model', type=str, default='real_ecg_data/best_model.joblib',
                        help='Path to the trained model file')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--sample', type=int, default=None,
                        help='Index of a specific sample to predict')
    parser.add_argument('--output_dir', type=str, default='real_ecg_data',
                        help='Directory to save output files')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        return
    
    # Load the trained classifier
    print(f"Loading model from {args.model}...")
    classifier = RealEcgClassifier.load_model(args.model)
    
    # If a specific sample index is provided
    if args.sample is not None:
        # Load and plot the sample
        sample, true_label = load_and_plot_sample(
            args.data, args.sample, 
            save_path=os.path.join(args.output_dir, f'sample_{args.sample}.png')
        )
        
        if sample is not None:
            # Make prediction
            prediction, probability = predict_sample(classifier, sample)
            
            # Print results
            print(f"\nSample {args.sample} Prediction:")
            print(f"True Label: {'Abnormal' if true_label == 1 else 'Normal'}")
            print(f"Predicted: {'Abnormal' if prediction == 1 else 'Normal'}")
            print(f"Confidence: {probability[int(prediction)]:.2%}")
            
        return
    
    # Load the dataset
    print(f"Loading ECG dataset from {args.data}...")
    try:
        ecg_data = pd.read_csv(args.data, header=None)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Separate features and labels
    X = ecg_data.iloc[:, :-1].values
    y = ecg_data.iloc[:, -1].values
    
    # Split data into training and test sets
    _, X_test, _, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
    
    print(f"Dataset shape: {ecg_data.shape}")
    print(f"Number of normal samples: {np.sum(y == 0)}")
    print(f"Number of abnormal samples: {np.sum(y == 1)}")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Make predictions on test set
    print("Making predictions on test set...")
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\nTest Set Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize feature importance
    visualize_feature_importance(
        classifier, 
        save_path=os.path.join(args.output_dir, 'feature_importance.png')
    )
    
    # Visualize some predictions
    visualize_predictions(
        X_test, y_test, y_pred, 
        save_path=os.path.join(args.output_dir, 'sample_predictions.png')
    )
    
    print(f"\nTest results and visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main() 