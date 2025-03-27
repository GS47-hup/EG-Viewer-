import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import os
import argparse
from real_ecg_classifier import RealEcgClassifier

def extract_misclassified_samples(X, y_true, y_pred):
    """
    Extract misclassified samples and their indices
    
    Args:
        X: Feature data
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Misclassified samples, their indices, and type of error
    """
    # Find misclassified samples
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    # Extract samples
    misclassified_X = X[misclassified_idx]
    misclassified_y_true = y_true[misclassified_idx]
    misclassified_y_pred = y_pred[misclassified_idx]
    
    # Determine type of error (false positive or false negative)
    error_types = []
    for true, pred in zip(misclassified_y_true, misclassified_y_pred):
        if true == 0 and pred == 1:
            error_types.append("False Positive")  # Normal misclassified as Abnormal
        else:
            error_types.append("False Negative")  # Abnormal misclassified as Normal
    
    return misclassified_X, misclassified_idx, misclassified_y_true, misclassified_y_pred, error_types

def plot_misclassified_samples(misclassified_X, true_labels, pred_labels, error_types, save_path=None):
    """
    Plot a selection of misclassified samples
    
    Args:
        misclassified_X: Misclassified ECG samples
        true_labels: True labels
        pred_labels: Predicted labels
        error_types: Type of error for each sample
        save_path: Path to save the plot
    """
    # Number of samples to plot
    n_samples = min(8, len(misclassified_X))
    
    if n_samples == 0:
        print("No misclassified samples to visualize")
        return
    
    # Select samples to plot
    indices = np.random.choice(len(misclassified_X), n_samples, replace=False)
    
    # Create plot
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))
    
    # Handle single subplot case
    if n_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Plot ECG signal
        axes[i].plot(misclassified_X[idx])
        axes[i].set_title(f"Error Type: {error_types[idx]}\n"
                         f"True: {'Abnormal' if true_labels[idx] == 1 else 'Normal'}, "
                         f"Predicted: {'Abnormal' if pred_labels[idx] == 1 else 'Normal'}")
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Misclassified samples plot saved to {save_path}")
    else:
        plt.show()

def analyze_feature_distributions(classifier, X, y_true, y_pred):
    """
    Analyze feature distributions for correctly and incorrectly classified samples
    
    Args:
        classifier: Trained RealEcgClassifier
        X: ECG samples
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with analysis results
    """
    # Extract features for all samples
    features = []
    for sample in X:
        feature_dict = classifier.extract_features(sample)
        features.append(list(feature_dict.values()))
    
    features = np.array(features)
    feature_names = list(classifier.extract_features(X[0]).keys())
    
    # Group samples
    correct_idx = np.where(y_true == y_pred)[0]
    incorrect_idx = np.where(y_true != y_pred)[0]
    
    # Create feature dataframes
    correct_df = pd.DataFrame(features[correct_idx], columns=feature_names)
    correct_df['true_class'] = y_true[correct_idx]
    
    incorrect_df = pd.DataFrame(features[incorrect_idx], columns=feature_names)
    incorrect_df['true_class'] = y_true[incorrect_idx]
    incorrect_df['predicted_class'] = y_pred[incorrect_idx]
    incorrect_df['error_type'] = np.where(y_true[incorrect_idx] == 0, 'False Positive', 'False Negative')
    
    # Analyze feature statistics
    correct_stats = correct_df.groupby('true_class').mean()
    incorrect_stats = incorrect_df.groupby('error_type').mean()
    
    return {
        'correct_features': correct_df,
        'incorrect_features': incorrect_df,
        'correct_stats': correct_stats,
        'incorrect_stats': incorrect_stats,
        'feature_names': feature_names
    }

def plot_feature_distributions(analysis_results, top_n=5, save_path=None):
    """
    Plot distributions of most important features
    
    Args:
        analysis_results: Results from analyze_feature_distributions
        top_n: Number of top features to plot
        save_path: Path to save the plot
    """
    correct_df = analysis_results['correct_features']
    incorrect_df = analysis_results['incorrect_features']
    feature_names = analysis_results['feature_names']
    
    # Find features with largest differences between correct and incorrect
    correct_mean = correct_df.drop('true_class', axis=1).mean()
    incorrect_mean = incorrect_df.drop(['true_class', 'predicted_class', 'error_type'], axis=1).mean()
    
    # Calculate absolute differences
    differences = abs(correct_mean - incorrect_mean)
    top_features = differences.nlargest(top_n).index.tolist()
    
    # Create figure
    fig, axes = plt.subplots(top_n, 1, figsize=(12, 4*top_n))
    
    # Handle single subplot case
    if top_n == 1:
        axes = [axes]
    
    for i, feature in enumerate(top_features):
        # Create DataFrame copies with correct labels for legend
        correct_normal = correct_df[correct_df['true_class'] == 0].copy()
        correct_normal['group'] = 'Normal (Correct)'
        
        correct_abnormal = correct_df[correct_df['true_class'] == 1].copy()
        correct_abnormal['group'] = 'Abnormal (Correct)'
        
        fp_samples = incorrect_df[incorrect_df['error_type'] == 'False Positive'].copy()
        fp_samples['group'] = 'Normal (False Positive)'
        
        fn_samples = incorrect_df[incorrect_df['error_type'] == 'False Negative'].copy()
        fn_samples['group'] = 'Abnormal (False Negative)'
        
        # Combine data for plotting
        plot_df = pd.concat([correct_normal, correct_abnormal, fp_samples, fn_samples])
        
        # Plot distributions with custom palette
        palette = {
            'Normal (Correct)': 'green',
            'Abnormal (Correct)': 'blue',
            'Normal (False Positive)': 'orange',
            'Abnormal (False Negative)': 'red'
        }
        
        # Plot using the group column for hue
        sns.kdeplot(
            data=plot_df, x=feature, hue='group',
            palette=palette, ax=axes[i], alpha=0.5, fill=True, common_norm=False
        )
        
        axes[i].set_title(f"Feature Distribution: {feature}")
        axes[i].grid(True)
        axes[i].legend(title='Class')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Feature distribution plot saved to {save_path}")
    else:
        plt.show()

def visualize_tsne(analysis_results, save_path=None):
    """
    Visualize data in 2D using t-SNE
    
    Args:
        analysis_results: Results from analyze_feature_distributions
        save_path: Path to save the plot
    """
    correct_df = analysis_results['correct_features']
    incorrect_df = analysis_results['incorrect_features']
    feature_names = analysis_results['feature_names']
    
    # Combine data
    X_correct = correct_df.drop('true_class', axis=1).values
    y_correct = correct_df['true_class'].values
    correct_labels = np.array(['Correct'] * len(y_correct))
    
    X_incorrect = incorrect_df.drop(['true_class', 'predicted_class', 'error_type'], axis=1).values
    y_incorrect = incorrect_df['true_class'].values
    error_types = incorrect_df['error_type'].values
    
    X_combined = np.vstack([X_correct, X_incorrect])
    y_combined = np.concatenate([y_correct, y_incorrect])
    
    # Labels for visualization
    visualization_labels = np.concatenate([
        np.array([f'Normal (Correct)' if y == 0 else f'Abnormal (Correct)' for y in y_correct]),
        np.array([f'Normal ({error})' if y == 0 else f'Abnormal ({error})' 
                 for y, error in zip(y_incorrect, error_types)])
    ])
    
    # Apply t-SNE
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_combined)
    
    # Create dataframe for visualization
    tsne_df = pd.DataFrame({
        'TSNE1': X_tsne[:, 0],
        'TSNE2': X_tsne[:, 1],
        'Class': visualization_labels
    })
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Define color palette
    palette = {
        'Normal (Correct)': 'green',
        'Abnormal (Correct)': 'blue',
        'Normal (False Positive)': 'orange',
        'Abnormal (False Negative)': 'red'
    }
    
    # Create plot
    sns.scatterplot(
        data=tsne_df, x='TSNE1', y='TSNE2', hue='Class',
        palette=palette, s=100, alpha=0.7
    )
    
    plt.title('t-SNE Visualization of ECG Features')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"t-SNE visualization saved to {save_path}")
    else:
        plt.show()

def suggest_improvements(analysis_results):
    """
    Suggest improvements based on misclassification analysis
    
    Args:
        analysis_results: Results from analyze_feature_distributions
        
    Returns:
        List of improvement suggestions
    """
    correct_stats = analysis_results['correct_stats']
    incorrect_stats = analysis_results['incorrect_stats']
    
    # Compare feature statistics between correct and incorrect classifications
    suggestions = []
    
    # Check for False Positives (Normal classified as Abnormal)
    if 'False Positive' in incorrect_stats.index:
        fp_stats = incorrect_stats.loc['False Positive']
        normal_stats = correct_stats.loc[0] if 0 in correct_stats.index else None
        
        if normal_stats is not None:
            # Find features with large differences
            differences = abs(fp_stats - normal_stats) / (normal_stats.abs() + 1e-10)
            top_diff_features = differences.nlargest(3).index.tolist()
            
            suggestions.append("For False Positives (Normal classified as Abnormal):")
            for feature in top_diff_features:
                suggestions.append(f"  - Adjust thresholds for '{feature}' feature")
    
    # Check for False Negatives (Abnormal classified as Normal)
    if 'False Negative' in incorrect_stats.index:
        fn_stats = incorrect_stats.loc['False Negative']
        abnormal_stats = correct_stats.loc[1] if 1 in correct_stats.index else None
        
        if abnormal_stats is not None:
            # Find features with large differences
            differences = abs(fn_stats - abnormal_stats) / (abnormal_stats.abs() + 1e-10)
            top_diff_features = differences.nlargest(3).index.tolist()
            
            suggestions.append("For False Negatives (Abnormal classified as Normal):")
            for feature in top_diff_features:
                suggestions.append(f"  - Adjust thresholds for '{feature}' feature")
    
    # General suggestions
    suggestions.append("\nGeneral Improvement Suggestions:")
    suggestions.append("  - Collect more data for underrepresented classes")
    suggestions.append("  - Try additional feature engineering focused on the problematic features")
    suggestions.append("  - Consider ensemble methods combining multiple classifiers")
    suggestions.append("  - Experiment with class weights to balance false positives and false negatives")
    
    return suggestions

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze misclassified ECG samples')
    parser.add_argument('--data', type=str, default='Real ECG.csv',
                        help='Path to the CSV file with ECG data')
    parser.add_argument('--model', type=str, default='real_ecg_data/best_model.joblib',
                        help='Path to the trained model file')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--output_dir', type=str, default='real_ecg_data/analysis',
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
    
    # Make predictions on test set
    print("Making predictions on test set...")
    y_pred = classifier.predict(X_test)
    
    # Extract misclassified samples
    misclassified_X, misclassified_idx, misclassified_y_true, misclassified_y_pred, error_types = extract_misclassified_samples(X_test, y_test, y_pred)
    
    print(f"\nFound {len(misclassified_idx)} misclassified samples out of {len(X_test)} test samples")
    print(f"False Positives: {error_types.count('False Positive')}")
    print(f"False Negatives: {error_types.count('False Negative')}")
    
    # Plot misclassified samples
    plot_misclassified_samples(
        misclassified_X, misclassified_y_true, misclassified_y_pred, error_types,
        save_path=os.path.join(args.output_dir, 'misclassified_samples.png')
    )
    
    # Analyze feature distributions
    print("\nAnalyzing feature distributions...")
    analysis_results = analyze_feature_distributions(classifier, X_test, y_test, y_pred)
    
    # Plot feature distributions
    plot_feature_distributions(
        analysis_results, top_n=5,
        save_path=os.path.join(args.output_dir, 'feature_distributions.png')
    )
    
    # Visualize data using t-SNE
    visualize_tsne(
        analysis_results,
        save_path=os.path.join(args.output_dir, 'tsne_visualization.png')
    )
    
    # Generate improvement suggestions
    suggestions = suggest_improvements(analysis_results)
    
    # Save suggestions to file
    with open(os.path.join(args.output_dir, 'improvement_suggestions.txt'), 'w') as f:
        f.write("\n".join(suggestions))
    
    # Print suggestions
    print("\nImprovement Suggestions:")
    for suggestion in suggestions:
        print(suggestion)
    
    print(f"\nAnalysis results and visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main() 