import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from ecg_classifier import ECGClassifier

def evaluate_on_dataset(data_dir, save_results=True, plot_results=True):
    """
    Evaluate the ECG classifier on a dataset
    
    Parameters:
    - data_dir: Directory containing normal/ and abnormal/ subdirectories with ECG samples
    - save_results: Whether to save results to CSV
    - plot_results: Whether to plot results
    
    Returns:
    - results_df: DataFrame with classification results
    """
    # Get the list of data files
    normal_dir = os.path.join(data_dir, 'normal')
    abnormal_dir = os.path.join(data_dir, 'abnormal')

    # Get the list of normal and abnormal files
    normal_files = [f for f in os.listdir(normal_dir) if f.endswith('.csv')]
    abnormal_files = [f for f in os.listdir(abnormal_dir) if f.endswith('.csv')]

    if len(normal_files) == 0 or len(abnormal_files) == 0:
        print(f"ERROR: Directory structure should contain 'normal' and 'abnormal' subdirectories")
        return None
    
    print(f"Found {len(normal_files)} normal and {len(abnormal_files)} abnormal samples")
    
    classifier = ECGClassifier()
    results = []
    
    # Process normal files
    for file in normal_files:
        print(f"Processing {file}...")
        file_path = os.path.join(normal_dir, file)
        
        # Check if this is synthetic data based on filename
        is_synthetic = "normal_hr" in file
        
        # Classify ECG using the classifier
        result = classifier.classify_ecg(file_path, is_synthetic=is_synthetic)
        predicted_class = result['prediction']
        confidence = result['confidence']
        reasons = result['reasons']

        # Additional analysis fields from the classifier's detailed_analysis
        detailed_analysis = classifier.detailed_analysis.copy()
        detailed_analysis.pop('file_path', None)  # Remove file_path from detailed analysis
        
        # Determine if the classification is correct
        correct = predicted_class == "normal"
        
        # Add to results
        results.append({
            "file": file,
            "true_class": "normal",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "reasons": reasons[0] if reasons else "",
            "correct": correct,
            **detailed_analysis
        })
    
    # Process abnormal files
    for file in abnormal_files:
        print(f"Processing {file}...")
        file_path = os.path.join(abnormal_dir, file)
        
        # Check if this is synthetic data based on filename
        is_synthetic = "abnormal_" in file
        
        # Classify ECG using the classifier
        result = classifier.classify_ecg(file_path, is_synthetic=is_synthetic)
        predicted_class = result['prediction']
        confidence = result['confidence']
        reasons = result['reasons']
        
        # Additional analysis fields from the classifier's detailed_analysis
        detailed_analysis = classifier.detailed_analysis.copy()
        detailed_analysis.pop('file_path', None)  # Remove file_path from detailed analysis
        
        # Determine if the classification is correct
        correct = predicted_class == "abnormal"
        
        # Add to results
        results.append({
            "file": file,
            "true_class": "abnormal",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "reasons": reasons[0] if reasons else "",
            "correct": correct,
            **detailed_analysis
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate overall performance metrics
    total_samples = len(results_df)
    correct_samples = results_df['correct'].sum()
    accuracy = correct_samples / total_samples * 100
    
    # Calculate class-specific metrics
    true_normal = results_df['true_class'] == 'normal'
    pred_normal = results_df['predicted_class'].str.lower() == 'normal'
    
    true_positive = ((results_df['true_class'] == 'abnormal') & (results_df['predicted_class'].str.lower() == 'abnormal')).sum()
    false_positive = ((results_df['true_class'] == 'normal') & (results_df['predicted_class'].str.lower() == 'abnormal')).sum()
    true_negative = ((results_df['true_class'] == 'normal') & (results_df['predicted_class'].str.lower() == 'normal')).sum()
    false_negative = ((results_df['true_class'] == 'abnormal') & (results_df['predicted_class'].str.lower() == 'normal')).sum()
    
    sensitivity = true_positive / (true_positive + false_negative) * 100 if (true_positive + false_negative) > 0 else 0
    specificity = true_negative / (true_negative + false_positive) * 100 if (true_negative + false_positive) > 0 else 0
    precision = true_positive / (true_positive + false_positive) * 100 if (true_positive + false_positive) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print("\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.1f}% ({correct_samples}/{total_samples})")
    print(f"Sensitivity: {sensitivity:.1f}%")
    print(f"Specificity: {specificity:.1f}%")
    print(f"Precision: {precision:.1f}%")
    print(f"F1 Score: {f1_score:.1f}%")
    
    # Display confusion matrix
    conf_matrix = confusion_matrix(
        results_df['true_class'].apply(lambda x: 1 if x == 'abnormal' else 0),
        results_df['predicted_class'].apply(lambda x: 1 if x.lower() == 'abnormal' else 0)
    )
    
    print("\nConfusion Matrix:")
    print("                    Predicted Normal    Predicted Abnormal")
    print(f"Actual Normal          {conf_matrix[0,0]}                {conf_matrix[0,1]}")
    print(f"Actual Abnormal        {conf_matrix[1,0]}                {conf_matrix[1,1]}")
    
    # Plot results if requested
    if plot_results:
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(conf_matrix, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.xticks([0, 1], ['Normal', 'Abnormal'])
        plt.yticks([0, 1], ['Normal', 'Abnormal'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(conf_matrix[i, j]), 
                        ha="center", va="center", color="black", fontsize=15)
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(
            results_df['true_class'].apply(lambda x: 1 if x == 'abnormal' else 0),
            results_df['confidence'] / 100  # Normalize confidence to 0-1
        )
        roc_auc = auc(fpr, tpr)
        
        plt.subplot(2, 2, 2)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Check if we have heart rate and ST elevation data for histograms
        has_hr_data = 'average_heart_rate_bpm' in results_df.columns
        has_st_data = 'st_segment_elevation_mv' in results_df.columns
        
        # Plot heart rate distribution by class
        if has_hr_data:
            plt.subplot(2, 2, 3)
            normal_hr = results_df[results_df['true_class'] == 'normal']['average_heart_rate_bpm']
            abnormal_hr = results_df[results_df['true_class'] == 'abnormal']['average_heart_rate_bpm']
            
            # Handle NaN values
            normal_hr = normal_hr.dropna()
            abnormal_hr = abnormal_hr.dropna()
            
            # Only plot histograms if we have valid data
            if len(normal_hr) > 0 and len(abnormal_hr) > 0:
                plt.hist(normal_hr, alpha=0.5, label='Normal', bins=20)
                plt.hist(abnormal_hr, alpha=0.5, label='Abnormal', bins=20)
                plt.xlabel('Heart Rate (BPM)')
                plt.ylabel('Count')
                plt.title('Heart Rate Distribution by Class')
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'Heart Rate Data Insufficient', 
                        ha='center', va='center', fontsize=12)
                plt.axis('off')
        else:
            plt.subplot(2, 2, 3)
            plt.text(0.5, 0.5, 'Heart Rate Data Not Available', 
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        # Plot ST elevation distribution by class
        if has_st_data:
            plt.subplot(2, 2, 4)
            normal_st = results_df[results_df['true_class'] == 'normal']['st_segment_elevation_mv']
            abnormal_st = results_df[results_df['true_class'] == 'abnormal']['st_segment_elevation_mv']
            
            # Handle NaN values
            normal_st = normal_st.dropna()
            abnormal_st = abnormal_st.dropna()
            
            # Only plot histograms if we have valid data
            if len(normal_st) > 0 and len(abnormal_st) > 0:
                plt.hist(normal_st, alpha=0.5, label='Normal', bins=20)
                plt.hist(abnormal_st, alpha=0.5, label='Abnormal', bins=20)
                plt.xlabel('ST Elevation (mV)')
                plt.ylabel('Count')
                plt.title('ST Elevation Distribution by Class')
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'ST Elevation Data Insufficient', 
                        ha='center', va='center', fontsize=12)
                plt.axis('off')
        else:
            plt.subplot(2, 2, 4)
            plt.text(0.5, 0.5, 'ST Elevation Data Not Available', 
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('classifier_performance.png', dpi=300)
        plt.close()
        
        print(f"Performance plots saved to classifier_performance.png")
    
    # Save results if requested
    if save_results:
        results_df.to_csv('classification_results.csv', index=False)
        print(f"Detailed results saved to classification_results.csv")
    
    return results_df

def generate_html_report(results_df, output_file='large_dataset_report.html'):
    """
    Generate an HTML report from classification results
    
    Parameters:
    - results_df: DataFrame with classification results
    - output_file: Path to save the HTML report
    """
    # Calculate performance metrics
    total_samples = len(results_df)
    correct_samples = results_df['correct'].sum()
    accuracy = correct_samples / total_samples * 100
    
    true_positive = ((results_df['true_class'] == 'abnormal') & (results_df['predicted_class'].str.lower() == 'abnormal')).sum()
    false_positive = ((results_df['true_class'] == 'normal') & (results_df['predicted_class'].str.lower() == 'abnormal')).sum()
    true_negative = ((results_df['true_class'] == 'normal') & (results_df['predicted_class'].str.lower() == 'normal')).sum()
    false_negative = ((results_df['true_class'] == 'abnormal') & (results_df['predicted_class'].str.lower() == 'normal')).sum()
    
    sensitivity = true_positive / (true_positive + false_negative) * 100 if (true_positive + false_negative) > 0 else 0
    specificity = true_negative / (true_negative + false_positive) * 100 if (true_negative + false_positive) > 0 else 0
    precision = true_positive / (true_positive + false_positive) * 100 if (true_positive + false_positive) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # Check for available columns
    has_hr_data = 'average_heart_rate_bpm' in results_df.columns
    has_st_data = 'st_segment_elevation_mv' in results_df.columns
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ECG Classification Report - Large Dataset</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metrics {{ display: flex; flex-wrap: wrap; }}
            .metric-card {{ background-color: #f0f0f0; border-radius: 5px; margin: 10px; padding: 15px; width: 200px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
            .correct {{ color: green; }}
            .incorrect {{ color: red; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>ECG Classification Report - Large Dataset</h1>
        
        <h2>Performance Metrics</h2>
        <div class="metrics">
            <div class="metric-card">
                <div>Accuracy</div>
                <div class="metric-value">{accuracy:.1f}%</div>
                <div>({correct_samples}/{total_samples})</div>
            </div>
            <div class="metric-card">
                <div>Sensitivity</div>
                <div class="metric-value">{sensitivity:.1f}%</div>
                <div>(True Positive Rate)</div>
            </div>
            <div class="metric-card">
                <div>Specificity</div>
                <div class="metric-value">{specificity:.1f}%</div>
                <div>(True Negative Rate)</div>
            </div>
            <div class="metric-card">
                <div>Precision</div>
                <div class="metric-value">{precision:.1f}%</div>
                <div>(Positive Predictive Value)</div>
            </div>
            <div class="metric-card">
                <div>F1 Score</div>
                <div class="metric-value">{f1_score:.1f}%</div>
                <div>(Harmonic Mean)</div>
            </div>
        </div>
        
        <h2>Confusion Matrix</h2>
        <table>
            <tr>
                <th></th>
                <th>Predicted Normal</th>
                <th>Predicted Abnormal</th>
            </tr>
            <tr>
                <th>Actual Normal</th>
                <td>{true_negative}</td>
                <td>{false_positive}</td>
            </tr>
            <tr>
                <th>Actual Abnormal</th>
                <td>{false_negative}</td>
                <td>{true_positive}</td>
            </tr>
        </table>
        
        <h2>Performance Visualization</h2>
        <img src="classifier_performance.png" alt="Performance Visualization">
        
        <h2>Classification Results</h2>
        <table>
            <tr>
                <th>File</th>
                <th>True Class</th>
                <th>Predicted Class</th>
                <th>Confidence</th>
                {('<th>Heart Rate (BPM)</th>' if has_hr_data else '')}
                {('<th>ST Elevation (mV)</th>' if has_st_data else '')}
                <th>Reasons</th>
            </tr>
    """
    
    # Add rows for each sample
    for _, row in results_df.iterrows():
        correct_class = "correct" if row['correct'] else "incorrect"
        html_content += f"""
            <tr class="{correct_class}">
                <td>{row['file']}</td>
                <td>{row['true_class'].upper()}</td>
                <td>{row['predicted_class']}</td>
                <td>{row['confidence']:.1f}%</td>
                {(f'<td>{row["average_heart_rate_bpm"]:.1f}</td>' if has_hr_data else '')}
                {(f'<td>{row["st_segment_elevation_mv"]:.3f}</td>' if has_st_data else '')}
                <td>{row['reasons']}</td>
            </tr>
        """
    
    # Close HTML
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {output_file}")

def main():
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = 'large_test_dataset'  # Default location
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        print(f"Please run generate_large_dataset.py first or specify a valid directory.")
        sys.exit(1)
    
    results = evaluate_on_dataset(data_dir)
    if results is not None:
        generate_html_report(results)

if __name__ == "__main__":
    main() 