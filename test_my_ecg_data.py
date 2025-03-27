import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from real_ecg_classifier import RealEcgClassifier
from ecg_classifier import ECGClassifier
import argparse
import traceback

def classify_single_file(file_path, is_synthetic=False):
    """
    Classify a single ECG file
    
    Args:
        file_path: Path to the ECG data file (CSV)
        is_synthetic: Whether this is synthetic data
    
    Returns:
        Classification results
    """
    # Initialize the ECG classifier
    classifier = ECGClassifier()
    
    try:
        print(f"Attempting to classify {os.path.basename(file_path)}...")
        
        # Load data to check format
        data = pd.read_csv(file_path)
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")
        print(f"First few rows: \n{data.head(3)}")
        
        # Perform classification
        try:
            classification, confidence, reasons = classifier.classify_ecg(
                file_path=file_path, 
                is_synthetic=is_synthetic
            )
            
            # Display results
            print(f"\nResults for {os.path.basename(file_path)}:")
            print(f"Classification: {classification.upper()}")
            print(f"Confidence: {confidence:.1f}%")
            print("Detailed analysis:")
            for reason in reasons:
                print(f" - {reason}")
            
            # Plot ECG with analysis markers
            try:
                plot_file = classifier.plot_ecg_with_analysis(
                    file_path, 
                    output_dir="my_analysis_results"
                )
                print(f"ECG plot saved to: {plot_file}")
            except Exception as plot_err:
                print(f"Warning: Could not create plot: {str(plot_err)}")
            
            return classification, confidence, reasons
        except ValueError as ve:
            # Special handling for common errors
            if "too many values to unpack" in str(ve):
                print(f"Error: The classifier returned an unexpected format. This might be due to the MIT-BIH data format.")
                # Try a fallback approach
                detailed_analysis = classifier.detailed_analysis
                classification = "unknown"
                confidence = 0
                reasons = ["Could not determine classification due to data format issues"]
                if detailed_analysis and 'average_heart_rate_bpm' in detailed_analysis:
                    if detailed_analysis['average_heart_rate_bpm'] < 60:
                        classification = "abnormal"
                        confidence = 60
                        reasons = ["Bradycardia detected (low heart rate)"]
                    elif detailed_analysis['average_heart_rate_bpm'] > 100:
                        classification = "abnormal"
                        confidence = 60
                        reasons = ["Tachycardia detected (high heart rate)"]
                    else:
                        classification = "normal"
                        confidence = 50
                        reasons = ["Heart rate within normal range"]
                    
                    print(f"Fallback classification: {classification.upper()} (confidence: {confidence}%)")
                    print(f"Reason: {reasons[0]}")
                    return classification, confidence, reasons
                else:
                    print("Could not perform fallback classification")
                    return "unknown", 0, ["Classification error"]
            else:
                raise
            
    except Exception as e:
        print(f"Error classifying {file_path}: {str(e)}")
        traceback.print_exc()
        return "unknown", 0, ["Error during classification"]

def classify_directory(directory_path, is_synthetic=False):
    """
    Classify all ECG files in a directory
    
    Args:
        directory_path: Path to the directory containing ECG files
        is_synthetic: Whether the files are synthetic data
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return
    
    # Create output directory for results
    output_dir = "my_analysis_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize results dictionary
    results = {
        'file': [],
        'classification': [],
        'confidence': [],
        'reasons': []
    }
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in '{directory_path}'")
        return
    
    print(f"Found {len(csv_files)} CSV files to analyze")
    
    # Process each file
    for i, file_name in enumerate(csv_files):
        file_path = os.path.join(directory_path, file_name)
        print(f"\nProcessing file {i+1}/{len(csv_files)}: {file_name}")
        
        classification, confidence, reasons = classify_single_file(
            file_path, 
            is_synthetic=is_synthetic
        )
        
        # Store results
        results['file'].append(file_name)
        results['classification'].append(classification)
        results['confidence'].append(confidence)
        results['reasons'].append('; '.join(reasons) if reasons else '')
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'classification_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nAnalysis complete. Results saved to {results_path}")
    
    # Summary statistics
    if len(results_df) > 0:
        normal_count = sum(results_df['classification'] == 'normal')
        abnormal_count = sum(results_df['classification'] == 'abnormal')
        unknown_count = sum(results_df['classification'] == 'unknown')
        print(f"\nSummary:")
        print(f"Total files analyzed: {len(results_df)}")
        print(f"Normal ECGs: {normal_count} ({normal_count/len(results_df)*100:.1f}%)")
        print(f"Abnormal ECGs: {abnormal_count} ({abnormal_count/len(results_df)*100:.1f}%)")
        if unknown_count > 0:
            print(f"Unknown/Error: {unknown_count} ({unknown_count/len(results_df)*100:.1f}%)")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test ECG classifier on your data')
    parser.add_argument('path', type=str, help='Path to ECG file or directory containing ECG files')
    parser.add_argument('--synthetic', action='store_true', help='Flag to indicate the data is synthetic')
    parser.add_argument('--debug', action='store_true', help='Show detailed debug information')
    args = parser.parse_args()
    
    # Set debug mode if requested
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Check if path exists
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' not found.")
        return 1
    
    # Process file or directory
    if os.path.isfile(args.path):
        # Classify a single file
        classify_single_file(args.path, is_synthetic=args.synthetic)
    else:
        # Classify all files in directory
        classify_directory(args.path, is_synthetic=args.synthetic)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 