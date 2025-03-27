import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from scipy import signal
import pywt  # PyWavelets for wavelet transform
import warnings
warnings.filterwarnings('ignore')

class RealEcgClassifier:
    """
    Classifier for real ECG data with 140 data points per sample
    """
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = []
        
    def extract_features(self, ecg_data):
        """
        Extract meaningful features from raw ECG signals
        
        Args:
            ecg_data: Raw ECG signal with 140 points
            
        Returns:
            Dictionary of extracted features
        """
        # Add time dimension (assuming consistent sampling rate)
        time = np.linspace(0, 1, len(ecg_data))
        
        # Filter the signal to remove noise
        filtered_signal = self.filter_ecg(ecg_data)
        
        # Find peaks (R peaks)
        peaks, _ = signal.find_peaks(filtered_signal, distance=10, prominence=0.1*np.max(filtered_signal))
        
        # Calculate heart rate if peaks are detected
        if len(peaks) > 1:
            # Assuming 1 second of data, convert to BPM
            heart_rate = len(peaks) * 60
        else:
            heart_rate = 0
            
        # Calculate statistical features
        mean_val = np.mean(ecg_data)
        std_val = np.std(ecg_data)
        max_val = np.max(ecg_data)
        min_val = np.min(ecg_data)
        range_val = max_val - min_val
        
        # Calculate frequency domain features using FFT
        fft_vals = np.abs(np.fft.rfft(filtered_signal))
        fft_freq = np.fft.rfftfreq(len(filtered_signal))
        
        # Get dominant frequency
        dom_freq_idx = np.argmax(fft_vals)
        dom_freq = fft_freq[dom_freq_idx]
        
        # Energy in different frequency bands
        low_freq_energy = np.sum(fft_vals[(fft_freq >= 0) & (fft_freq < 5)])
        med_freq_energy = np.sum(fft_vals[(fft_freq >= 5) & (fft_freq < 15)])
        high_freq_energy = np.sum(fft_vals[(fft_freq >= 15)])
        
        # Wavelet features using PyWavelets
        wavelet_energies = []
        try:
            # Use DB4 wavelet for decomposition
            coeffs = pywt.wavedec(filtered_signal, 'db4', level=4)
            
            # Calculate energy for each level
            for i, coeff in enumerate(coeffs):
                wavelet_energies.append(np.sum(coeff**2))
                
            # If we have fewer than 5 wavelet levels, pad with zeros
            while len(wavelet_energies) < 5:
                wavelet_energies.append(0)
        except:
            # In case of error, use zeros
            wavelet_energies = [0, 0, 0, 0, 0]
        
        # Calculate RR intervals if multiple peaks found
        if len(peaks) > 1:
            rr_intervals = np.diff(time[peaks]) * 1000  # in ms
            rr_mean = np.mean(rr_intervals)
            rr_std = np.std(rr_intervals)
            rr_max = np.max(rr_intervals)
            rr_min = np.min(rr_intervals)
            rr_range = rr_max - rr_min
        else:
            rr_mean = rr_std = rr_max = rr_min = rr_range = 0
            
        # Calculate ST segment features if possible
        if len(peaks) > 0:
            # Try to find ST segments (typically 80-120ms after R peak)
            st_segments = []
            for peak in peaks:
                if peak + 30 < len(filtered_signal):
                    st_start = peak + 10
                    st_end = peak + 30
                    st_segments.append(np.mean(filtered_signal[st_start:st_end]))
                    
            if st_segments:
                st_elevation = np.mean(st_segments)
            else:
                st_elevation = 0
        else:
            st_elevation = 0
            
        # Return all features as a dictionary
        features = {
            'heart_rate': heart_rate,
            'mean': mean_val,
            'std': std_val,
            'max': max_val,
            'min': min_val,
            'range': range_val,
            'dom_freq': dom_freq,
            'low_freq_energy': low_freq_energy,
            'med_freq_energy': med_freq_energy,
            'high_freq_energy': high_freq_energy,
            'wavelet_energy_1': wavelet_energies[0],
            'wavelet_energy_2': wavelet_energies[1],
            'wavelet_energy_3': wavelet_energies[2],
            'wavelet_energy_4': wavelet_energies[3],
            'wavelet_energy_5': wavelet_energies[4],
            'rr_mean': rr_mean,
            'rr_std': rr_std,
            'rr_range': rr_range,
            'num_peaks': len(peaks),
            'st_elevation': st_elevation
        }
        
        # Store feature names
        self.features = list(features.keys())
        
        return features
    
    def filter_ecg(self, ecg_signal, sampling_rate=140):
        """Filter the ECG signal to remove noise"""
        # Calculate frequency parameters
        nyquist_freq = 0.5 * sampling_rate
        low = 0.5 / nyquist_freq  # 0.5 Hz - removes baseline wander
        high = 40.0 / nyquist_freq  # 40 Hz - removes high-frequency noise
        
        # Apply bandpass filter
        try:
            b, a = signal.butter(2, [low, high], btype='band')
            ecg_filtered = signal.filtfilt(b, a, ecg_signal)
        except Exception as e:
            print(f"Filtering error: {e}")
            ecg_filtered = ecg_signal
        
        return ecg_filtered
    
    def prepare_features(self, X):
        """
        Extract features from multiple ECG samples
        
        Args:
            X: Array of ECG samples, each with 140 data points
            
        Returns:
            Feature matrix for model training/prediction
        """
        feature_matrix = []
        
        for i, ecg_sample in enumerate(X):
            if i % 500 == 0 and i > 0:
                print(f"Processed {i} samples")
                
            features = self.extract_features(ecg_sample)
            feature_matrix.append(list(features.values()))
            
        return np.array(feature_matrix)
    
    def fit(self, X_train, y_train):
        """
        Train the classifier on ECG data
        
        Args:
            X_train: Training ECG samples (each with 140 data points)
            y_train: Training labels (0 or 1)
        """
        print(f"Extracting features from {len(X_train)} training samples...")
        X_features = self.prepare_features(X_train)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Initialize and train model
        print("Training Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y_train)
        
        # Calculate feature importance
        feature_importances = self.model.feature_importances_
        importance_dict = dict(zip(self.features, feature_importances))
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\nTop 10 important features:")
        for feature, importance in top_features:
            print(f"{feature}: {importance:.4f}")
    
    def predict(self, X_test):
        """
        Predict ECG classifications
        
        Args:
            X_test: Test ECG samples (each with 140 data points)
            
        Returns:
            Predicted labels (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Extract features and scale
        X_features = self.prepare_features(X_test)
        X_scaled = self.scaler.transform(X_features)
        
        # Make predictions
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X_test):
        """
        Predict ECG classification probabilities
        
        Args:
            X_test: Test ECG samples (each with 140 data points)
            
        Returns:
            Predicted probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Extract features and scale
        X_features = self.prepare_features(X_test)
        X_scaled = self.scaler.transform(X_features)
        
        # Make predictions
        return self.model.predict_proba(X_scaled)
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        if self.model is None:
            raise ValueError("No trained model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and scaler
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        # Load model and scaler
        saved_data = joblib.load(filepath)
        
        # Create new instance and restore saved data
        classifier = cls()
        classifier.model = saved_data['model']
        classifier.scaler = saved_data['scaler']
        classifier.features = saved_data['features']
        
        return classifier


def evaluate_classifier(classifier, X_test, y_test):
    """
    Evaluate classifier performance
    
    Args:
        classifier: Trained RealEcgClassifier
        X_test: Test ECG samples
        y_test: True labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Extract specific metrics
    sensitivity = report['1.0']['recall'] if '1.0' in report else report['1']['recall']
    specificity = report['0.0']['recall'] if '0.0' in report else report['0']['recall']
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': conf_matrix,
        'classification_report': report
    }


def visualize_results(X_test, y_test, y_pred, save_path=None):
    """
    Visualize classification results
    
    Args:
        X_test: Test ECG samples
        y_test: True labels
        y_pred: Predicted labels
        save_path: Path to save visualization
    """
    # Find misclassified samples
    misclassified = np.where(y_test != y_pred)[0]
    
    # Select a few misclassified samples
    num_samples = min(4, len(misclassified))
    if num_samples == 0:
        print("No misclassified samples to visualize")
        return
    
    sample_indices = misclassified[:num_samples]
    
    # Create plot
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(sample_indices):
        # Plot ECG signal
        axes[i].plot(X_test[idx])
        axes[i].set_title(f"True: {'Abnormal' if y_test[idx] == 1 else 'Normal'}, "
                         f"Predicted: {'Abnormal' if y_pred[idx] == 1 else 'Normal'}")
        axes[i].set_ylabel('Amplitude')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Load the dataset
    print("Loading ECG dataset...")
    ecg_data = pd.read_csv('Real ECG.csv', header=None)
    
    # Separate features (ECG readings) and labels
    X = ecg_data.iloc[:, :-1].values  # All columns except the last one
    y = ecg_data.iloc[:, -1].values   # Last column contains labels
    
    # Split data into training, validation, and test sets (60/20/20 split)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Dataset shape: {ecg_data.shape}")
    print(f"Number of normal samples: {np.sum(y == 0)}")
    print(f"Number of abnormal samples: {np.sum(y == 1)}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train classifier
    classifier = RealEcgClassifier()
    classifier.fit(X_train, y_train)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set:")
    val_metrics = evaluate_classifier(classifier, X_val, y_val)
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set:")
    test_metrics = evaluate_classifier(classifier, X_test, y_test)
    
    # Save trained model
    model_path = os.path.join('real_ecg_data', 'real_ecg_classifier.joblib')
    classifier.save_model(model_path)
    
    # Visualize some results
    y_pred_test = classifier.predict(X_test)
    visualization_path = os.path.join('real_ecg_data', 'misclassified_samples.png')
    visualize_results(X_test, y_test, y_pred_test, save_path=visualization_path)
    
    print("\nTraining and evaluation complete!") 