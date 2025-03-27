import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import time
import os
from scipy import signal
from real_ecg_classifier import RealEcgClassifier

def extract_features_batch(X, feature_extractor):
    """
    Extract features from a batch of ECG samples
    
    Args:
        X: Array of ECG samples
        feature_extractor: Feature extraction function
        
    Returns:
        Feature matrix
    """
    print(f"Extracting features from {len(X)} samples...")
    features = []
    
    for i, ecg_sample in enumerate(X):
        if i % 500 == 0 and i > 0:
            print(f"Processed {i} samples")
        
        # Extract features for this sample
        sample_features = feature_extractor(ecg_sample)
        features.append(list(sample_features.values()))
    
    return np.array(features)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train and evaluate a model
    
    Args:
        model: ML model to evaluate
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Extract specific metrics
    sensitivity = report['1.0']['recall'] if '1.0' in report else report['1']['recall']
    specificity = report['0.0']['recall'] if '0.0' in report else report['0']['recall']
    
    # For models that support predict_proba, calculate ROC AUC
    roc_auc = None
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
    except:
        pass
    
    # Print results
    print(f"\n--- {model_name} Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    if roc_auc:
        print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Prediction time: {prediction_time:.2f} seconds")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Return metrics
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred
    }

def plot_comparison(results, save_path=None):
    """
    Plot comparison of model performance
    
    Args:
        results: List of evaluation result dictionaries
        save_path: Path to save the plot
    """
    # Extract metrics for plotting
    model_names = [r['model_name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    sensitivities = [r['sensitivity'] for r in results]
    specificities = [r['specificity'] for r in results]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of bars on X axis
    r1 = np.arange(len(model_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    ax.bar(r1, accuracies, width=barWidth, label='Accuracy', color='skyblue')
    ax.bar(r2, sensitivities, width=barWidth, label='Sensitivity', color='lightgreen')
    ax.bar(r3, specificities, width=barWidth, label='Specificity', color='salmon')
    
    # Add xticks on the middle of the group bars
    ax.set_xlabel('Model')
    ax.set_xticks([r + barWidth for r in range(len(model_names))])
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Create legend & title
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    
    # Show plot
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()

def plot_roc_curves(results, X_test, y_test, save_path=None):
    """
    Plot ROC curves for different models
    
    Args:
        results: List of evaluation result dictionaries
        X_test: Test features
        y_test: Test labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for result in results:
        if 'model' in result and hasattr(result['model'], 'predict_proba'):
            # Get probabilities
            y_proba = result['model'].predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, label=f'{result["model_name"]} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curves saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Create output directory
    os.makedirs('real_ecg_data', exist_ok=True)
    
    # Load the dataset
    print("Loading ECG dataset...")
    ecg_data = pd.read_csv('Real ECG.csv', header=None)
    
    # Separate features (ECG readings) and labels
    X = ecg_data.iloc[:, :-1].values  # All columns except the last one
    y = ecg_data.iloc[:, -1].values   # Last column contains labels
    
    # Print dataset info
    print(f"Dataset shape: {ecg_data.shape}")
    print(f"Number of normal samples: {np.sum(y == 0)}")
    print(f"Number of abnormal samples: {np.sum(y == 1)}")
    
    # Split data into training and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create feature extractor
    feature_extractor = RealEcgClassifier().extract_features
    
    # Extract features
    X_train_features = extract_features_batch(X_train, feature_extractor)
    X_test_features = extract_features_batch(X_test, feature_extractor)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    # Define models to evaluate
    models = [
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        },
        {
            'name': 'Gradient Boosting',
            'model': GradientBoostingClassifier(n_estimators=100, random_state=42)
        },
        {
            'name': 'SVM',
            'model': SVC(probability=True, random_state=42)
        },
        {
            'name': 'Neural Network',
            'model': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        },
        {
            'name': 'K-Nearest Neighbors',
            'model': KNeighborsClassifier(n_neighbors=5)
        }
    ]
    
    # Evaluate each model
    results = []
    for model_info in models:
        print(f"\nEvaluating {model_info['name']}...")
        model = model_info['model']
        result = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, model_info['name'])
        result['model'] = model  # Store model for later use
        results.append(result)
    
    # Find best performing model
    best_model = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest model: {best_model['model_name']} with accuracy {best_model['accuracy']:.4f}")
    
    # Plot comparison
    plot_comparison(results, save_path=os.path.join('real_ecg_data', 'model_comparison.png'))
    
    # Plot ROC curves
    plot_roc_curves(results, X_test_scaled, y_test, save_path=os.path.join('real_ecg_data', 'roc_curves.png'))
    
    # Tune best model if it's Random Forest
    if best_model['model_name'] == 'Random Forest':
        print("\nPerforming hyperparameter tuning for Random Forest...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create grid search
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_train_scaled, y_train)
        
        # Print best parameters
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
        
        # Evaluate best model
        best_rf = grid_search.best_estimator_
        tuned_result = evaluate_model(best_rf, X_train_scaled, X_test_scaled, y_train, y_test, 'Tuned Random Forest')
        
        # Save best model
        from real_ecg_classifier import RealEcgClassifier
        
        # Create a properly configured classifier
        best_classifier = RealEcgClassifier()
        best_classifier.model = best_rf
        best_classifier.scaler = scaler
        best_classifier.features = list(feature_extractor(X[0]).keys())
        
        # Save model
        best_classifier.save_model(os.path.join('real_ecg_data', 'best_model.joblib'))
    
    print("\nModel evaluation complete!") 