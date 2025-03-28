import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import datetime
import os

# Configure logging
logging.basicConfig(filename='model_logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Dataset generation function
def generate_dataset(num_samples=10000, num_features=10, test=False):
    """Generates synthetic training/test datasets with different characteristics"""
    np.random.seed(42)
    
    data = {}
    for i in range(num_features):
        if test:
            # Test data: wider ranges and uniform distribution
            data[f'feature_{i}'] = np.random.uniform(-20, 20, num_samples)
            
            # Add 10% outliers
            if i % 2 == 0:
                data[f'feature_{i}'] *= np.random.choice([1, 5], num_samples, p=[0.9, 0.1])
        else:
            # Training data: normal distribution with patterns
            base = np.random.normal(i*0.5, 2, num_samples)
            noise = np.random.normal(0, 1, num_samples)
            data[f'feature_{i}'] = base + noise

    df = pd.DataFrame(data)
    
    # Create labels with non-linear relationships
    if test:
        label_condition = (
            (df['feature_0'] * df['feature_1']) > np.random.normal(15, 5, num_samples)
        ) & (
            (df['feature_2'] + df['feature_3']**2) < np.random.uniform(50, 200, num_samples)
        )
    else:
        label_condition = (
            (df['feature_0'] + df['feature_1']*2) > np.random.normal(5, 2, num_samples)
        )

    df['label'] = np.where(label_condition, 1, 0)
    
    # Add label noise
    flip_mask = np.random.rand(num_samples) < 0.05
    df['label'] = np.where(flip_mask, 1 - df['label'], df['label'])
    
    return df

# Data loading functions
def load_dataset(file_path):
    if not os.path.exists(file_path):
        return None, None

    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    logging.info(f"Loaded dataset: {file_path}")
    return X, y

def load_test_dataset(file_path, scaler):
    if not os.path.exists(file_path):
        return None, None

    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    X_scaled = scaler.transform(X)
    return X_scaled, y

# Preprocessing
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Model building
def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Evaluation functions
def conformity_score(predictions, true_labels):
    return 1 - np.abs(predictions.flatten() - true_labels)

def save_report(accuracy, precision, recall, f1, auc, sensitivity, conformity_scores):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "Sensitivity"],
        "Value": [accuracy, precision, recall, f1, auc, sensitivity]
    }).to_csv(f"model_report_{timestamp}.csv", index=False)
    
    pd.DataFrame({"Conformity Score": conformity_scores}).to_csv(f"conformity_scores_{timestamp}.csv", index=False)

# Plotting functions
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title("Model Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Model Loss")
    plt.savefig("plots/training_history.png")
    plt.close()

def plot_conformity_distribution(conformity_scores):
    plt.figure(figsize=(8, 6))
    sns.histplot(conformity_scores, bins=50, kde=True, color='blue')
    plt.title("Conformity Score Distribution")
    plt.savefig("plots/conformity_distribution.png")
    plt.close()

def plot_roc_curve(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("plots/roc_curve.png")
    plt.close()

# Training and evaluation pipeline
def train_and_evaluate(train_path, test_path):
    # Create directories if needed
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # Generate datasets if missing
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Generating datasets...")
        train_df = generate_dataset(num_samples=10000, test=False)
        test_df = generate_dataset(num_samples=2000, test=True)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Generated datasets:\n- {train_path}\n- {test_path}")

    # Load and process data
    X_train, y_train = load_dataset(train_path)
    if X_train is None:
        return None, None
    
    X_train_scaled, scaler = preprocess_data(X_train)
    
    # Build and train model
    model = build_model(X_train_scaled.shape[1])
    history = model.fit(
        X_train_scaled, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate on test set
    X_test, y_test = load_test_dataset(test_path, scaler)
    y_pred = model.predict(X_test).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    conformity_scores = conformity_score(y_pred, y_test)
    save_report(
        accuracy_score(y_test, y_pred_binary),
        precision_score(y_test, y_pred_binary),
        recall_score(y_test, y_pred_binary),
        f1_score(y_test, y_pred_binary),
        roc_auc_score(y_test, y_pred),
        recall_score(y_test, y_pred_binary),  # Sensitivity = recall
        conformity_scores
    )
    
    # Generate plots
    plot_training_history(history)
    plot_conformity_distribution(conformity_scores)
    plot_roc_curve(y_test, y_pred)
    
    return model, scaler

# Live detection interface
def live_input_detection(model, scaler):
    print("\nLive detection mode (type 'exit' to quit)")
    while True:
        try:
            input_str = input("Enter comma-separated feature values: ")
            if input_str.lower() == 'exit':
                break
            
            features = np.array([list(map(float, input_str.split(',')))])
            features_scaled = scaler.transform(features)
            
            prediction = model.predict(features_scaled).flatten()[0]
            conformity = max(prediction, 1 - prediction)
            
            result = "False Data" if prediction > 0.5 else "Legit Data"
            print(f"Result: {result} | Confidence: {prediction:.4f} | Conformity: {conformity:.4f}")
            
        except Exception as e:
            print(f"Error: {str(e)}")

# Main execution
if __name__ == "__main__":
    train_path = 'data/dataset.csv'
    test_path = 'data/test_dataset.csv'
    
    model, scaler = train_and_evaluate(train_path, test_path)
    
    if model and scaler:
        live_input_detection(model, scaler)