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

# Configure logging
logging.basicConfig(filename='model_logs.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset from CSV file
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # Features (all columns except last)
    y = data.iloc[:, -1].values   # Labels (last column)
    return X, y

# Preprocess dataset
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Define the neural network model
def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output: 0 (legit) or 1 (false data)
    ])
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Function to compute conformity scores
def conformity_score(predictions, true_labels):
    probabilities = predictions.flatten()
    return 1 - np.abs(probabilities - true_labels)

# Save report and conformity scores
def save_report(accuracy, precision, recall, f1, auc, sensitivity, conformity_scores):
    # Save performance metrics
    report = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "Sensitivity"],
        "Value": [accuracy, precision, recall, f1, auc, sensitivity]
    })
    report.to_csv("model_report.csv", index=False)
    logging.info("Model report saved as model_report.csv")
    
    # Save conformity scores
    conformity_df = pd.DataFrame({"Conformity Score": conformity_scores})
    conformity_df.to_csv("conformity_scores.csv", index=False)
    logging.info("Conformity scores saved as conformity_scores.csv")

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title("Model Accuracy")
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Model Loss")
    
    plt.show()

# Plot conformity score distribution
def plot_conformity_distribution(conformity_scores):
    plt.figure(figsize=(8, 6))
    sns.histplot(conformity_scores, bins=50, kde=True, color='blue')
    plt.title("Conformity Score Distribution")
    plt.xlabel("Conformity Score")
    plt.ylabel("Frequency")
    plt.show()

# Plot ROC curve
def plot_roc_curve(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Train and evaluate the model
def train_and_evaluate(file_path):
    # Load and preprocess the dataset
    X, y = load_dataset(file_path)
    X, scaler = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)
    
    # Model predictions
    y_pred = model.predict(X_test).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Conformity scores and metrics
    conformity_scores = conformity_score(y_pred, y_test)
    
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    auc_roc = roc_auc_score(y_test, y_pred)
    sensitivity = recall
    
    # Save results
    save_report(accuracy, precision, recall, f1, auc_roc, sensitivity, conformity_scores)
    
    # Plot results
    plot_training_history(history)
    plot_conformity_distribution(conformity_scores)
    plot_roc_curve(y_test, y_pred)
    
    logging.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}, Sensitivity: {sensitivity:.4f}")
    
    return model, scaler

# Live input detection with conformity scores
def live_input_detection(model, scaler):
    print("Enter feature values continuously (comma-separated). Type 'exit' to stop.")
    while True:
        user_input = input("Enter feature values: ")
        if user_input.lower() == 'exit':
            break
        try:
            input_values = np.array([list(map(float, user_input.split(',')))])
            input_values = scaler.transform(input_values)
            
            prediction = model.predict(input_values).flatten()
            conformity = 1 - np.abs(prediction - 1)  # Conformity score for binary class
            
            result = "False Data" if prediction > 0.5 else "Legit Data"
            print(f"Prediction: {result} (Confidence: {prediction[0]:.4f}, Conformity: {conformity[0]:.4f})")
        except Exception as e:
            print("Invalid input. Please enter numeric values correctly.")

# Main execution
if __name__ == "__main__":
    # Update this with your actual dataset path
    file_path = 'dataset.csv'

    # Train and evaluate the model
    model, scaler = train_and_evaluate(file_path)

    # Live input detection with conformity scores
    live_input_detection(model, scaler)

