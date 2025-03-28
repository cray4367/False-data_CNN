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

# Load dataset from CSV file
def load_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        logging.error(f"File {file_path} not found.")
        return None, None

    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # Features (all columns except last)
    y = data.iloc[:, -1].values   # Labels (last column)

    logging.info(f"Loaded dataset: {file_path}")
    logging.info(f"Input features shape: {X.shape}")
    logging.info(f"Labels shape: {y.shape}")

    return X, y

# Preprocess dataset
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info(f"Preprocessed input (after scaling). Shape: {X_scaled.shape}")
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

# Compute conformity scores
def conformity_score(predictions, true_labels):
    probabilities = predictions.flatten()
    return 1 - np.abs(probabilities - true_labels)

# Save report and conformity scores
def save_report(accuracy, precision, recall, f1, auc, sensitivity, conformity_scores):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    report_filename = f"model_report_{timestamp}.csv"
    conformity_filename = f"conformity_scores_{timestamp}.csv"

    report = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "Sensitivity"],
        "Value": [accuracy, precision, recall, f1, auc, sensitivity]
    })
    report.to_csv(report_filename, index=False)
    logging.info(f"Model report saved as {report_filename}")

    conformity_df = pd.DataFrame({"Conformity Score": conformity_scores})
    conformity_df.to_csv(conformity_filename, index=False)
    logging.info(f"Conformity scores saved as {conformity_filename}")

# Plot training history and save it as an image
def plot_training_history(history, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

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

    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()
    logging.info("Training history plot saved as 'training_history.png'")

# Plot conformity score distribution and save it as an image
def plot_conformity_distribution(conformity_scores, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.histplot(conformity_scores, bins=50, kde=True, color='blue')
    plt.title("Conformity Score Distribution")
    plt.xlabel("Conformity Score")
    plt.ylabel("Frequency")

    plt.savefig(os.path.join(output_dir, "conformity_distribution.png"))
    plt.close()
    logging.info("Conformity distribution plot saved as 'conformity_distribution.png'")

# Plot ROC curve and save it as an image
def plot_roc_curve(y_test, y_pred, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()
    logging.info("ROC curve plot saved as 'roc_curve.png'")

# Train and evaluate the model
def train_and_evaluate(file_path):
    X, y = load_dataset(file_path)

    if X is None or y is None:
        print("No valid data provided. Exiting...")
        return None, None

    X, scaler = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

    y_pred = model.predict(X_test).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int)

    conformity_scores = conformity_score(y_pred, y_test)

    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    auc_roc = roc_auc_score(y_test, y_pred)
    sensitivity = recall

    save_report(accuracy, precision, recall, f1, auc_roc, sensitivity, conformity_scores)

    # Save plots as images
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
            logging.info(f"Live input values: {input_values}")

            input_values = scaler.transform(input_values)

            prediction = model.predict(input_values).flatten()
            conformity = np.maximum(prediction, 1 - prediction)

            result = "False Data" if prediction > 0.5 else "Legit Data"
            print(f"Prediction: {result} (Confidence: {prediction[0]:.4f}, Conformity: {conformity[0]:.4f})")

            logging.info(f"Prediction: {result}, Confidence: {prediction[0]:.4f}, Conformity: {conformity[0]:.4f}")

            # Ask if the user wants to continue
            cont = input("Do you want to check more values? (yes/no): ").strip().lower()
            if cont not in ["yes", "y"]:
                break


        except Exception as e:
            print("Invalid input. Please enter numeric values correctly.")
            logging.error(f"Invalid input error: {e}")

# Main execution
if __name__ == "__main__":
    file_path = 'dataset.csv'

    model, scaler = train_and_evaluate(file_path)

    if model and scaler:
        live_input_detection(model, scaler)
