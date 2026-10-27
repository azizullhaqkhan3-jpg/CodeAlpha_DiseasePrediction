
from pathlib import Path

import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# Project directories
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

for folder in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# 1. Load the Breast Cancer dataset
print("Loading Breast Cancer dataset...")

dataset = load_breast_cancer(as_frame=True)

X = dataset.data.copy()

# Original sklearn labels:
# 0 = malignant, 1 = benign
# Convert to readable class names.
y = dataset.target.map({
    0: "Malignant",
    1: "Benign",
})

# Save dataset locally
df = X.copy()
df["Diagnosis"] = y
df.to_csv(DATA_DIR / "breast_cancer.csv", index=False)

print("Dataset shape:", df.shape)
print("Number of features:", X.shape[1])
print("\nClass distribution:")
print(y.value_counts())


# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 3. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)

# 4. Train the model
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)


# 5. Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Find the probability column for Malignant
malignant_index = list(model.classes_).index("Malignant")
y_test_malignant = (y_test == "Malignant").astype(int)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

roc_auc = roc_auc_score(
    y_test_malignant,
    y_proba[:, malignant_index],
)

report = classification_report(
    y_test,
    y_pred,
    labels=["Malignant", "Benign"],
    zero_division=0,
)

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=["Malignant", "Benign"],
)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")

print("\nClassification Report:")
print(report)

print("Confusion Matrix (Malignant, Benign):")
print(matrix)


# 7. Save the trained model
model_path = MODEL_DIR / "disease_prediction_model.joblib"
joblib.dump(model, model_path)

# 8. Save evaluation report
report_path = OUTPUT_DIR / "evaluation_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("Breast Cancer Classification Model\n")
    file.write(f"Dataset records: {len(df)}\n")
    file.write(f"Number of features: {X.shape[1]}\n")
    file.write(f"Training records: {len(X_train)}\n")
    file.write(f"Testing records: {len(X_test)}\n\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\nConfusion Matrix (Malignant, Benign):\n")
    file.write(str(matrix))
    file.write("\n")

print("\nModel saved to:", model_path)
print("Evaluation report saved to:", report_path)
print("\nTraining completed successfully!")
print("Educational demonstration only; not a medical diagnosis.")
