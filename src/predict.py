
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "disease_prediction_model.joblib"
DATA_PATH = ROOT / "data" / "breast_cancer.csv"

# Load trained model and dataset
model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

# Select one example record, excluding the target column
sample = data.drop(columns=["Diagnosis"]).iloc[[0]]

# Predict the class
prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

print("\n===== DISEASE PREDICTION =====")
print("Predicted category:", prediction)

for category, probability in zip(model.classes_, probabilities):
    print(f"Model probability for {category}: {probability:.2%}")

print("\nEducational demonstration only.")
print("This result is not a medical diagnosis.")
