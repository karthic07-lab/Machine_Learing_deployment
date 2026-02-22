import os
import pandas as pd
from flask import Flask, request, jsonify
import joblib

# -----------------------------
# Flask initialization
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Base directory (Render safe)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Load ML artifacts
# -----------------------------
model = joblib.load(os.path.join(BASE_DIR, "logistic_regression_model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
original_categorical_cols = joblib.load(os.path.join(BASE_DIR, "categorical_cols.joblib"))

# -----------------------------
# MANUALLY DEFINE FEATURE NAMES
# (MUST MATCH TRAINING DATA)
# -----------------------------
feature_names = [
    "Voltage",
    "Current",
    "Temperature",
    "SOC",
    "MoistureDetected"
]

# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # JSON → DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format"}), 400

        # Ensure correct type
        if "MoistureDetected" in df.columns:
            df["MoistureDetected"] = df["MoistureDetected"].astype(int)

        # One-hot encoding (if any categorical columns)
        df = pd.get_dummies(
            df,
            columns=original_categorical_cols,
            drop_first=True
        )

        # Align features
        df = df.reindex(columns=feature_names, fill_value=0)

        # Scale
        df_scaled = scaler.transform(df)

        # Predict
        preds = model.predict(df_scaled)
        probs = model.predict_proba(df_scaled)

        return jsonify([{
            "prediction": int(preds[0]),
            "probability_class_0": float(probs[0][0]),
            "probability_class_1": float(probs[0][1])
        }]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Render port binding
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
