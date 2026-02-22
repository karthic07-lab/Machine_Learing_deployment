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
original_categorical_cols = joblib.load(
    os.path.join(BASE_DIR, "categorical_cols.joblib")
)

# -----------------------------
# HARD-CODED FEATURE LIST
# -----------------------------
FEATURE_NAMES = [
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

        # Convert JSON to DataFrame
        df = pd.DataFrame([data])

        # Type correction
        if "MoistureDetected" in df.columns:
            df["MoistureDetected"] = df["MoistureDetected"].astype(int)

        # One-hot encoding (if applicable)
        df = pd.get_dummies(
            df,
            columns=original_categorical_cols,
            drop_first=True
        )

        # Align features
        df = df.reindex(columns=FEATURE_NAMES, fill_value=0)

        # Scale
        df_scaled = scaler.transform(df)

        # Predict
        pred = model.predict(df_scaled)
        prob = model.predict_proba(df_scaled)

        return jsonify({
            "prediction": int(pred[0]),
            "probability_class_0": float(prob[0][0]),
            "probability_class_1": float(prob[0][1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Render PORT binding (MANDATORY)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("Starting server on port:", port)
    app.run(host="0.0.0.0", port=port)
