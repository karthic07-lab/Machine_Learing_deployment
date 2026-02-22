import os
import pandas as pd
from flask import Flask, request, jsonify
import joblib

# -----------------------------
# Flask App Initialization
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Absolute Path Handling (RENDER SAFE)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "logistic_regression_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.joblib")
CATEGORICAL_PATH = os.path.join(BASE_DIR, "categorical_cols.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")

# -----------------------------
# Load ML Artifacts
# -----------------------------
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
original_categorical_cols = joblib.load(CATEGORICAL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Convert JSON to DataFrame
        if isinstance(data, dict):
            df_input = pd.DataFrame([data])
        elif isinstance(data, list):
            df_input = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format"}), 400

        # Example type fix (keep only if relevant)
        if "MoistureDetected" in df_input.columns:
            df_input["MoistureDetected"] = df_input["MoistureDetected"].astype(int)

        # One-hot encode categorical columns
        df_processed = pd.get_dummies(
            df_input,
            columns=original_categorical_cols,
            drop_first=True
        )

        # Align columns with training features
        df_final = df_processed.reindex(
            columns=feature_names,
            fill_value=0
        )

        # Scale features
        scaled_data = scaler.transform(df_final)

        # Predict
        preds = model.predict(scaled_data)
        probs = model.predict_proba(scaled_data)

        results = []
        for i in range(len(preds)):
            results.append({
                "prediction": int(preds[i]),
                "probability_class_0": float(probs[i][0]),
                "probability_class_1": float(probs[i][1])
            })

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Render-Compatible Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
