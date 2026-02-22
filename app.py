import os
import pandas as pd
from flask import Flask, request, jsonify
import joblib

# -----------------------------
# Flask initialization (ONLY ONCE)
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Base directory (Render safe)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Load ML artifacts
# -----------------------------
MODEL_PATH = os.path.join(BASE_DIR, "logistic_regression_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.joblib")
CATEGORICAL_PATH = os.path.join(BASE_DIR, "categorical_cols.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)
original_categorical_cols = joblib.load(CATEGORICAL_PATH)

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

        # Type fix if present
        if "MoistureDetected" in df.columns:
            df["MoistureDetected"] = df["MoistureDetected"].astype(int)

        # One-hot encoding
        df = pd.get_dummies(
            df,
            columns=original_categorical_cols,
            drop_first=True
        )

        # Feature alignment
        df = df.reindex(columns=feature_names, fill_value=0)

        # Scaling
        df_scaled = scaler.transform(df)

        # Prediction
        preds = model.predict(df_scaled)
        probs = model.predict_proba(df_scaled)

        output = []
        for i in range(len(preds)):
            output.append({
                "prediction": int(preds[i]),
                "probability_class_0": float(probs[i][0]),
                "probability_class_1": float(probs[i][1])
            })

        return jsonify(output), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Render port binding (MANDATORY)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
