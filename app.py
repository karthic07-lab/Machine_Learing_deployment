import os
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# -------- Load model files --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "logistic_regression_model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
original_categorical_cols = joblib.load(os.path.join(BASE_DIR, "categorical_cols.joblib"))

# HARD-CODED feature list (since feature_names.joblib is missing)
FEATURE_NAMES = [
    "Voltage",
    "Current",
    "Temperature",
    "SOC",
    "MoistureDetected"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    df = pd.DataFrame([data])

    if "MoistureDetected" in df.columns:
        df["MoistureDetected"] = df["MoistureDetected"].astype(int)

    df = pd.get_dummies(
        df,
        columns=original_categorical_cols,
        drop_first=True
    )

    df = df.reindex(columns=FEATURE_NAMES, fill_value=0)
    df_scaled = scaler.transform(df)

    pred = model.predict(df_scaled)
    prob = model.predict_proba(df_scaled)

    return jsonify({
        "prediction": int(pred[0]),
        "probability_class_0": float(prob[0][0]),
        "probability_class_1": float(prob[0][1])
    })

# -------- Render PORT binding --------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("Starting server on port:", port)   # IMPORTANT
    app.run(host="0.0.0.0", port=port)
