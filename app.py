import pandas as pd
from flask import Flask, request, jsonify
import joblib

# Initialize Flask app (ONLY ONCE)
app = Flask(__name__)

# Load trained model and preprocessing objects
model = joblib.load("logistic_regression_model.joblib")
feature_names = joblib.load("feature_names.joblib")
original_categorical_cols = joblib.load("categorical_cols.joblib")
scaler = joblib.load("scaler.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Convert input JSON to DataFrame
        if isinstance(data, dict):
            df_input = pd.DataFrame([data])
        elif isinstance(data, list):
            df_input = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format"}), 400

        # Convert MoistureDetected to int if present
        if "MoistureDetected" in df_input.columns:
            df_input["MoistureDetected"] = df_input["MoistureDetected"].astype(int)

        # One-hot encoding
        df_processed = pd.get_dummies(
            df_input,
            columns=original_categorical_cols,
            drop_first=True
        )

        # Align features with training data
        df_final = df_processed.reindex(
            columns=feature_names,
            fill_value=0
        )

        # Scale features
        scaled_data = scaler.transform(df_final)

        # Predict
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)

        results = []
        for i in range(len(prediction)):
            results.append({
                "prediction": int(prediction[i]),
                "probability_class_0": float(prediction_proba[i][0]),
                "probability_class_1": float(prediction_proba[i][1])
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Render-compatible run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
