import pandas as pd
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and preprocessing information
model = joblib.load('logistic_regression_model.joblib')
feature_names = joblib.load('feature_names.joblib')
original_categorical_cols = joblib.load('categorical_cols.joblib')
scaler = joblib.load('scaler.joblib') # Load the scaler

# Initialize Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        # Convert incoming data to DataFrame
        # Ensure data is in list format if a single record is sent
        if isinstance(data, dict):
            df_input = pd.DataFrame([data])
        elif isinstance(data, list):
            df_input = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid input data format, expected dict or list of dicts'}), 400

        # Convert 'MoistureDetected' to int if present
        if 'MoistureDetected' in df_input.columns:
            df_input['MoistureDetected'] = df_input['MoistureDetected'].astype(int)

        # Apply one-hot encoding to categorical columns
        df_processed = pd.get_dummies(df_input, columns=original_categorical_cols, drop_first=True)

        # Reindex to ensure all feature_names are present and in the correct order
        # Fill missing columns with 0
        df_final = df_processed.reindex(columns=feature_names, fill_value=0)

        # Apply scaler to the final DataFrame
        scaled_data = scaler.transform(df_final) # Apply scaling

        # Make prediction
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)

        results = []
        for i in range(len(prediction)):
            results.append({
                'prediction': int(prediction[i]),
                'probability_class_0': float(prediction_proba[i][0]),
                'probability_class_1': float(prediction_proba[i][1])
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
