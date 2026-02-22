from flask import Flask, request, jsonify
import joblib

# Initialize a Flask application instance
app = Flask(__name__)

# Export the trained model
model_filename = 'logistic_regression_model.joblib'
joblib.dump(model, model_filename)

print(f"Model successfully exported to {model_filename}")

# Save the StandardScaler objectscaler_filename = 'standard_scaler.joblib'joblib.dump(scaler, scaler_filename)print(f"StandardScaler successfully exported to {scaler_filename}")

 StandardScaler successfully exported to standard_scaler.joblib

# Load the trained model and scaler
try:
    model = joblib.load('logistic_regression_model.joblib')
    scaler = joblib.load('standard_scaler.joblib')
    # Define feature_columns based on the columns used during training
    # This assumes X_train.columns is available from the training phase.
    # In a real deployment, this would be saved alongside the model/scaler.
    # For this exercise, we will manually reconstruct based on previous notebook state.
    feature_columns = ['PackVoltage_V', 'CellVoltage_V', 'DemandVoltage_V', 'ChargeCurrent_A', 'DemandCurrent_A', 'SOC_%', 'MaxTemp_C', 'MinTemp_C', 'AvgTemp_C', 'AmbientTemp_C', 'InternalResistance_mOhm', 'StateOfHealth_%', 'VibrationLevel_mg', 'MoistureDetected', 'ChargePower_kW', 'Pressure_kPa', 'ChargingStage_Handshake', 'ChargingStage_Parameter_Config', 'ChargingStage_Recharge', 'BMS_Status_OK', 'BMS_Status_Warning']

    print("Model and scaler loaded successfully.")
    print(f"Expected feature columns: {feature_columns}")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None
    feature_columns = []

# Create a basic route for the root URL
@app.route('/')
def home():
    return 'Welcome to the EV Battery Thermal Runaway Prediction API!'

# Standard block to run the Flask application in debug mode
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
