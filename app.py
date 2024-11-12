from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load('latest_claim_predictor_model.pkl')

app = Flask(__name__)

# Serve the index.html file
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint to receive user input and return the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request (JSON)
    data = request.get_json(force=True)

    # Prepare the input data for prediction (adjust as per model input requirements)
    input_data = np.array([[
        data['age'],
        data['bmi'],
        data['bloodpressure'],
        data['children'],
        1 if data['smoker'] == 'Yes' else 0,  # Convert categorical data
        data['region']
    ]])

    # Get the prediction
    prediction = model.predict(input_data)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)