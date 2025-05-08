from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd

app = Flask(__name__)

# Load the trained model (pipeline + preprocessor)
model_path = os.path.join(os.path.dirname(__file__), 'logistic_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define the expected input fields 
numeric_cols = [
    'CreditScore', 'FirstTimeHomebuyer', 'MIP', 'Units', 'OCLTV', 'DTI',
    'OrigUPB', 'LTV', 'OrigInterestRate', 'OrigLoanTerm',
    'LoanPurpose_P', 'LoanPurpose_R',
    'Occupancy_INV', 'Occupancy_PR', 'Occupancy_SH',
    'PropertyType_Condo', 'PropertyType_Multi-Family',
    'PropertyType_Other', 'PropertyType_Single-Family',
    'Channel_B', 'Channel_C', 'Channel_R', 'Channel_T'
]

categorical_cols = ['PropertyState', 'NumBorrowers', 'CreditScoreCategory', 'DTICategory']
expected_fields = numeric_cols + categorical_cols

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form
        input_data = {}

        for col in numeric_cols:
            input_data[col] = float(form_data.get(col, 0))

        for col in categorical_cols:
            input_data[col] = form_data.get(col, '')

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        return render_template('index.html', prediction_text=f'Delinquency Prediction: {int(prediction)}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        input_data = {}

        for col in expected_fields:
            if col not in data:
                return jsonify({'error': f'Missing input: {col}'}), 400

        for col in numeric_cols:
            try:
                input_data[col] = float(data[col])
            except ValueError:
                return jsonify({'error': f'Invalid numeric value for {col}'}), 400

        for col in categorical_cols:
            input_data[col] = str(data[col])

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use the environment's PORT variable or fallback to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

