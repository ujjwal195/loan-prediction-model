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

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction via HTML form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form
        input_data = {}

        # Extract and cast numeric fields
        for col in numeric_cols:
            input_data[col] = float(form_data.get(col, 0))

        # Extract categorical fields as strings
        for col in categorical_cols:
            input_data[col] = form_data.get(col, '')

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict
        prediction = model.predict(input_df)[0]
        return render_template('index.html', prediction_text=f'Delinquency Prediction: {int(prediction)}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

# Prediction via API
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        input_data = {}

        # Check all required fields
        for col in expected_fields:
            if col not in data:
                return jsonify({'error': f'Missing input: {col}'}), 400

        # Cast numeric
        for col in numeric_cols:
            try:
                input_data[col] = float(data[col])
            except ValueError:
                return jsonify({'error': f'Invalid numeric value for {col}'}), 400

        # Keep categorical as strings
        for col in categorical_cols:
            input_data[col] = str(data[col])

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict
        prediction = model.predict(input_df)[0]
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    st.title("Loan Delinquency Prediction")

    # Input fields for numeric columns
    input_data = {}
    
    # Numeric columns input
    for col in numeric_cols:
        input_data[col] = st.number_input(col, value=0.0)

    # Categorical columns input
    for col in categorical_cols:
        input_data[col] = st.text_input(col)

    # Prediction button
    if st.button("Predict"):
        try:
            # Convert input data to DataFrame for prediction
            input_df = pd.DataFrame([input_data])
            # Predict
            prediction = model.predict(input_df)[0]
            st.success(f"Prediction: {'Delinquent' if prediction == 1 else 'Not Delinquent'}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
