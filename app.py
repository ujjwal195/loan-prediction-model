import streamlit as st
import pickle
import pandas as pd
import os

# Load the trained pipeline (model + preprocessor)
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

# Streamlit app UI and prediction logic
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
    main()
