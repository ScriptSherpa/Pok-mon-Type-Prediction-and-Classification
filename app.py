import streamlit as st
import joblib
import pandas as pd

# Step 1: Load the trained model
loaded_model = joblib.load('trained_model.pkl')

# Step 2: Define the feature names (same as when the model was trained)
column_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 
                'feature11', 'feature12', 'feature13', 'feature14', 'feature15', 'feature16', 'feature17', 'feature18', 'feature19', 'feature20', 
                'feature21', 'feature22', 'feature23', 'feature24', 'feature25', 'feature26', 'feature27', 'feature28', 'feature29', 'feature30', 
                'feature31', 'feature32', 'feature33']  # Replace with actual feature names used during training

# Step 3: Create input fields for the user to enter feature values
st.title("Random Forest Model Prediction")

st.write("""
Please input values for each feature to make a prediction:
""")

# Create a dictionary to store user input for each feature
user_input = {}
for feature in column_names:
    user_input[feature] = st.number_input(feature, value=0.0, step=0.1)

# Step 4: Convert the input data to a DataFrame
X_new_df = pd.DataFrame([user_input], columns=column_names)

# Step 5: Make the prediction
if st.button('Predict'):
    prediction = loaded_model.predict(X_new_df)
    
    # Display the prediction
    if prediction == 0:
        st.write("Prediction: Class 0 (Not purchasing)")
    else:
        st.write("Prediction: Class 1 (Purchasing)")

