import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Load unique values and relationships
unique_values_df = pd.read_excel('final_data.xlsx')

# Extract unique values for initial dropdown
states = unique_values_df['state'].dropna().unique()

# Define function to get filtered values based on the selection
def get_filtered_values(df, filter_col, filter_val, return_col):
    filtered_values = df[df[filter_col] == filter_val][return_col].dropna().unique()
    return filtered_values.tolist()

st.title('Car Price Prediction')

# Select State
selected_state = st.selectbox('State', states)

# Filter Registration Year based on selected state
filtered_registration_years = get_filtered_values(unique_values_df, 'state', selected_state, 'Registration Year')
selected_registration_year = st.selectbox('Registration Year', filtered_registration_years)

# Filter models based on selected registration year
filtered_models = get_filtered_values(unique_values_df, 'Registration Year', selected_registration_year, 'model')
selected_model = st.selectbox('Model', filtered_models)

# Filter RTO codes based on selected model
filtered_rtos = get_filtered_values(unique_values_df, 'model', selected_model, 'RTO')
selected_rto = st.selectbox('RTO', filtered_rtos)

# Filter Max Power based on selected RTO
filtered_max_power = get_filtered_values(unique_values_df, 'RTO', selected_rto, 'Max Power')
selected_max_power = st.selectbox('Max Power', filtered_max_power)

# Input fields for features
wheel_base = st.number_input('Wheel Base')
width = st.number_input('Width')
kms_driven = st.number_input('Kms Driven')
central_variant_id = st.number_input('Central Variant ID')

# Prepare input data
if st.button('Predict'):
    # Create DataFrame from user input
    input_data = {
        'state': selected_state,
        'Registration Year': selected_registration_year,
        'model': selected_model,
        'RTO': selected_rto,
        'Max Power': selected_max_power,
        'Wheel Base': wheel_base,
        'Width': width,
        'Kms Driven': kms_driven,
        'centralVariantId': central_variant_id
    }
    input_df = pd.DataFrame([input_data])

    # Load the trained model and label encoder
    with open('trained_model.pkl', 'rb') as model_file:
        grid_search = pickle.load(model_file)
    
    with open('label_encoders.pkl', 'rb') as encoder_file:
        label_encoders = pickle.load(encoder_file)
    
    # Encode the input data using the loaded encoders
    for col in input_df.select_dtypes(include='object').columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    # Ensure input_df has the same columns as x_train
    x_train_columns = grid_search.best_estimator_.feature_names_in_
    missing_cols = set(x_train_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Fill missing columns with 0 or any other default value

    # Ensure the order of columns matches
    input_df = input_df[x_train_columns]

    # Predict with the best model
    predicted_price = grid_search.predict(input_df)[0]

    # Display the predicted price
    st.write(f"Predicted price: ₹{predicted_price:.2f}")

    # Model evaluation (optional)
    # (Assuming you still want to show evaluation metrics)
    df = pd.read_excel('final_data.xlsx')
    df.drop(columns=["Unnamed: 0"], inplace=True)
    for col in df.select_dtypes(include="object").columns:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])

    x = df.drop(columns=["price"])
    y = df["price"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
    
    y_pred = grid_search.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Model Evaluation:")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"R-squared: {r2}")
