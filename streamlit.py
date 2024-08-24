import streamlit as st
import pandas as pd
import pickle

# Load unique values and relationships
unique_values_df = pd.read_excel('final_data.xlsx')

# Extract unique values for initial dropdown
states = unique_values_df['state'].dropna().unique()

# Define function to get filtered values based on the selection
def get_filtered_values(df, filter_col, filter_val, return_col):
    filtered_values = df[df[filter_col] == filter_val][return_col].dropna().unique()
    return filtered_values.tolist()

# Set the layout to wide mode
st.set_page_config(layout="wide")

# Center the title using Markdown
st.markdown("<h1 style='text-align: center;'>🚗 Car Price Prediction</h1>", unsafe_allow_html=True)

# Create columns for input fields with specified proportions
col1, col2 = st.columns([4, 5])

with col1:

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
with col2:

    # Input fields for features
    wheel_base = st.number_input('Wheel Base', min_value=0.0)
    width = st.number_input('Width', min_value=0.0)
    kms_driven = st.number_input('Kms Driven', min_value=0.0)
    central_variant_id = st.number_input('Central Variant ID', min_value=0)

    # Button to trigger prediction
    if st.button('Predict'):
        # Check if all required inputs are provided
        if selected_state and selected_registration_year and selected_model and selected_rto and selected_max_power:
            # Load the model and label encoder
            with open('trained_model.pkl', 'rb') as model_file:
                trained_model = pickle.load(model_file)

            with open('label_encoder.pkl', 'rb') as encoder_file:
                label_encoder = pickle.load(encoder_file)

            # Prepare input data
            input_data = {
                'model': selected_model,
                'Wheel Base': wheel_base,
                'Width': width,
                'state': selected_state,
                'Max Power': selected_max_power,
                'RTO': selected_rto,
                'Registration Year': selected_registration_year,
                'Kms Driven': kms_driven,
                'centralVariantId': central_variant_id
            }

            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])

            # Preprocess the input data
            for col in input_df.select_dtypes(include='object').columns:
                if col in label_encoder.classes_:
                    try:
                        input_df[col] = label_encoder.transform(input_df[col])
                    except KeyError:
                        input_df[col] = -1  # or some default value
                else:
                    input_df[col] = -1  # or some default value

            # Ensure all columns match the model's expected input
            input_df = input_df.reindex(columns=['model', 'Wheel Base', 'Width', 'state', 'Max Power', 'RTO', 'Registration Year', 'Kms Driven', 'centralVariantId'], fill_value=0)

            # Make the prediction
            predicted_price = trained_model.predict(input_df)
            st.subheader(f"💰 Predicted Price: ₹ {int(predicted_price[0]):,}")
        else:
            st.warning("Please fill in all the fields before predicting.")
