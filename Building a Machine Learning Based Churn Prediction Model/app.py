import streamlit as st
import pandas as pd
import pickle
import base64
import matplotlib.pyplot as plt
import seaborn as sns



# Load the trained model
model = pickle.load(open("model.sav", "rb"))

# Set the page title and icon
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon=":bar_chart:",
)

# Define a background image
st.markdown(
    """
    <style>
    .reportview-container {
        background: url('images.jpg');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define the app header
st.title("Customer Churn Prediction")

# Input section in the sidebar
st.sidebar.subheader("Input Data")
inputQuery1 = st.sidebar.number_input("Senior Citizen (0 for No, 1 for Yes):", min_value=0, max_value=1, value=0)
inputQuery2 = st.sidebar.number_input("Monthly Charges:", min_value=0.0)
inputQuery3 = st.sidebar.number_input("Total Charges:", min_value=0.0)
inputQuery4 = st.sidebar.radio("Gender:", ["Male", "Female"])
inputQuery5 = st.sidebar.radio("Partner:", ["Yes", "No"])
inputQuery6 = st.sidebar.radio("Dependents:", ["Yes", "No"])
inputQuery7 = st.sidebar.radio("Phone Service:", ["Yes", "No"])
inputQuery8 = st.sidebar.radio("Multiple Lines:", ["No phone service", "No", "Yes"])
inputQuery9 = st.sidebar.radio("Internet Service:", ["DSL", "Fiber optic", "No"])
inputQuery10 = st.sidebar.radio("Online Security:", ["No", "Yes", "No internet service"])
inputQuery11 = st.sidebar.radio("Online Backup:", ["No", "Yes", "No internet service"])
inputQuery12 = st.sidebar.radio("Device Protection:", ["No", "Yes", "No internet service"])
inputQuery13 = st.sidebar.radio("Tech Support:", ["No", "Yes", "No internet service"])
inputQuery14 = st.sidebar.radio("Streaming TV:", ["No", "Yes", "No internet service"])
inputQuery15 = st.sidebar.radio("Streaming Movies:", ["No", "Yes", "No internet service"])
inputQuery16 = st.sidebar.radio("Contract:", ["Month-to-month", "One year", "Two year"])
inputQuery17 = st.sidebar.radio("Paperless Billing:", ["Yes", "No"])
inputQuery18 = st.sidebar.radio("Payment Method:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
inputQuery19 = st.sidebar.number_input("Tenure (in months):", min_value=0)

# Store the selected input data in a dictionary
selected_input_data = {
    "Senior Citizen": inputQuery1,
    "Monthly Charges": inputQuery2,
    "Total Charges": inputQuery3,
    "Gender": inputQuery4,
    "Partner": inputQuery5,
    "Dependents": inputQuery6,
    "Phone Service": inputQuery7,
    "Multiple Lines": inputQuery8,
    "Internet Service": inputQuery9,
    "Online Security": inputQuery10,
    "Online Backup": inputQuery11,
    "Device Protection": inputQuery12,
    "Tech Support": inputQuery13,
    "Streaming TV": inputQuery14,
    "Streaming Movies": inputQuery15,
    "Contract": inputQuery16,
    "Paperless Billing": inputQuery17,
    "Payment Method": inputQuery18,
    "Tenure (in months)": inputQuery19
}

# Define a button to view input data
if st.sidebar.button("View Input Data"):
    # Display the input data in the main content area as a DataFrame
    st.subheader("Input Data")
    input_df = pd.DataFrame.from_dict(selected_input_data, orient='index', columns=["Value"])
    st.write(input_df)

# Delete selected input data
if st.sidebar.button("Delete Selected Input Data"):
    selected_input_data = {}

# Prediction section in the main content area
if st.sidebar.button("Customer Churn Prediction Model"):
    # Prepare the input data for prediction
    input_data = {
        "Senior Citizen": inputQuery1,
        "Monthly Charges": inputQuery2,
        "Total Charges": inputQuery3,
        "Gender": inputQuery4,
        "Partner": inputQuery5,
        "Dependents": inputQuery6,
        "Phone Service": inputQuery7,
        "Multiple Lines": inputQuery8,
        "Internet Service": inputQuery9,
        "Online Security": inputQuery10,
        "Online Backup": inputQuery11,
        "Device Protection": inputQuery12,
        "Tech Support": inputQuery13,
        "Streaming TV": inputQuery14,
        "Streaming Movies": inputQuery15,
        "Contract": inputQuery16,
        "Paperless Billing": inputQuery17,
        "Payment Method": inputQuery18,
        "Tenure (in months)": inputQuery19
    }

    input_df = pd.DataFrame.from_dict(input_data, orient='index', columns=["Value"]).T

    # Preprocess categorical variables using one-hot encoding
    categorical_cols = [
        "Gender",
        "Partner",
        "Dependents",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method"
    ]

    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Ensure that the input data contains all the columns used for training
    training_columns = model.feature_names_in_
    missing_columns = set(training_columns) - set(input_df.columns)

    # Fill in the missing columns with zeros
    for col in missing_columns:
        input_df[col] = 0

    # Reorder columns to match the training data
    input_df = input_df[training_columns]

    # Make the prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]

    # Display the prediction result in the main content area
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is likely to continue.")
    st.write(f"Confidence: {probability[0] * 100:.2f}%")

# Add a button to download the selected input data as a CSV file
if st.sidebar.button("Download Selected Input Data as CSV"):
    selected_input_data_df = pd.DataFrame.from_dict(selected_input_data, orient='index', columns=["Value"])
    selected_input_data_csv = selected_input_data_df.to_csv(index=False)
    b64 = base64.b64encode(selected_input_data_csv.encode()).decode()  # Convert to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="selected_input_data.csv">Download Selected Input Data</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sample data for demonstration purposes (replace with your data)
churn_count = 1869
no_churn_count = 5174

# Create a DataFrame for the predictions
prediction_df = pd.DataFrame({
    "Prediction": ["Churn", "No Churn"],
    "Count": [churn_count, no_churn_count]
})

# Add a button to visualize the distribution of predictions
if st.sidebar.button("Visualize Predictions"):
    st.subheader("Churn Predictions Distribution")
    
    # Create a pie chart to visualize the distribution
    fig, ax = plt.subplots()
    ax.pie(prediction_df["Count"], labels=prediction_df["Prediction"], autopct='%1.1f%%', startangle=90)
    
    # Display the pie chart in Streamlit
    st.pyplot(fig)



# Add a footer
st.markdown("---")
st.write("© 2023 Prasad D Wilagama")
