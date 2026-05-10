# ==========================================
# CALIFORNIA HOUSE PRICE PREDICTION APP
# STREAMLIT WEB APPLICATION
# ==========================================


# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd
import pickle


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = pickle.load(
    open("lightgbm_model.pkl", "rb")
)

scaler = pickle.load(
    open("scaler.pkl", "rb")
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.title("Welocme 😊 To California House  ")
st.set_page_config(
    page_title="Let's Find a house together ",
    page_icon="🏠",
    layout="centered"
)

st.image("image.png")


# ==========================================
# TITLE
# ==========================================

st.title("🏠 California House Price Prediction")

st.write(
    "Predict California house prices using a trained LightGBM Machine Learning model."
)


# ==========================================
# USER INPUTS
# ==========================================

st.header("Enter House Details")


longitude = st.number_input(
    "Longitude",
    value=-122.23
)

latitude = st.number_input(
    "Latitude",
    value=37.88
)

housing_median_age = st.slider(
    "Housing Median Age",
    1,
    60,
    20
)

total_rooms = st.number_input(
    "Total Rooms",
    value=2000
)

total_bedrooms = st.number_input(
    "Total Bedrooms",
    value=400
)

population = st.number_input(
    "Population",
    value=1000
)

households = st.number_input(
    "Households",
    value=300
)

median_income = st.number_input(
    "Median Income(in tens of thousands)",
    min_value=3.5,
    
)


# ==========================================
# OCEAN PROXIMITY
# ==========================================

ocean_proximity = st.selectbox(

    "Ocean Proximity",

    (
        "<1H_OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR_BAY",
        "NEAR_OCEAN"
    )

)


# ==========================================
# ONE HOT ENCODING FOR OCEAN PROXIMITY
# ==========================================

ocean_proximity_1H_OCEAN = 0
ocean_proximity_INLAND = 0
ocean_proximity_ISLAND = 0
ocean_proximity_NEAR_BAY = 0
ocean_proximity_NEAR_OCEAN = 0


if ocean_proximity == "<1H_OCEAN":
    ocean_proximity_1H_OCEAN = 1

elif ocean_proximity == "INLAND":
    ocean_proximity_INLAND = 1

elif ocean_proximity == "ISLAND":
    ocean_proximity_ISLAND = 1

elif ocean_proximity == "NEAR_BAY":
    ocean_proximity_NEAR_BAY = 1

elif ocean_proximity == "NEAR_OCEAN":
    ocean_proximity_NEAR_OCEAN = 1


# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame({

    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity_<1H_OCEAN": [ocean_proximity_1H_OCEAN],
    "ocean_proximity_INLAND": [ocean_proximity_INLAND],
    "ocean_proximity_ISLAND": [ocean_proximity_ISLAND],
    "ocean_proximity_NEAR_BAY": [ocean_proximity_NEAR_BAY],
    "ocean_proximity_NEAR_OCEAN": [ocean_proximity_NEAR_OCEAN]

})


# ==========================================
# SCALE INPUT DATA
# ==========================================

scaled_data = scaler.transform(input_data)


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("Predict House Price"):

    prediction = model.predict(scaled_data)

    st.success(
        f"Predicted House Price: ${prediction[0]:,.2f}  Thank  You for using our platform 😊❤️"
    )

st.image("people.png")