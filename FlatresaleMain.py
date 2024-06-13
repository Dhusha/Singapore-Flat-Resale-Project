import pandas as pd
import numpy as np
import streamlit as st
import json
import pickle

st.set_page_config(layout="wide")

st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Singapore Resale Flat Price Prediction Application</h1>
</div>
""", unsafe_allow_html=True)

tab1 = st.tabs(["Prediction with DT, RF & LR"])

with tab1:
    with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Singapore_Project\town.json", 'r') as file:
        town = json.load(file)
    with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Singapore_Project\flat_type.json", 'r') as file:
        flat_type = json.load(file)
    with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Singapore_Project\street_name.json", 'r') as file:
        street_name = json.load(file)
    with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Singapore_Project\storey_range.json", 'r') as file:
        storey_range = json.load(file)
    with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Singapore_Project\flat_model.json", 'r') as file:
        flat_model = json.load(file)

    # Define the possible values for the dropdown menus
    months = list(range(1, 13))
    Town = town
    Flat_type = flat_type
    Street_name = street_name
    Storey_range = storey_range
    Flat_model = flat_model

    # Define the widgets for user input
    with st.form("my_form"):
        col1, col2, col3 = st.columns([5, 2, 5])
        
        with col1:
            st.write(' ')
            month = st.selectbox("Month", months, key=1)
            Town = st.selectbox("Town", Town, key=2)
            Flat_type = st.selectbox('Flat Type', Flat_type, key=3)
            Block = st.number_input("Enter block", value=1, step=1)
            Street_name = st.selectbox("Street Name", Street_name, key=4)
        
        with col3:
            Storey_range = st.selectbox("Storey Range", Storey_range, key=5)
            Floor_area_sqm = st.number_input("Enter floor area (sqm)", value=50.0, step=0.1)
            Flat_model = st.selectbox("Flat Model", Flat_model, key=6)
            Lease_commence_date = st.number_input("Enter Lease commence date", value=1998, step=1)
            Year = st.number_input("Enter the year", value=1998, step=1)
            submit_button = st.form_submit_button(label="PREDICT Resale Price")
            st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: #009999;
                    color: white;
                    width: 100%;
                }
                </style>
            """, unsafe_allow_html=True)

        if submit_button:
            with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Singapore_Project\dt.pkl", 'rb') as file:
                dt = pickle.load(file)
            with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Singapore_Project\randomforest.pkl", 'rb') as f:
                rf = pickle.load(f)
            with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Singapore_Project\linearregg.pkl", 'rb') as f:
                lr = pickle.load(f)

            ns = np.array([[month, Town, Flat_type, Block, Street_name, Storey_range, 
                            Floor_area_sqm, Flat_model, Lease_commence_date, Year]])

            with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Singapore_Project\label_encoder1twn.pkl", 'rb') as file:
                le1 = pickle.load(file)
            with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Singapore_Project\label_encoder2FT.pkl", 'rb') as file:
                le2 = pickle.load(file)
            with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Singapore_Project\label_encoder3SN.pkl", 'rb') as file:
                le3 = pickle.load(file)
            with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Singapore_Project\label_encoder4SR.pkl", 'rb') as file:
                le4 = pickle.load(file)
            with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Singapore_Project\label_encoder5FM.pkl", 'rb') as file:
                le5 = pickle.load(file)

            # Encode the categorical variables
            en1 = le1.transform(ns[:, [1]])
            en2 = le2.transform(ns[:, [2]])
            en3 = le3.transform(ns[:, [4]])
            en4 = le4.transform(ns[:, [5]])
            en5 = le5.transform(ns[:, [7]])

            # Convert 1D arrays to 2D arrays
            en1 = en1[:, np.newaxis]
            en2 = en2[:, np.newaxis]
            en3 = en3[:, np.newaxis]
            en4 = en4[:, np.newaxis]
            en5 = en5[:, np.newaxis]

            ns = np.concatenate((ns[:, [0]], en1, en2, ns[:, [3]], en3, en4, ns[:, [6]], en5, ns[:, [8]], ns[:, [9]]), axis=1)
            
            # Make predictions
            dt_pred = dt.predict(ns)
            rf_pred = rf.predict(ns)
            lr_pred = lr.predict(ns)
            
            # Display the results
            st.write('## :green[Predicted Resale Price:] ')
            st.write('### :red[Decision Tree Regressor] :', dt_pred[0])
            st.write('### :red[Random Forest Regressor] :', rf_pred[0])
            st.write('### :red[Linear Regression] :', lr_pred[0])

