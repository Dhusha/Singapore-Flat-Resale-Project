import pandas as pd
import numpy as np
import streamlit as st
import json
import pickle

def main():
    st.set_page_config(layout="wide")

    st.markdown("<h1 style='text-align:center;'>🏠 Singapore Flat Resale Price Predictor 🚀</h1>", unsafe_allow_html=True)

    with st.sidebar:
        selected = st.selectbox(
            "Menu 📋",
            ["Home", "Discover Insights", "Prediction"],
            index=0,
            format_func=lambda x: x.upper(),  # Optional: Format labels to uppercase
            help="Choose a section to navigate"
        )

    if selected == 'Home':
        left, right = st.columns([2, 1])

        with left:
            st.write("""
            <div style='text-align:center'>
                <h2 style='color:#006666;'>Project Overview 🎯</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            ##### ***Welcome to the Singapore Resale Flat Price Prediction Application. This project aims to develop a machine learning model that predicts the resale prices of flats in Singapore based on historical transaction data. The application assists potential buyers and sellers in estimating the resale value of flats, leveraging models such as Decision Tree, Random Forest, and Linear Regression.*** 💡
            """)

        with right:
            # Displaying an image
            st.image(r"Sibnbngaporeflatesimage.jpg")

        left, right = st.columns([2, 3])
        with left:
            st.markdown("<h3 style='color:red;'>TECHNOLOGY USED 🛠️</h3>", unsafe_allow_html=True)
            st.write("- Python (Pandas, NumPy) 🐍")
            st.write("- Scikit-Learn 🧠")
            st.write("- Machine Learning 🤖")
            st.write("- Streamlit 🌟")

        with right:
            st.markdown("<h3 style='color:red;'>MACHINE LEARNING MODELS 📊</h3>", unsafe_allow_html=True)
            st.write("#### REGRESSION")
            st.write("- Decision Tree Regressor 🌳")
            st.write("- Random Forest Regressor 🌲")
            st.write("- Linear Regression 📈")

    elif selected == 'Discover Insights':
        st.write("""
            <div style='text-align:center'>
                <h2 style='color:#006666;'>About Data 📊</h2>
            </div>
            """, unsafe_allow_html=True)

        st.header("Problem Statement: 🎯")
        st.write("The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.")

        st.write("---")

        st.header("Motivation: 💡")
        st.write("The resale flat market in Singapore is highly competitive, and it can be challenging to accurately estimate the resale value of a flat. There are many factors that can affect resale prices, such as location, flat type, floor area, and lease duration. A predictive model can help to overcome these challenges by providing users with an estimated resale price based on these factors.")

        st.write("---")

        st.header("Data Description: 📋")
        st.write("""
            - **Town:** The town where the flat is located. 🏙️
            - **Flat Type:** The type of the flat (e.g., 3-room, 4-room). 🏠
            - **Street Name:** The name of the street where the flat is located. 🛣️
            - **Storey Range:** The range of the storey where the flat is located. 🏢
            - **Flat Model:** The model of the flat. 🏘️
            - **Floor Area (sqm):** The floor area of the flat in square meters. 📏
            - **Lease Commence Date:** The year the lease of the flat commenced. 📅
            - **Year:** The year the flat is being resold. 🗓️
            - **Block:** The block number of the flat. 🏢
            """)

        st.write("---")

        st.header("Methodology: 🧪")
        st.write("""
            - **Data Preprocessing:** Data cleaning, encoding categorical variables, and feature scaling. 🧹
            - **Model Training:** Training multiple machine learning models including Decision Tree, Random Forest, and Linear Regression. 🌳
            - **Model Evaluation:** Evaluating the models based on performance metrics like R-squared and Mean Absolute Error (MAE). 📊
            """)

        st.write("---")

        st.header("Results: 🏆")
        st.write("The application provides predictions using three different models: Decision Tree Regressor, Random Forest Regressor, and Linear Regression. Users can compare the predictions from these models to get an estimated resale price of a flat.")

        st.write("---")

        st.header("Future Work: 🚀")
        st.write("""
            - Incorporate additional features such as proximity to amenities and transportation. 🚇
            - Enhance the model by using more advanced machine learning techniques and algorithms. 🤖
            - Continuously update the model with new data to improve accuracy. 🔄
            """)

        st.write("---")

        st.header("Conclusion: 🎉")
        st.write("This application serves as a valuable tool for both potential buyers and sellers in the resale flat market in Singapore. By providing accurate resale price predictions, it helps users make informed decisions.")
    
    elif selected == 'Prediction':
        with open(r"town.json", 'r') as file:
            town = json.load(file)
        with open(r"flat_type.json", 'r') as file:
            flat_type = json.load(file)
        with open(r"street_name.json", 'r') as file:
            street_name = json.load(file)
        with open(r"storey_range.json", 'r') as file:
            storey_range = json.load(file)
        with open(r"flat_model.json", 'r') as file:
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
                with open(r"data.pkl.gz", 'rb') as file:
                    dt = pickle.load(file)
                with open(r"randomforest.pkl", 'rb') as f:
                    rf = pickle.load(file)
                with open(r"linearregg.pkl", 'rb') as f:
                    lr = pickle.load(file)

                # Load label encoders
                with open(r"label_encoder1twn.pkl", 'rb') as file:
                    le1 = pickle.load(file)
                with open(r"label_encoder2FT.pkl", 'rb') as file:
                    le2 = pickle.load(file)
                with open(r"label_encoder3SN.pkl", 'rb') as file:
                    le3 = pickle.load(file)
                with open(r"label_encoder4SR.pkl", 'rb') as file:
                    le4 = pickle.load(file)
                with open(r"label_encoder5FM.pkl", 'rb') as file:
                    le5 = pickle.load(file)

                # Encode categorical variables
                town_encoded = le1.transform([Town])[0]
                flat_type_encoded = le2.transform([Flat_type])[0]
                street_name_encoded = le3.transform([Street_name])[0]
                storey_range_encoded = le4.transform([Storey_range])[0]
                flat_model_encoded = le5.transform([Flat_model])[0]

                # Create input array with correct data types
                ns = np.array([[
                    float(month),
                    town_encoded,
                    flat_type_encoded,
                    float(Block),
                    street_name_encoded,
                    storey_range_encoded,
                    float(Floor_area_sqm),
                    flat_model_encoded,
                    float(Lease_commence_date),
                    float(Year)
                ]])

                # Make predictions
                dt_pred = dt.predict(ns)
                rf_pred = rf.predict(ns)
                lr_pred = lr.predict(ns)

                # Display the results
                st.write('## :green[Predicted Resale Price:] ')
                st.write('### :red[Decision Tree Regressor] :', dt_pred[0])
                st.write('### :red[Random Forest Regressor] :', rf_pred[0])
                st.write('### :red[Linear Regression] :', lr_pred[0])
if __name__ == "__main__":
    main()
