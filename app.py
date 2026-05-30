import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

def main():
    st.markdown("<p style='font-family:Serif Fonts; font-size:40px'>Fraud Detection App</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='font-family:cursive; font-size:large'>Welcome</p>", unsafe_allow_html=True)
    st.sidebar.image('hands-4519047.png')

    # Collect user inputs (FIX: added value= to avoid int/float mismatch)
    distance_from_home = st.number_input("Distance from Home", min_value=0.004874, max_value=10632.723672, value=0.004874)
    distance_from_last_transaction = st.number_input("Distance from Last Transaction", min_value=0.000118, max_value=11851.104565, value=0.000118)
    ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", min_value=0.004399, max_value=267.802942, value=0.004399)
    repeat_retailer = st.selectbox("Repeat Retailer", [0, 1])
    used_chip = st.selectbox("Used Chip", [0, 1])
    used_pin_number = st.selectbox("Used Pin Number", [0, 1])
    online_order = st.selectbox("Online Order", [0, 1])

    if st.button("Predict"):
        data = {
            'distance_from_home': [distance_from_home],
            'distance_from_last_transaction': [distance_from_last_transaction],
            'ratio_to_median_purchase_price': [ratio_to_median_purchase_price],
            'repeat_retailer': [repeat_retailer],
            'used_chip': [used_chip],
            'used_pin_number': [used_pin_number],
            'online_order': [online_order]
        }
        input_df = pd.DataFrame(data)
        prediction = model.predict(input_df)

        if prediction[0] == 1:
            st.error("Fraudulent Transaction ❌")
        else:
            st.success("Non-Fraudulent Transaction ✅")

if __name__ == '__main__':
    main()
