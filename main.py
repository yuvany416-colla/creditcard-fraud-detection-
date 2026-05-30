import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model

# Load the trained fusion model
fusion_model = load_model("fusion_model.h5")

sequence_length = 1
lstm_input_dim = 7

# FIX Bug 2: correct reshape for single sample
def preprocess_input_lstm(input_data):
    input_array = np.array(input_data)
    processed_input = input_array.reshape(1, sequence_length, lstm_input_dim)
    return processed_input

# FIX Bug 1: removed wrong single-row StandardScaler, use raw values
def preprocess_input_encoder(input_data):
    input_df = pd.DataFrame(input_data, index=[0])
    return input_df.values  # shape (1, 7)

def main():
    st.title("Fraud Detection App")
    st.sidebar.subheader("Enter input data to predict whether it's fraud or not.")
    st.sidebar.image('hands-4519047.png')

    distance_from_home = st.number_input("Distance from home", value=0.0)
    distance_from_last_transaction = st.number_input("Distance from last transaction", value=0.0)
    ratio_to_median_purchase_price = st.number_input("Ratio to median purchase price", value=0.0)
    repeat_retailer = st.number_input("Repeat retailer", value=0.0)
    used_chip = st.number_input("Used chip", value=0.0)
    used_pin_number = st.number_input("Used PIN number", value=0.0)
    online_order = st.number_input("Online order", value=0.0)

    if st.button("Predict"):
        input_data = {
            'distance_from_home': distance_from_home,
            'distance_from_last_transaction': distance_from_last_transaction,
            'ratio_to_median_purchase_price': ratio_to_median_purchase_price,
            'repeat_retailer': repeat_retailer,
            'used_chip': used_chip,
            'used_pin_number': used_pin_number,
            'online_order': online_order
        }

        # FIX Bug 1: preprocess for encoder
        processed_input_encoder = preprocess_input_encoder(input_data)

        # FIX Bug 3: correct argument passed to lstm
        processed_input_lstm = preprocess_input_lstm(np.array([list(input_data.values())]))

        # Make prediction
        prediction = fusion_model.predict([processed_input_encoder, processed_input_lstm])

        # FIX Bug 4 & 5: correct indexing + proper st.error/st.success display
        if prediction[0][0] > 0.5:
            st.error("Fraudulent Transaction ❌")
        else:
            st.success("Non-Fraudulent Transaction ✅")

if __name__ == "__main__":
    main()
