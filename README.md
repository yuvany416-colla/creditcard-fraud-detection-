# Fraud Detection App — Deployment Guide

## Files Included
- `app.py`          → Logistic Regression model app
- `main.py`         → Fusion Deep Learning (Encoder + LSTM) model app
- `logistic_regression_model.pkl` → Trained LR model
- `fusion_model.h5`               → Trained Fusion model
- `hands-4519047.png`             → Sidebar image
- `requirements.txt`              → All dependencies
- `.streamlit/config.toml`        → Streamlit cloud config

## Deploy on Streamlit Cloud

1. Go to https://github.com and create a new PUBLIC repository
2. Upload ALL files from this zip (keep folder structure)
3. Go to https://share.streamlit.io
4. Click "New app"
5. Select your repository
6. Set Main file:
   - For Logistic Regression → app.py
   - For Deep Learning       → main.py
7. Click Deploy!

## Run Locally

pip install -r requirements.txt
streamlit run app.py
# or
streamlit run main.py

## Notes
- scikit-learn is pinned to 1.2.2 to match the saved model version
- tensorflow-cpu is used instead of tensorflow to reduce memory usage on cloud
