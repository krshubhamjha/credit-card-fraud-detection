import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import json

# load model
with open('model/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# load scaler
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# load feature names
with open('model/feature_names.json', 'r') as f:
    feature_names = json.load(f)

# app title
st.title("Credit Card Fraud Detection")
st.write("Enter transaction details below to check if it is fraud or legitimate.")
st.write("---")

# user input
st.subheader("Transaction Details")

amount = st.number_input("Transaction Amount (€)", min_value=0.0, value=100.0)
hour = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=12)

st.write("---")
st.subheader("V Features (from PCA)")

# two columns for cleaner layout
col1, col2 = st.columns(2)

v_features = {}
for i in range(1, 29):
    if i <= 14:
        with col1:
            v_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.6f")
    else:
        with col2:
            v_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.6f")

st.write("---")

# predict button
if st.button("Check Transaction"):

    # scale amount
    amount_scaled = scaler.transform([[amount]])[0][0]

    # build input dataframe
    input_data = {
        'V1':  v_features['V1'],
        'V2':  v_features['V2'],
        'V3':  v_features['V3'],
        'V4':  v_features['V4'],
        'V5':  v_features['V5'],
        'V6':  v_features['V6'],
        'V7':  v_features['V7'],
        'V8':  v_features['V8'],
        'V9':  v_features['V9'],
        'V10': v_features['V10'],
        'V11': v_features['V11'],
        'V12': v_features['V12'],
        'V13': v_features['V13'],
        'V14': v_features['V14'],
        'V15': v_features['V15'],
        'V16': v_features['V16'],
        'V17': v_features['V17'],
        'V18': v_features['V18'],
        'V19': v_features['V19'],
        'V20': v_features['V20'],
        'V21': v_features['V21'],
        'V22': v_features['V22'],
        'V23': v_features['V23'],
        'V24': v_features['V24'],
        'V25': v_features['V25'],
        'V26': v_features['V26'],
        'V27': v_features['V27'],
        'V28': v_features['V28'],
        'Amount_Scaled': amount_scaled,
        'Time_Scaled': -0.796134,
        'Hour': hour
    }

    input_df = pd.DataFrame([input_data])[feature_names]

    # predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # show result
    st.write("---")
    if prediction == 1:
        st.error(f"FRAUD DETECTED! Fraud Probability: {probability:.2%}")
    else:
        st.success(f"LEGITIMATE Transaction. Fraud Probability: {probability:.2%}")

    # shap waterfall
    st.subheader("Why did model make this decision?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=feature_names
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()