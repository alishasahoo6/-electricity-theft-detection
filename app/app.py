import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.data_processing import load_data, preprocess
from src.feature_engineering import create_features

st.title("⚡ Electricity Theft Detection")
 
n = st.slider("Select number of data points", 500, 5000, 2000)

 
model = joblib.load("../models/isolation_forest.pkl")

 
df = load_data("../data/raw/Household Power Consumption.csv")
df = preprocess(df)
df = create_features(df)

 
df["tampered_power"] = df["Global_active_power"]

feature_cols = [
    "tampered_power",
    "rolling_mean_10",
    "rolling_std_10",
    "lag_1",
    "hour",
    "is_weekend"
]

 
preds = model.predict(df[feature_cols])
df["anomaly"] = [1 if x == -1 else 0 for x in preds]

st.write("### Sample Data")
st.dataframe(df.head())

 
st.write("### Anomaly Detection Plot")

fig, ax = plt.subplots()

ax.plot(df["Global_active_power"][:n], label="Power")

anomaly_idx = df[df["anomaly"] == 1].index
anomaly_idx = anomaly_idx[anomaly_idx < n]

ax.scatter(
    anomaly_idx,
    df.loc[anomaly_idx, "Global_active_power"],
    color="red",
    s=10,
    label="Anomaly"
)

ax.legend()
st.pyplot(fig)