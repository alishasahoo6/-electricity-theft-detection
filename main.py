from src.data_processing import load_data, preprocess
from src.feature_engineering import create_features
from src.anomaly_detection import simulate_theft
from src.model import train_model, predict_anomalies

from sklearn.metrics import classification_report
import joblib

# load + preprocess
df = load_data("data/raw/Household Power Consumption.csv")
df = preprocess(df)
df = create_features(df)

# simulate theft
df = simulate_theft(df)

print("Columns after simulation:", df.columns)

# features for model
feature_cols = [
    "tampered_power",
    "rolling_mean_10",
    "rolling_std_10",
    "lag_1",
    "hour",
    "is_weekend"
]

# train model
model = train_model(df, feature_cols)

# predict anomalies
df = predict_anomalies(model, df, feature_cols)

# evaluation
print(df[["is_theft", "anomaly"]].head(20))
print(classification_report(df["is_theft"], df["anomaly"]))

# save model
joblib.dump(model, "models/isolation_forest.pkl")

import matplotlib.pyplot as plt

plt.plot(df["tampered_power"][:2000])
plt.scatter(
    df.index[:2000][df["anomaly"][:2000] == 1],
    df["tampered_power"][:2000][df["anomaly"][:2000] == 1],
    s=10
)
plt.title("Detected Anomalies")
plt.show()

df.to_csv("outputs/results.csv", index=False)