from sklearn.ensemble import IsolationForest

def train_model(df, feature_cols):

    model = IsolationForest(
        contamination=0.01,
        random_state=42
    )

    model.fit(df[feature_cols])

    return model

def predict_anomalies(model, df, feature_cols):

    preds = model.predict(df[feature_cols])

    # convert (-1 = anomaly, 1 = normal)
    df["anomaly"] = preds
    df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

    return df