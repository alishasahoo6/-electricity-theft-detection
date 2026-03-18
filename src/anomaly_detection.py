def simulate_theft(df, start=50000, end=60000, reduction_factor=0.5):

    df = df.copy()

    df["tampered_power"] = df["Global_active_power"]

    df.loc[start:end, "tampered_power"] *= reduction_factor

    df["is_theft"] = 0
    df.loc[start:end, "is_theft"] = 1

    return df