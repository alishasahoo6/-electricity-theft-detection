def create_features(df):
    
    """ 
    Create time-series features for anomaly detection

    """
    # Rolling mean (10-minute window)
    df['rolling_mean_10']=(df['Global_active_power'].rolling(window=10).mean())
    
    # Rolling standard deviation (10-minute window)
    df['rolling_std_10']=(df['Global_active_power'].rolling(window=10).std())

    # Hour of day
    df['hour']=df['datetime'].dt.hour

    # Day of week
    df['is_weekend']=(df['datetime'].dt.dayofweek>=5).astype(int)

    # Lag festure (previous minute power)
    df['lag_1']=df['Global_active_power'].shift(1)

    # Drop rows with NaN values created by rolling and lag features
    df=df.dropna().reset_index(drop=True)

    return df