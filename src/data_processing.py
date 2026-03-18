import pandas as pd

def load_data(path):
    df=pd.read_csv(path)
    return df

def preprocess(df):

    # convert numeric columns first
    numeric_cols = df.columns.drop(['Date','Time'])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # create datetime column
    df['datetime'] = pd.to_datetime(
    df['Date'] + ' ' + df['Time'],
    format="%m/%d/%Y %I:%M:%S %p",
    errors="coerce")

    # drop missing rows
    df = df.dropna()

    # sort
    df = df.sort_values('datetime').reset_index(drop=True)

    return df
