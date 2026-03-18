import pandas as pd

def load_data(path):
    df=pd.read_csv(path)
    return df

def preprocess(df):
    #create datetime column
    df['datetime']= pd.to_datetime(df['Date']+' '+df['Time'])

    #convert power column to numeric
    df['Global_active_power']=pd.to_numeric(df['Global_active_power'], errors='coerce')

    #drop rows with missing values
    df=df.dropna()

    #sort by time
    df=df.sort_values('datetime')

    return df
