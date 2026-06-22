import numpy as np
import pandas as pd
def data_clean(df):
    cols=[
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]
    df[cols]=df[cols].replace(0,np.nan)
    for col in cols:
        df[col]=df[col].fillna(df[col].median())

    return df

df=pd.read_csv(r"data\diabetes.csv")
print(data_clean(df))
print("Check If any Colums Is Zero \n")
print(df.isnull().sum())
df.to_csv(r"data/clean_dataset.csv")
print("Dataset created Successfully ❤️ ")