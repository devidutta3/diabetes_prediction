import pandas as pd

def statics_review(df):
    print("The Info Of The DataSets Are:\n")
    df.info()
    print("The Stastical Review Of The DataSets Are:\n")
    print(df.describe())


 

def check_zero_values(df):
    cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

    print("\nZero Value Analysis:\n")

    for col in cols:
        print(f"{col}: {(df[col] == 0).sum()}")

df=pd.read_csv(r"data\diabetes.csv")
statics_review(df) 
check_zero_values(df)
print(df["Outcome"].value_counts())
print("\n"+"---"*30)
print(df.corr()["Outcome"].sort_values(ascending=False))