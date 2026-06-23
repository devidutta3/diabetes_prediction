import numpy as np
import pandas as pd


def data_clean(df):
    cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    # Replace invalid 0 values with NaN
    df[cols] = df[cols].replace(0, np.nan)

    # Fill missing values with median
    for col in cols:
        df[col] = df[col].fillna(df[col].median())

    return df


# Load dataset
df = pd.read_csv(r"data\diabetes.csv")

# Clean dataset
df = data_clean(df)

print("Missing Values Check:\n")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv(r"data\clean_dataset.csv", index=False)

print("\nDataset created successfully ❤️")
print("\n" + "---" * 30)


def feature(df_clean):
    X = df_clean.drop("Outcome", axis=1)
    y = df_clean["Outcome"]

    return X, y


# Load cleaned dataset
df_clean = pd.read_csv(r"data\clean_dataset.csv")

# Create features and target
X, y = feature(df_clean)

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeature Columns:")
print(X.columns)