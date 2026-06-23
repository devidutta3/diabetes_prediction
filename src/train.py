import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
# Load cleaned dataset
df = pd.read_csv(r"data\clean_dataset.csv")

# Features (Inputs)
X = df.drop("Outcome", axis=1)

# Target (Output)
y = df["Outcome"]

# Split dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Before Scaling")
print("X_train Shape:", X_train.shape)
print("X_test Shape :", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape :", y_test.shape)

print("\n" + "---" * 30)

# Create Scaler Object
scaler = StandardScaler()

# Learn mean/std from training data and scale it
X_train = scaler.fit_transform(X_train)

# Use same mean/std on testing data
X_test = scaler.transform(X_test)

print("After Scaling")
print("X_train Shape:", X_train.shape)
print("X_test Shape :", X_test.shape)

print("\nFirst 5 Rows of Scaled X_train:\n")
print(X_train[:5])

model=LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
joblib.dump(model,r"models/model.pkl")
joblib.dump(scaler, r"models/scaler.pkl")
print("The Model Save Successfully ✅")