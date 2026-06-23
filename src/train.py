import pandas as pd
from sklearn.model_selection import train_test_split as tt

#Load The Clean DataSet
df=pd.read_csv(r"data\clean_dataset.csv")
print("Data Set Loaded Successfully:\n")
print(df.head(10))
X=df.drop("Outcome",axis=1)
y=df["Outcome"]
def train_split():
    X_train,X_test,y_train,y_test=tt(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    return X_train.shape,X_test.shape,y_train.shape,y_test.shape

result=train_split()
print(result,end="\n")