import numpy as np
import pandas as pd

df = pd.read_csv("datasets/train.csv", index_col=0)

# female = 2, male = 1
df["Sex"] = np.where(df.Sex == 'male', 1, 2)

# Cherbourg = 1, Queenstown = 2, Southampton = 3, Unknown = 0
df["Embarked"] = np.where(df.Embarked == 'C', 1,
                 np.where(df.Embarked == 'Q', 2,
                 np.where(df.Embarked == 'S', 3, 0)))

# Unknown Age=0
df["Age"] = df.Age.fillna(0)

df = df.drop(columns=['Name', 'Ticket', 'Cabin'])

df.to_csv("datasets/features.csv")