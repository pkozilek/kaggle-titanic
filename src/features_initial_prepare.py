import numpy as np
import pandas as pd

#df = pd.read_csv("datasets/train.csv", index_col=0)
df = pd.read_csv("datasets/test.csv", index_col=0)

df = pd.concat([df, pd.get_dummies(df.Sex)], axis=1)
df = pd.concat([df, pd.get_dummies(df.Embarked)], axis=1)

# Unknown Age=0
df["Age"] = df.Age.fillna(0)

df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked'])

#df.to_csv("datasets/features.csv")
df.to_csv("datasets/features_test.csv")