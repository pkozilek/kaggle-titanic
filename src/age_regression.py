import joblib
import pandas as pd

#data = pd.read_csv("datasets/features.csv", index_col=0)
data = pd.read_csv("datasets/features_test.csv", index_col=0)
age_regression_model = joblib.load('models/age_regressor_02.pkl')

data_ai = data.loc[data.Age == 0] # Age incomplete
#features_ai = data_ai.drop(columns=['Survived', 'Age'])
features_ai = data_ai.drop(columns=['Age'])

age_predictions = age_regression_model.predict(features_ai)
features_ai['Age'] = age_predictions

data = data.merge(features_ai[['Age']], left_index=True, right_index=True, how='left')
data['Age'] = data.Age_y.fillna(data.Age_x)
data = data.drop(columns=['Age_x', 'Age_y'])

#data.to_csv("datasets/features_age_estimated.csv")
data.to_csv("datasets/features_test_age_estimated.csv")
