import joblib
import pandas as pd

test_data = pd.read_csv('datasets/features_test_age_estimated.csv', index_col=0)
test_data = test_data.fillna(0)
model = joblib.load('models/svm_survivor_classification.pkl')

survived_prediction = model.predict(test_data)
test_data['Survived'] = survived_prediction

test_data[['Survived']].to_csv('datasets/submission_01.csv')