import pandas as pd
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import functions.graphs as g
import joblib

start = time.time() 

data = pd.read_csv('datasets/features_age_estimated.csv', index_col=0)
features = data.drop(columns=['Survived'])
labels = np.ravel(data[['Survived']])

# Optimun SVM parameters
# C = 77.7867, gamma = 0.01189

svc_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('regressor', svm.SVC(C = 77.7867, gamma = 0.01189))
])

svc_pipeline.fit(features, labels)

joblib.dump(svc_pipeline, 'models/svm_survivor_classification.pkl', compress=9)


### Parameter Optimization ###

# # C_range = np.logspace(0, 2, 10)
# # gamma_range = np.logspace(-4, -2, 6)
# C_range = np.linspace(0.01, 700, 10)
# gamma_range = np.linspace(0.001, 0.05, 10)
# print(C_range, gamma_range)
# param_grid = [dict(regressor__gamma=gamma_range, regressor__C=C_range)]
# cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
# grid = GridSearchCV(svc_pipeline, param_grid=param_grid, 
#                     cv=cv, verbose=10)
# grid.fit(features, labels)
# #scoring='neg_root_mean_squared_error'

# results = grid.cv_results_
# C_range = results['param_regressor__C']
# gamma_range = results['param_regressor__gamma']
# mean_test_score = results['mean_test_score']

# result_df = pd.DataFrame(data=[C_range, gamma_range, mean_test_score])
# result_df = result_df.transpose()
# result_df.columns = ['C', 'gamma', 'RMSE']

# result_df.to_csv('results/survive_classification/SVC_01.csv')

# heatmap = g.heatmap(
#     x=C_range, 
#     y=gamma_range, 
#     z=mean_test_score,
#     title='RMSE x gamma x C',
#     xaxis_title='C',
#     yaxis_title='gamma')
# heatmap.show()


# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

end = time.time()
print("SCRIPT TIME: {}".format(end - start))