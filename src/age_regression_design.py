import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import functions.graphs as g
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import math
import plotly.express as px
import time
import joblib

start = time.time() 

data = pd.read_csv('datasets/features.csv', index_col=0)
data_ac = data.loc[data.Age != 0] # Age completed

labels = np.ravel(data_ac[['Age']])
features = data_ac.drop(columns=['Survived', 'Age'])
features = features.values.tolist()

### Optimized Parameters ###
C = 1000000
gamma = 3.9810717055349695e-05
# R2 = 0.23, RMSE = 12.29

svr_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('regressor', svm.SVR(C=C, gamma=gamma))
])

svr_pipeline.fit(features, labels)

joblib.dump(svr_pipeline, 'models/age_regressor.pkl', compress=9)

end = time.time()

### Parameter Optimization ###

# C_range = np.logspace(-3, 6, 6)
# gamma_range = np.logspace(-6, -2, 6)
# print(C_range, gamma_range)
# param_grid = [dict(regressor__gamma=gamma_range, regressor__C=C_range)]
# cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
# grid = GridSearchCV(svr_pipeline, param_grid=param_grid, 
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

# result_df.to_csv('results/age_regression/SVR_01.csv')

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

print("SCRIPT TIME: {}".format(end - start))