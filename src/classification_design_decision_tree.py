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
from sklearn import tree

start = time.time() 

data = pd.read_csv('datasets/features_age_estimated.csv', index_col=0)
features = data.drop(columns=['Survived'])
labels = np.ravel(data[['Survived']])

# Optimun SVM parameters
# C = 77.7867, gamma = 0.01189

dt_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('regressor', tree.DecisionTreeClassifier())
])

# svc_pipeline.fit(features, labels)

# joblib.dump(svc_pipeline, 'models/svm_survivor_classification.pkl', compress=9)


## Parameter Optimization ###

random_state = np.arange(1, 10)
max_depth = np.arange(1, 10)

param_grid = [dict(regressor__random_state=random_state, regressor__max_depth=max_depth)]
cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
grid = GridSearchCV(dt_pipeline, param_grid=param_grid, 
                    cv=cv, verbose=10)
grid.fit(features, labels)
#scoring='neg_root_mean_squared_error'

results = grid.cv_results_
random_state = results['param_regressor__random_state']
max_depth = results['param_regressor__max_depth']
mean_test_score = results['mean_test_score']

result_df = pd.DataFrame(data=[random_state, max_depth, mean_test_score])
result_df = result_df.transpose()
result_df.columns = ['random_state', 'max_depth', 'RMSE']

result_df.to_csv('results/survive_classification/decistion_tree_01.csv')

heatmap = g.heatmap(
    x=random_state, 
    y=max_depth, 
    z=mean_test_score,
    title='RMSE x gamma x C',
    xaxis_title='random_state',
    yaxis_title='max_depth')
heatmap.show()


print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

end = time.time()
print("SCRIPT TIME: {}".format(end - start))