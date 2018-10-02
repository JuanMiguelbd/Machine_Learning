# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:05:39 2017

@author: 106380
"""

import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler


X_train = np.load(open('X_train.npy','rb'))
Y_train = np.load(open('Y_train.npy','rb'))
X_test = np.load(open('X_test.npy','rb'))
Y_test = np.load(open('Y_test.npy','rb'))


scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.fit_transform(X_test)


# We create a instance of model.     
Estimator =ensemble.GradientBoostingClassifier()

# Now, we are going to use a grid search cross-validation to explore combinations of parameters.
param_grid = {'n_estimators': [10,20,30],'max_features':['auto', 'log2'],\
              'min_samples_split':[5,10,15], 'max_depth': range(2,15)}

Grid_GBoost= GridSearchCV(Estimator,param_grid,cv=10, scoring='f1', verbose=2)
Grid_GBoost.fit(X_train,Y_train)

# Once it has been fitted, we get several parameters.

print("ParameterGrid: ",'\n',list(ParameterGrid(param_grid)),'\n')
print("Best estimator: " , Grid_GBoost.best_estimator_,'\n')
print("Best Score: ",round(Grid_GBoost.best_score_,2))
print("Best Parameters ",Grid_GBoost.best_params_)
print("Classes ",Grid_GBoost.classes_,'\n')


# Now, we came back fit it Best_Grid_estimator with.

Best_Grid_estimator_GBOOST = Grid_GBoost.best_estimator_
Best_Grid_estimator_GBOOST.fit(X_train,Y_train)

# We use best_estimator attribute and predict method to predict test data.

Y_pred = Best_Grid_estimator_GBOOST.predict(X_test)

# and finally, the score test
print("f1_score:",round(f1_score(Y_test, Y_pred),2))
