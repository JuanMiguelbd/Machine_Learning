# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:05:39 2017

@author: 106380
"""

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

######################################################
# CONFIGURAR
######################################################

X_train = np.load(open('X_train.npy','rb'))
Y_train = np.load(open('Y_train.npy','rb'))
X_test = np.load(open('X_test.npy','rb'))
Y_test = np.load(open('Y_test.npy','rb'))

# We create a instance of model.
        
Estimator =  LinearSVC ()

# Now, we are going to use a grid search cross-validation to explore combinations of parameters.

param_grid ={'C':[1.0], 'class_weight':[None], 'dual':[True], 'fit_intercept':[True],
     'intercept_scaling':[1], 'loss':['squared_hinge'], 'max_iter':[1000],
     'multi_class':['ovr'], 'penalty':['l2'], 'random_state':[0], 'tol':[0.0001],
     'verbose':[0]}

Grid_CBoost= GridSearchCV(Estimator,param_grid,cv=10, verbose=2, scoring='f1')
Grid_CBoost.fit(X_train,Y_train)

# Once it has been fitted, we get several parameters.

print("ParameterGrid: ",'\n',list(ParameterGrid(param_grid)),'\n')
print("Best estimator: " , Grid_CBoost.best_estimator_,'\n')
print("Best Score: ",round(Grid_CBoost.best_score_,2))
print("Best Parameters ",Grid_CBoost.best_params_)
print("Classes ",Grid_CBoost.classes_,'\n')


# Now, we came back fit it Best_Grid_estimator with.

Best_Grid_estimator_CBoost = Grid_CBoost.best_estimator_
Best_Grid_estimator_CBoost.fit(X_train,Y_train)

# We use best_estimator attribute and predict method to predict test data.

Y_pred = Best_Grid_estimator_CBoost.predict(X_test)

# and finally, the score test
print("f1_score:",round(f1_score(Y_test, Y_pred),2))



