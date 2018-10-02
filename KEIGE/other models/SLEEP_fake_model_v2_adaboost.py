# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:05:39 2017

@author: 106380
"""

import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score

######################################################
# PARTE DEL MODELO
######################################################

X_train = np.load(open('X_train.npy','rb'))
Y_train = np.load(open('Y_train.npy','rb'))
X_test = np.load(open('X_test.npy','rb'))
Y_test = np.load(open('Y_test.npy','rb'))

#scaler = StandardScaler()
#X_train_scaler = scaler.fit_transform(X_train)
#X_test_scaler = scaler.fit_transform(X_test)

# We create a instance of model.     
Estimator_ADA = AdaBoostClassifier()

# Fit KernelRidge with parameter selection based on 10-fold cross validation

param_grid = {
'n_estimators': [400,450,500],
'learning_rate' : [0.3,0.4,0.5]}

Grid_ADA= GridSearchCV(Estimator_ADA,param_grid,cv=10,scoring='f1', verbose=2)
Grid_ADA.fit(X_train,Y_train)

# Once it has been fitted, we get several parameters.

print("ParameterGrid: ",'\n',list(ParameterGrid(param_grid)),'\n')
print("Best estimator: " , Grid_ADA.best_estimator_,'\n')
print("Best Score: ",round(Grid_ADA.best_score_,2))

print("Best Parameters ",Grid_ADA.best_params_)


# Now, we came back fit it Best_Grid_estimator with.

Best_Grid_estimator_ADA = Grid_ADA.best_estimator_
Best_Grid_estimator_ADA.fit(X_train,Y_train)

# We use best_estimator attribute and predict method to predict test data.

Y_pred = Best_Grid_estimator_ADA.predict(X_test)

# and finally, the score test
print("f1_score:",round(f1_score(Y_test, Y_pred),2))

