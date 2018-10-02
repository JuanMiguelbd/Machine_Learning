# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:05:39 2017

@author: 106380
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

######################################################
# CONFIGURAR
######################################################

NAME_OF_GROUP = 'THEBEASTS' # NO USAR ESPACIOS
NAME_OF_MODEL = 'KNEIGHBOURNS' # NO USAR ESPACIOS
fromaddr = "rauljmigueltareas@gmail.com"
passwd = "rauljmiguel"

######################################################
# PARTE DEL MODELO
######################################################

X_train = np.load(open('X_train.npy','rb'))
Y_train = np.load(open('Y_train.npy','rb'))
X_test = np.load(open('X_test.npy','rb'))
Y_test = np.load(open('Y_test.npy','rb'))

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.fit_transform(X_test)

Estimator = MLPClassifier()

# Now, we are going to use a grid search cross-validation to explore combinations of parameters.
param_grid = {'hidden_layer_sizes': [i for i in range(1,15)],
              'activation': ['identity', 'logistic','tanh','relu'],
              'solver': ['lbfgs', 'sgd','adam'],
              'learning_rate': ['constant', 'invscaling','adaptive'],
              'learning_rate_init': [0.001],
              'power_t': [0.5],
              'alpha': [0.0001],
              'max_iter': [1000],
              'early_stopping': [False],
              'warm_start': [False]}

Grid_MLP= GridSearchCV(Estimator,param_grid,cv=10,scoring='f1', verbose=2)
Grid_MLP.fit(X_train,Y_train)

# Once it has been fitted, we get several parameters.

print("ParameterGrid: ",'\n',list(ParameterGrid(param_grid)),'\n')
print("Best estimator: " , Grid_MLP.best_estimator_,'\n')
print("Best Score: ",round(Grid_MLP.best_score_,2))

print("Best Parameters ",Grid_MLP.best_params_)


# Now, we came back fit it Best_Grid_estimator with.

Best_Grid_estimator_SVC = Grid_MLP.best_estimator_
Best_Grid_estimator_SVC.fit(X_train_scaler,Y_train)

# We use best_estimator attribute and predict method to predict test data.

Y_pred = Best_Grid_estimator_SVC.predict(X_test_scaler)

# and finally, the score test
print("f1_score:",round(f1_score(Y_test, Y_pred),2))
print("f1_score:",round(f1_score(Y_test, Y_pred, average='weighted', labels=np.unique(Y_pred)),2))

