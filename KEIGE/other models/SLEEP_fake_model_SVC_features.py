# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:05:39 2017

@author: 106380
"""

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

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

#X_train_main_features =np.delete(X_train,[0,3,5,6,8],axis = 1) 
#X_test_main_features =np.delete(X_test,[0,3,5,6,8],axis = 1) 

X_train_main_features =np.delete(X_train,[2,3,4,5],axis = 1) 
X_test_main_features =np.delete(X_test,[2,3,4,5],axis = 1)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train_main_features)
X_test_scaler = scaler.fit_transform(X_test_main_features)

# We create a instance of model.
       
Estimator_SVC =  SVC ()

# Now, we are going to use a grid search cross-validation to explore combinations of parameters.

param_grid = {'C':[250,275,300], 'gamma':[0.008,0.009,0.01], 'kernel':['rbf']}

Grid_S_CV= GridSearchCV(Estimator_SVC,param_grid,cv=10,scoring='f1', verbose=2)
Grid_S_CV.fit(X_train_scaler,Y_train)

# Once it has been fitted, we get several parameters.

print("ParameterGrid: ",'\n',list(ParameterGrid(param_grid)),'\n')
print("Best estimator: " , Grid_S_CV.best_estimator_,'\n')
print("Best Score: ",Grid_S_CV.best_score_)
print("Best Parameters ",Grid_S_CV.best_params_)
print("Classes ",Grid_S_CV.classes_,'\n')

# Now, we came back fit it Best_Grid_estimator with.

Best_Grid_estimator_SVC = Grid_S_CV.best_estimator_
Best_Grid_estimator_SVC.fit(X_train_scaler,Y_train)

# We use best_estimator attribute and predict method to predict test data.

Y_pred = Best_Grid_estimator_SVC.predict(X_test_scaler)

# and finally, the score test
print("f1_score:",round(f1_score(Y_test, Y_pred),2))

