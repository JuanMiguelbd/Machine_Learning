# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:05:39 2017

@author: 106380
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV

######################################################
# CONFIGURAR
######################################################

NAME_OF_GROUP = 'THEBEASTS' # NO USAR ESPACIOS
NAME_OF_MODEL = 'RFOREST' # NO USAR ESPACIOS
fromaddr = "rauljmigueltareas@gmail.com"
passwd = "rauljmiguel"

######################################################
# PARTE DEL MODELO
######################################################

X_train = np.load(open('X_train.npy','rb'))
Y_train = np.load(open('Y_train.npy','rb'))
X_test = np.load(open('X_test.npy','rb'))
Y_test = np.load(open('Y_test.npy','rb'))

X_train_main_features =np.delete(X_train,[0,3,5,6,8],axis = 1) 
X_test_main_features =np.delete(X_test,[0,3,5,6,8],axis = 1) 

print(X_train_main_features.shape)
print(X_test_main_features.shape)

X_train=X_train_main_features
X_test=X_test_main_features

#scaler = StandardScaler()
#X_train_scaler = scaler.fit_transform(X_train)
#X_test_scaler = scaler.fit_transform(X_test)



# We create a instance of model.     
Estimator_RForest = RandomForestClassifier()

# Now, we are going to use a grid search cross-validation to explore combinations of parameters.
param_grid = {'n_estimators': [85,100,120],
              'criterion':['gini'],
              'max_features':['auto'],
              'min_samples_split':[2,3,4],
              'max_depth': range(13,15),
              'min_samples_leaf':[1,2,3]}


Grid_RForest= GridSearchCV(Estimator_RForest,param_grid,cv=10,scoring='f1', verbose=2)
Grid_RForest.fit(X_train,Y_train)

# Once it has been fitted, we get several parameters.
print("ParameterGrid: ",'\n',list(ParameterGrid(param_grid)),'\n')
print("Best estimator: " , Grid_RForest.best_estimator_,'\n')
print("Best Score: ",round(Grid_RForest.best_score_,2))
print("Best Parameters ",Grid_RForest.best_params_)


# Now, we came back fit it Best_Grid_estimator with.

Best_Grid_estimator_SVC = Grid_RForest.best_estimator_
Best_Grid_estimator_SVC.fit(X_train,Y_train)

# We use best_estimator attribute and predict method to predict test data.

Y_pred = Best_Grid_estimator_SVC.predict(X_test)

# and finally, the score test
print("f1_score:",round(f1_score(Y_test, Y_pred),2))



