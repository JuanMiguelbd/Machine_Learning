# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:05:39 2017

@author: 106380
"""

import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import matplotlib.pyplot as plt

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

from sklearn.metrics import f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid

# ejemplo!!!
# We create a instance of model.     
Estimator_DTree = DecisionTreeClassifier()

# Now, we are going to use a grid search cross-validation to explore combinations of parameters.
param_grid = {'criterion':['gini'],
              'max_features':['auto'],
              'splitter':['random','best'],
              'min_samples_split':[25,30,35,40,45], 
              'max_depth': range(4,6),
              'random_state':[0]}

Grid_DTree= GridSearchCV(Estimator_DTree,param_grid,cv=10, verbose=2,scoring='f1')
Grid_DTree.fit(X_train,Y_train)

# Once it has been fitted, we get several parameters.

#print("ParameterGrid: ",'\n',list(ParameterGrid(param_grid)),'\n')
print("Best estimator: " , Grid_DTree.best_estimator_,'\n')
print("Best Score: ",round(Grid_DTree.best_score_,2))

print("Best Parameters ",Grid_DTree.best_params_)


# Now, we came back fit it Best_Grid_estimator with.

Best_Grid_estimator_DTree = Grid_DTree.best_estimator_
Best_Grid_estimator_DTree.fit(X_train,Y_train)

# We use best_estimator attribute and predict method to predict test data.

Y_pred = Best_Grid_estimator_DTree.predict(X_test)

# and finally, the score test
print("f1_score:",round(f1_score(Y_test, Y_pred),2))

# Examinando la importancia de las features
features=["0","1","2","3","4","5","6","7","8","9"]
caract=X_train.shape[1]
plt.figure(figsize=(5,10))
plt.barh(range(caract),Best_Grid_estimator_DTree.feature_importances_)
plt.yticks(np.arange(caract),features)

plt.xlabel('Importancia de las features')
plt.ylabel('Features')

mediana = np.median(Best_Grid_estimator_DTree.feature_importances_)
values = Best_Grid_estimator_DTree.feature_importances_

print(mediana)

list = []
for i in range(caract):
    if values[i]>mediana:
        list.append(i)

print("The main Features: ",list)

features_main = np.asarray(list)

print(features_main)

print(X_train.shape)

X_train_main_features =np.delete(X_train,[0,3,5,6,8],axis = 1) 
X_test_main_features =np.delete(X_test,[0,3,5,6,8],axis = 1) 

print(X_train_main_features.shape)
print(X_test_main_features.shape)


#-------------------------------------------------------------------------------------------

# We create a instance of model.
        
Estimator_SVC =  SVC () 

# Now, we are going to use a grid search cross-validation to explore combinations of parameters.

param_grid = {'C':[1, 5, 10, 50], 'gamma':[0.0001, 0.0005, 0.001, 0.005], 'kernel':['linear', 'rbf', 'sigmoid']}




Grid_S_CV= GridSearchCV(Estimator_SVC,param_grid, verbose=1, scoring='f1', cv=10)
Grid_S_CV.fit(X_train_main_features,Y_train)

# Once it has been fitted, we get several parameters.


# Now, we came back fit it Best_Grid_estimator with.

Best_Grid_estimator_SVC = Grid_S_CV.best_estimator_
Best_Grid_estimator_SVC.fit(X_train_main_features,Y_train)

# We use best_estimator attribute and predict method to predict test data.

y_pred = Best_Grid_estimator_SVC.predict(X_test_main_features)

print("ParameterGrid: ",'\n',list(ParameterGrid(param_grid)),'\n')
print("Best estimator: " , Grid_S_CV.best_estimator_,'\n')
print("Best Score: ",Grid_S_CV.best_score_)
print("Best Parameters ",Grid_S_CV.best_params_)
print("Classes ",Grid_S_CV.classes_,'\n')

print("f1_score:",f1_score(Y_test, y_pred))


