# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:05:39 2017

@author: 106380
"""

import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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


#X_train_main_features =np.delete(X_train,[2,3,4,5],axis = 1) 
#X_test_main_features =np.delete(X_test,[2,3,4,5],axis = 1) 

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA


import time


start_computing_time = time.time()



estimators =[("PCA",PCA()),("nmz",Normalizer()),("LSVC",LinearSVC())]

model = Pipeline(steps=estimators)



param_grid = {"LSVC__loss": ['hinge'],
              "LSVC__class_weight":['balanced']}



Grid = GridSearchCV(model, param_grid =param_grid, cv=5)
Grid.fit(X_train, Y_train) 

Best_Grid_estimator = Grid.best_estimator_
Best_Grid_estimator.fit(X_train, Y_train)

print(Best_Grid_estimator)

pred=Best_Grid_estimator.predict(X_test)

print("Accuracy of predictions:")
print(accuracy_score(Y_test, pred))

print("f1_score:",f1_score(Y_test, pred))

total_computing_time = time.time() - start_computing_time
print("Computing time: ", str(total_computing_time))




