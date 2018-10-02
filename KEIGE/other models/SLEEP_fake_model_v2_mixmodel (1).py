# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 18:10:23 2018

@author: Juan Miguel
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn import preprocessing
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedKFold


import time


start_computing_time = time.time()

# fix random seed for reproducibility
seed = 0
np.random.seed(seed)

# load your data
X_train = np.load(open('X_train.npy','rb'))
Y_train = np.load(open('Y_train.npy','rb'))
X_test = np.load(open('X_test.npy','rb'))
Y_test = np.load(open('Y_test.npy','rb'))


le=LabelEncoder()
Y_train=le.fit_transform(Y_train)
Y_test=le.fit_transform(Y_test)

Y_train_cat = to_categorical(Y_train)                          
Y_test_cat = to_categorical(Y_test)


# create a function that returns a model, taking as parameters things you
# want to verify using cross-valdiation and model selection
def create_model(optimizer='adam',
                 kernel_initializer='glorot_uniform'):
    model = Sequential()
    model.add(Dense(50, input_dim=10))
    model.add(Activation('relu'))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])

    return model

# wrap the model using the function you created

clf = KerasClassifier(build_fn=create_model, epochs=100, batch_size=128, verbose=2)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# create parameter grid, as usual, but note that you can
# vary other model parameters such as 'epochs' (and others 
# such as 'batch_size' too)
param_grid = {
    'clf__optimizer':['adam'],
    'clf__epochs':[4,8],
    'clf__kernel_initializer':['glorot_uniform','normal','uniform']
}

scaler = StandardScaler()

pipeline = Pipeline([
    ('preprocess',scaler),
    ('clf',clf)
])

# if you're not using a GPU, you can set n_jobs to something other than 1
grid = GridSearchCV(pipeline, cv=kfold, param_grid=param_grid)
grid.fit(X_train, Y_train)

Best_Grid_estimator_ = grid.best_estimator_
Best_Grid_estimator_.fit(X_train,Y_train)

# We use best_estimator attribute and predict method to predict test data.

Y_pred = Best_Grid_estimator_.predict(X_test)

# summarize results
print("f1_score:",f1_score(Y_test, Y_pred))

total_computing_time = time.time() - start_computing_time
print("Computing time: ", str(total_computing_time))

