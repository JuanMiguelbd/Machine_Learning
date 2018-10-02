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


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt



X_train = np.load(open('X_train.npy','rb'))
Y_train = np.load(open('Y_train.npy','rb'))
X_test = np.load(open('X_test.npy','rb'))
Y_test = np.load(open('Y_test.npy','rb'))

print(X_train.shape)
print(X_test.shape)
le=LabelEncoder()
Y_train=le.fit_transform(Y_train)
Y_test=le.fit_transform(Y_test)

Y_train_cat = to_categorical(Y_train)                          
Y_test_cat = to_categorical(Y_test)

model = Sequential()
model.add(Dense(32, input_dim=10))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(
  X_train, # Training data
  Y_train_cat, # Labels of training data
  batch_size=32, # Batch size for the optimizer algorithm
  epochs=500, # Number of epochs to run the optimizer algorithm
  verbose=2, # Level of verbosity of the log messages
  validation_data=(X_test,Y_test_cat)
)
# No cambiar nombre de la variable!!!

preds = model.predict_classes(X_test)

print("f1_score:",f1_score(Y_test, preds))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


######################################################
# NO TOCAR
######################################################