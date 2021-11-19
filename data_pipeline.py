from __future__ import print_function
from numpy import random
from numpy.core.fromnumeric import shape
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout, Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy.random import shuffle
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.engine import training
from sklearn.model_selection import KFold
import tensorflow
# import data
data = pd.read_csv("spotify_data.csv")
data_classes=["Classical","Jazz","Rock","Techno"]
data = data.to_numpy()
# Shuffle data
shuffle(data)
X=data[:,1:]
y=data[:,0]

# Scale data
MinMaxScaler().fit_transform(X)


print(shape(y))
print(shape(X))
# Create test and training sets
    # 500 training samples, 74 test samples
X_train,X_test,y_train,y_test = X[:500,:],X[500:,:],y[:500],y[500:]


class MCDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)



# train and evaluate models

kFold = KFold(n_splits=5)

fold_n=1
for train,test in kFold.split(X_train,y_train):
    no_layers = 100
    model =  keras.Sequential()
    model.optimizer=keras.optimizers.SGD(learning_rate=0.01)
    for i in range(no_layers):
        model.add(Dropout(.1))
        model.add(Dense(200, activation="elu", kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l1_l2(0.01)))
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01,momentum=0.8,),
    loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    #Why is loss: nan in some cases?!?!
    model.fit(X_train[train],y_train[train],epochs=100,validation_data=(X_train[test],y_train[test]))


    #do monte carlo dropout
    # with keras.backend.learning_phase_scope(1):
    #     for X,y in X_train[test],y_train[test]:
    #         np.stack(model.predict(X))
    #calculate metrics
    # loss,metrics,metrics1 = model.evaluate(x=X_train[test],y=y_train[test])

    # print("For fold number ",fold_n,":")
    # print("Loss:")
    # print(loss)
    # print("Metrics:")
    # print(metrics)
    fold_n=fold_n+1



