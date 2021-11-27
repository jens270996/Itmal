from __future__ import print_function
from numpy import random
from numpy.core.fromnumeric import shape
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from tensorflow.keras.layers import Dropout, Dense, GaussianDropout, Activation, BatchNormalization
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy.random import shuffle
from sklearn.model_selection import KFold
from functools import partial
from tensorflow.python.keras.backend import batch_normalization, dropout
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



# train and evaluate models\
kFold = KFold(n_splits=5, shuffle=True, random_state=42)

fold_n=1
for train,test in kFold.split(X_train,y_train):
    no_layers = 50
    neurons_pr_layer = 200
    RegularizedDense = partial(Dense, activation='relu',
                                    kernel_initializer='he_normal', 
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001))
    model =  keras.Sequential() 

    model.add(Dropout(0.2))
    model.add(RegularizedDense(4*300))
    model.add(Dropout(0.2))
    model.add(RegularizedDense(4*200))
    model.add(Dropout(0.1))
    model.add(RegularizedDense(4*50))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(RegularizedDense(4*25))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(RegularizedDense(4*10))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(4, activation='softmax'))

    opt =keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt,
    loss="sparse_categorical_crossentropy",metrics=['accuracy'])
    #Why is loss: nan in some cases?!?!
    hist = model.fit(X_train[train],y_train[train],epochs=100, validation_data=(X_train[test],y_train[test]))


    #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.SparseTopKCategoricalAccuracy()])
    #do monte carlo dropout
    # with keras.backend.learning_phase_scope(1):
    #     for X,y in X_train[test],y_train[test]:
    #         np.stack(model.predict(X))
    #calculate metrics
    # loss,metrics,metrics1 
    # = model.evaluate(x=X_train[test],y=y_train[test])

    # print("For fold number ",fold_n,":")
    # print("Loss:")
    # print(loss)
    # print("Metrics:")
    # print(metrics)
    fold_n=fold_n+1
    print(max(hist.history['accuracy']))