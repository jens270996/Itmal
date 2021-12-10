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
import numpy.random
from sklearn.model_selection import KFold
from functools import partial
from tensorflow.python.keras.backend import batch_normalization, dropout
from sklearn.metrics import classification_report
# import data
data = pd.read_csv("spotify_data.csv")
data_classes=["Classical","Jazz","Rock","Techno"]
data = data.to_numpy()
# Shuffle data
numpy.random.seed(42)
shuffle(data)
X=data[:,1:]
y=data[:,0]
# Scale data
MinMaxScaler().fit_transform(X)
#region  Output from O3 exercise
currmode="N/A" # GLOBAL var!
def SearchReport(model): 
    
    def GetBestModelCTOR(model, best_params):
        def GetParams(best_params):
            ret_str=""          
            for key in sorted(best_params):
                value = best_params[key]
                temp_str = "'" if str(type(value))=="<class 'str'>" else ""
                if len(ret_str)>0:
                    ret_str += ','
                ret_str += f'{key}={temp_str}{value}{temp_str}'  
            return ret_str          
        try:
            param_str = GetParams(best_params)
            return type(model).__name__ + '(' + param_str + ')' 
        except:
            return "N/A(1)"
        
    print("\nBest model set found on train set:")
    print()
    print(f"\tbest parameters={model.best_params_}")
    print(f"\tbest '{model.scoring}' score={model.best_score_}")
    print(f"\tbest index={model.best_index_}")
    print()
    print(f"Best estimator CTOR:")
    print(f"\t{model.best_estimator_}")
    print()
    try:
        print(f"Grid scores ('{model.scoring}') on development set:")
        means = model.cv_results_['mean_test_score']
        stds  = model.cv_results_['std_test_score']
        i=0
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
            print("\t[%2d]: %0.3f (+/-%0.03f) for %r" % (i, mean, std * 2, params))
            i += 1
    except:
        print("WARNING: the random search do not provide means/stds")
    
    global currmode                
    assert "f1_micro"==str(model.scoring), f"come on, we need to fix the scoring to be able to compare model-fits! Your scoreing={str(model.scoring)}...remember to add scoring='f1_micro' to the search"   
    return f"best: dat={currmode}, score={model.best_score_:0.5f}, model={GetBestModelCTOR(model.estimator,model.best_params_)}", model.best_estimator_ 

def ClassificationReport(model, X_test, y_test, target_names=None):
    assert X_test.shape[0]==y_test.shape[0]
    print("\nDetailed classification report:")
    print("\tThe model is trained on the full development set.")
    print("\tThe scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, model.predict(X_test)                 
    print(classification_report(y_true, y_pred, target_names))
    print()

def FullReport(model, X_test, y_test, t):
    print(f"SEARCH TIME: {t:0.2f} sec")
    beststr, bestmodel = SearchReport(model)
    ClassificationReport(model, X_test, y_test)    
    print(f"CTOR for best model: {bestmodel}\n")
    print(f"{beststr}\n")
    return beststr, bestmodel

#endregion


print(shape(y))
print(shape(X))
# Create test and training sets
split_index = 6000
    # 500 training samples, 74 test samples
X_train,X_test,y_train,y_test = X[:split_index,:],X[split_index:,:],y[:split_index],y[split_index:]

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
    hist = model.fit(X_train[train],y_train[train],epochs=2, validation_data=(X_train[test],y_train[test]))
    fold_n=fold_n+1
    print(max(hist.history['accuracy']))
