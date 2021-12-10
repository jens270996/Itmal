from time import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy.random import shuffle
from numpy.core.fromnumeric import shape
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
data = pd.read_csv("spotify_data.csv")

data_classes=["Classical","Jazz","Rock","Techno"]

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

#rock_data = data.where(data["genre"] == 3).dropna().reset_index(drop=True) #red

#print(rock_data)
data = data.to_numpy()
# Shuffle data
np.random.seed(42)
shuffle(data)
X=data[:,1:]
y=data[:,0]
# Scale data
MinMaxScaler().fit_transform(X)
# Create test and training sets
split_index_1 = 6500
split_index_2 = 6500
X_train,X_val,X_test,y_train,y_val,y_test = X[:split_index_1,:],X[split_index_1:split_index_2,:],X[split_index_2:,:],y[:split_index_1],y[split_index_1:split_index_2],y[split_index_2:]

model = RandomForestClassifier()
tuning_parameters = {
    'n_estimators': (30, 50, 500, 1000),
    'criterion': ('gini', 'entropy'),
    'max_features': (2, 3, 4, 6, 8)
}

CV = 5
VERBOSE = 0
start = time()
random_tuned = RandomizedSearchCV(model,
                        tuning_parameters,
                        cv= CV,
                        scoring='f1_micro',
                        verbose=VERBOSE,
                        n_iter=100,
                        n_jobs=-1)

hist = random_tuned.fit(X_train, y_train)
t = time() - start
estimator = hist.best_estimator_
feature_arr = ["popularity","danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_ms","time_signature"]
print("popularity,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,time_signature")
print(estimator.feature_importances_)
print("Best score: ")
print(hist.best_score_)
print(hist.best_params_)
train_mse=mean_squared_error(y_train,random_tuned.predict(X_train))
val_mse = mean_squared_error(y_test,random_tuned.predict(X_test))
confusion_matrix_ = confusion_matrix(y_train,random_tuned.predict(X_train))
print("Confusion matrix:")
print(confusion_matrix_)
confusion_matrix_ = confusion_matrix(y_test,random_tuned.predict(X_test))
print("Confusion matrix on validation:")
print(confusion_matrix_)
print("train mse is:",train_mse,"validation mse is:",val_mse)
acc=accuracy_score(y_test,random_tuned.predict(X_test))
print("accuracy score is:",acc)
print(random_tuned.n_features_in_)
FullReport(random_tuned, X_test, y_test, t)
#print(b0)

