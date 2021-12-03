import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy.random import shuffle
from numpy.core.fromnumeric import shape
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
data = pd.read_csv("spotify_data.csv")

data_classes=["Classical","Jazz","Rock","Techno"]

#rock_data = data.where(data["genre"] == 3).dropna().reset_index(drop=True) #red

#print(rock_data)
data = data.to_numpy()
# Shuffle data
shuffle(data)
X=data[:,1:]
y=data[:,0]
# Scale data
MinMaxScaler().fit_transform(X)
# Create test and training sets
    # 500 training samples, 74 test samples
X_train,X_test,y_train,y_test = X[:3600,:],X[3600:,:],y[:3600],y[3600:]
sgd_clf = SGDClassifier(random_state=42)
y_train_3 = (y_train == 2)
y_test_3 = (y_test == 2)
print(np.unique(y_train))
print(shape(X_train))
print(shape(y_train_3))
print(shape(y_train))
#print(y_train)
sgd_clf.fit(X_train, y_train_3)
print(f'{sgd_clf.predict([X_test[0]])}')
print(f'y_test[0]={y_test_3[0]}')

print(f'{cross_val_score(sgd_clf, X_train, y_train, cv=510, scoring="accuracy")}')


