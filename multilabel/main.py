from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

from helpers import RBF

# data=pd.read_csv('data/breast-cancer.data')
# data1=data.apply(lambda x:x.astype('category').cat.codes)

# def split(data,percent):
    
#     np.random.seed(1)
    
#     perm=np.random.permutation(data.index)
#     train=int(len(data)*percent)
    
#     X_train=data.iloc[perm[:train],:-1]
#     X_test = data.iloc[perm[train:],:-1]
    
#     y_train = data.iloc[perm[:train],-1]
#     y_test =data.iloc[perm[train:],-1]
    
#     return X_train , X_test , y_train,y_test
# X_train , X_test , y_train, y_test =split(data1,0.8)

# cols = X_train.columns

X = np.array([[1,1], [0,0], [1,0], [0,1]])
y = np.array([1, 1, 0, 0])

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)

# X_test = scaler.transform(X_test)

# X_train = pd.DataFrame(X_train, columns=[cols])
# X_test = pd.DataFrame(X_test, columns=[cols])

print('Testing XOR')

for clf, name in [(SVC(kernel=RBF(), C=1000), 'pykernel'), (SVC(kernel='rbf', C=1000), 'sklearn')]:
    clf.fit(X, y)
    print(name)
    print(clf)
    print('Predictions:', clf.predict(X))
    print('Accuracy:', accuracy_score(clf.predict(X), y))
    print()