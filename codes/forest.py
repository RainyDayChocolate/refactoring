from sklearn.ensemble import IsolationForest
from sklearn import svm
import numpy as np

def forest(x):
    a=[]
    clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma='auto')
    clf.fit(x)
    preds = clf.fit_predict(x)
    for i in preds:
        if i < 0 :
            a.append(0.9)

        else:
            a.append(0.7)

    return np.array(a)
