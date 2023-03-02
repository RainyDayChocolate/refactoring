from sklearn.ensemble import IsolationForest
from sklearn import svm
import numpy as np

def tree(x):
    a=[]
    clf = IsolationForest(max_samples=100, random_state = 1, contamination= 'auto')

    preds = clf.fit_predict(x)
    for i in preds:
        if i < 0 :
            a.append(0.9)

        else:
            a.append(0.7)

    return np.array(a)
