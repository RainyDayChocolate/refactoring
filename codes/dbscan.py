from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import numpy as np

def dbscan(x):
    a=[]
    outlier_detection = DBSCAN(min_samples=2, eps=0.5)
    clusters = outlier_detection.fit_predict(x)
    #clusters = [i+1.1 for i in clusters]
    #clusters=[1-i/10 for i in clusters]
    for i in clusters:
        if i < 0 :
            a.append(0.9)
        else:
            a.append(0.7)


    return np.array(a)