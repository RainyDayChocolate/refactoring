from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import numpy as np

def dbscan(x):

    outlier_detection = DBSCAN(min_samples=2, eps=3)
    clusters = outlier_detection.fit_predict(x)

    return clusters