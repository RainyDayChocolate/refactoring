import numpy as np
import scipy.io
from numpy.random import choice, laplace
from scipy.special import lambertw
from scipy.stats import laplace
from codes.kse_test import kse_test


def get_dpod():
    dataset4 = scipy.io.loadmat('4.mat')['dataset4']
    sizeofD = 100
    dataset2 = dataset4[:, :3]
    S, _ = np.histogram(dataset2, bins=sizeofD)
    eps = np.linspace(0.1, 2, 4)
    B = S.reshape(-1, 1)
    C = B[:sizeofD]
    Output = kse_test(C)

    pp = 1. / (Output ** 5)
    pp = pp.reshape(-1)
    pp /= pp.sum()
    NN = 2 ** (15 * Output)
    px = np.arange(1, sizeofD+1)

    gamma = 0.5
    rho = np.exp(lambertw(-gamma / (2 * np.exp(0.5)), k=-1) + 0.5)
    rho = np.real(rho)
    m = np.log(1 / rho) / (2 * (gamma - rho) ** 2)
    k = m * (1 - gamma + rho + np.sqrt(np.log(1 / rho) / (2 * m)))
    m = int(float(np.ceil(m)))
    k = int(float(np.ceil(k)))

    EVAL = np.zeros((4, 7))
    EVAL1 = np.zeros((4, 7))

    for rr in range(4):
        for j in range(1):
            C = B[:sizeofD].reshape(-1)
            GS = np.sort(np.random.choice(C, size=m, replace=False))
            Sensunif = GS[k-1]
            SS = np.sort(np.random.choice(px, size=m, replace=True, p=pp))
            SS = NN[np.floor(SS).astype(int)]
            SS = np.sort(SS)
            Sensanom = SS[k-1]

            unif = laplace.rvs(loc=0, scale=Sensunif/(eps[rr]), size=sizeofD) + C
            noiseanom = laplace.rvs(loc=0, scale=Sensanom/(eps[rr]), size=sizeofD) + NN

            noiseunif = kse_test(unif)
            noiseanom = kse_test(noiseanom)

            Index = np.zeros((sizeofD, 2))
            Index[:, 0] = np.arange(1, sizeofD+1)

            for i in range(sizeofD):
                if Output[i] > 0.7:
                    Index[i, 1] = 1

            ACTUAL = Index
            Origindex = np.where(ACTUAL[:, 1] == 1)[0]

            Index = np.zeros((sizeofD, 2))
            Index[:, 0] = np.arange(1, sizeofD+1)

            for i in range(sizeofD):
                if noiseunif[i] > 0.7:
                    Index[i, 1] = 1

            Uniformm = Index
            Pertindex = np.where(Uniformm[:, 1] == 1)[0]

            Index = np.zeros((sizeofD, 2))
            Index[:, 0] = np.arange(1, sizeofD+1)

            for i in range(sizeofD):
                if noiseanom[i] > 0.7:
                    Index[i, 1] = 1

            Anomm = Index
            Pertindex1 = np.where(Anomm[:, 1] == 1)[0]
            IND1 = Anomm
            IND = Uniformm

            C = len(set(Origindex) | set(Pertindex))
            p = len(Origindex)
            n = sizeofD - p
            N = p + n
            tp = sum(ACTUAL[Origindex, 1] == IND[Origindex, 1])
            tn = N - C
            fp = n - tn
            fn = p - tp
            tp_rate = tp / p
            tn_rate = tn / n
            accuracy = (tp + tn) / N
            sensitivity = tp_rate
            specificity = tn_rate
            precision = tp / (tp + fp)
            recall = sensitivity
            f_measure = 2 * ((precision * recall) / (precision + recall))
            gmean = np.sqrt(tp_rate * tn)
            EVAL1[rr, :] += np.array([accuracy, tp, specificity, precision, recall, f_measure, gmean])
        EVAL[rr, :] = EVAL[rr, :] / j
        EVAL1[rr, :] = EVAL1[rr, :] / j

    return EVAL, EVAL1
