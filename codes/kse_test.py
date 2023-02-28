import numpy as np
# assuming distSample0 and distSampleTemp are already defined
# run the two-sample KS test
from scipy.stats import ks_2samp


# assuming distSample0 and distSampleTemp are already defined
# run the two-sample KS test
def get_d(distSample0, distSampleTemp):
    test_statistic, pvalue = ks_2samp(distSample0, distSampleTemp)

    # calculate the distance between the empirical distribution functions
    n0 = len(distSample0)
    nTemp = len(distSampleTemp)
    x = np.sort(np.concatenate([distSample0, distSampleTemp]))
    cdf0 = np.searchsorted(distSample0, x, side='right') / n0
    cdfTemp = np.searchsorted(distSampleTemp, x, side='right') / nTemp
    d = np.max(np.abs(cdf0 - cdfTemp))
    return d


def kse_test(x):
    # Compute the outlier score for each p-dimensional data point
    # The highest scores are possible outliers, scores between [0,1]
    # Original scoring algorithm by Michael S Kim (mikeskim@gmail.com)
    # Version 1.00 (12/22/2012) for Matlab ported from R
    # not fully tested on Matlab, tested on GNU Octave and R
    np.seterr(all='ignore')  # to ignore runtime warnings
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    nrows, dcols = x.shape
    scores = np.zeros(nrows)
    nsample = int(np.ceil(nrows * 0.95))
    if nsample > 300:
        nsample = 300
    for i in range(nrows):
        # Sample points to build dpop
        tmp1 = np.random.permutation(nrows)[:nsample]
        dpop = x[tmp1, :]
        distSample0 = np.zeros(nsample)

        # Build distances from point i to sampled points
        for j in range(nsample):
            distSample0[j] = np.sqrt(np.sum((dpop[j, :] - x[i, :]) ** 2))

        tmp2 = np.random.permutation(nrows)[:nsample]
        bpop = x[tmp2, :]

        # Build distances from bpop k to sampled points dpop
        for k in range(nsample):
            distSampleTemp = np.zeros(nsample)
            for j in range(nsample):
                distSampleTemp[j] = np.sqrt(np.sum((dpop[j, :] - bpop[k, :]) ** 2))
            # ks, pval = kstest(distSample0, distSampleTemp)
            # n1, n2 = len(distSample0), len(distSampleTemp)
            d = get_d(distSample0, distSampleTemp)
            # d = ks * np.sqrt((n1 + n2) / (n1 * n2))
            scores[i] += d / nsample

    np.seterr(all='warn')  # to enable runtime warnings
    return scores
