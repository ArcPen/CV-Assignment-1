import numpy as np
# import cupy as np

def linearInd2Binary(ind: np.ndarray, labels: int):
    n = ind.shape[0]
    y = -1 * np.ones((n, labels))
    for i in range(n):
        y[i, ind[i]] = 1
    return y
