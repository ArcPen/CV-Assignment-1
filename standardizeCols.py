import numpy as np

def standardizeCols(M: np.ndarray, mu: float=None, sigma2: float=None):
    # Make each column of M be zero mean, std 1.
    # If mu, sigma2 are omitted, they are computed from M
    nrows, ncols = M.shape

    if (mu is None and sigma2 is None):
        mu = M.mean(axis=0).reshape(1, ncols) # 1x256
        sigma2 = M.std(axis=0).reshape(1, ncols)
        sigma2[sigma2 < 1e-15] = 1

    S = M - mu.repeat(nrows, axis=0)
    if ncols > 0:
        S = S / sigma2.repeat(nrows, axis=0)

    return S, mu, sigma2


