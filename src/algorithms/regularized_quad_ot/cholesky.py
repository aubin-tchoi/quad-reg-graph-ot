import numpy as np


def cholesky(R: np.ndarray, x: np.ndarray, sign: bool) -> np.ndarray:
    p = np.size(x)
    x_transpose = x.T.copy()
    factor = 2 * int(sign) - 1
    for k in range(p):
        r = np.sqrt(R[k, k] ** 2 + factor * x_transpose[k] ** 2)
        c = r / R[k, k]
        s = x_transpose[k] / R[k, k]
        R[k, k] = r
        R[k, k + 1 : p] = (R[k, k + 1 : p] + factor * s * x_transpose[k + 1 : p]) / c
        x_transpose[k + 1 : p] = c * x_transpose[k + 1 : p] - s * R[k, k + 1 : p]
    return R


def cholupdate(R: np.ndarray, x: np.ndarray) -> np.ndarray:
    return cholesky(R, x, True)


def choldowndate(R: np.ndarray, x: np.ndarray) -> np.ndarray:
    return cholesky(R, x, False)
