import numpy as np
from ._algebra import get_covariance


def _get_lambdas_K_N(X: np.ndarray) -> (np.ndarray, int, int):
    R = get_covariance(X)
    K = X.shape[1]
    lambdas, _ = np.linalg.eig(R)
    lambdas = np.abs(lambdas)
    lambdas[::-1].sort()
    N = len(lambdas)
    return lambdas, K, N


def _est_model_order(X: np.ndarray, method: str, FBA: bool) -> int:
    lambdas, K, N = _get_lambdas_K_N(X)
    res = np.zeros(N)
    for i in range(N - 1):
        d = i + 1
        lsum = np.sum(lambdas[d:])
        ld = K * (N - d) * np.log(((1 / (N - d)) * lsum) / (lsum ** (1 / (N - d))))
        if method == "AIC":
            if not FBA:
                res[i] = ld + (d * (2 * N - d))
            else:
                res[i] = ld + (1 / 2) * (d * (2 * N - d + 1))
        elif method == "MDL":
            if not FBA:
                res[i] = ld + (1 / 2) * (d * (2 * N - d) + 1) * np.log(K)
            else:
                res[i] = ld + (1 / 4) * (d * (2 * N - d) + 1) * np.log(K)
        else:
            raise Exception("Invalid model order estimator, currently only AIC and MDL are supported")
    return np.argmin(res) + 1


def AIC(X: np.ndarray, FBA: bool = False) -> int:
    return _est_model_order(X, "AIC", FBA)


def MDL(X: np.ndarray, FBA: bool = False) -> int:
    return _est_model_order(X, "MDL", FBA)
