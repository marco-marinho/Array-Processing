import numpy as np
from ._algebra import covariance


def _get_lambdas_K_N(X: np.ndarray) -> (np.ndarray, int, int):
    R = covariance(X)
    K = X.shape[1]
    lambdas, _ = np.linalg.eig(R)
    lambdas = np.abs(lambdas)
    lambdas[::-1].sort()
    N = len(lambdas)
    return lambdas, K, N


def __geo_mean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1.0 / len(a))


def _est_model_order(X: np.ndarray, method: str, FBA: bool = False) -> int:
    lambdas, K, N = _get_lambdas_K_N(X)
    res = np.zeros(N)
    for d in range(N):
        amean = np.mean(lambdas[d:])
        gmean = __geo_mean(lambdas[d:])
        if method == "AIC":
            if not FBA:
                res[d] = K * (N - d) * np.log(amean / gmean) + d * (2 * N - d)
            else:
                res[d] = K * (N - d) * np.log(amean / gmean) + 0.5 * d * (2 * N - d + 1)
        elif method == "MDL":
            if not FBA:
                res[d] = K * (N - d) * np.log(amean / gmean) + 0.5 * (d * (2 * N - d) + 1) * np.log(K)
            else:
                res[d] = K * (N - d) * np.log(amean / gmean) + 0.25 * d * (2 * N - d + 1) * np.log(K)
        else:
            raise Exception("Invalid model order estimator, currently only AIC and MDL are supported")
    return int(np.argmin(res))


def AIC(X: np.ndarray, FBA: bool = False) -> int:
    return _est_model_order(X, "AIC", FBA)


def MDL(X: np.ndarray, FBA: bool = False) -> int:
    return _est_model_order(X, "MDL", FBA)
