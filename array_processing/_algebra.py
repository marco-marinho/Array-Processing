import numpy as np


def covariance(signal):

    return np.matmul(signal, signal.conj().T) / np.shape(signal)[1]


def subspaces(signal, model_order):

    R = covariance(signal)
    w, v = np.linalg.eig(R)

    idx = np.absolute(w).argsort()[::-1]
    eigenValues = w[idx]
    eigenVectors = v[:, idx]

    Qs = eigenVectors[:, 0:model_order]
    Qn = eigenVectors[:, model_order:]

    return Qs, Qn

