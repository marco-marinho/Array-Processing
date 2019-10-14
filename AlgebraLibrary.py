import numpy as np


def get_covariance(signal):

    return np.matmul(signal, signal.conj().T) / np.shape(signal)[1]