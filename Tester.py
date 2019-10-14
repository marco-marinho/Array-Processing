import SignalGenerator as sig
import matplotlib.pyplot as plt
import SteeringGenerator as str
import DOAEstimators as est
import numpy as np
import AlgebraLibrary as alg

A = str.generate_ula_vectors([-45, 60], 8)
S = sig.gen_signal(2, 100, 15)

X = np.matmul(A, S)

print(est.ESPRIT(X, 2))



