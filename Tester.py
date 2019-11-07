import SignalGenerator as sig
import matplotlib.pyplot as plt
import SteeringGenerator as str
import DOAEstimators as est
import numpy as np
import AlgebraLibrary as alg

A = str.generate_ula_vectors([-45, 60], 8, 1/2)
S = sig.gen_signal(2, 100, 15)

X = np.matmul(A, S)

angular_power, angles = est.conventional_beamformer(X, 2, 1/2)

plt.plot(angles, angular_power)
plt.show()



