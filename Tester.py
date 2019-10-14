import SignalGenerator as sig
import matplotlib.pyplot as plt
import SteeringGenerator as str
import DOAEstimators as est
import numpy as np

A = str.generate_ula_vectors([-45, 60], 20 , 1/2)
S = sig.gen_signal(2, 100, 100)

X = np.matmul(A, S)

P, angles = est.conventional_beamformer(X, 0.1)
print(np.shape(P))
print(np.shape(angles))
plt.plot(angles, P)
plt.show()



