import SignalGenerator as sig
import matplotlib.pyplot as plt
import SteeringGenerator as str
import DOAEstimators as est
import numpy as np
import AlgebraLibrary as alg
import GeometryLibrary as geo
import tensorly.tenalg as tls

# A = str.generate_ula_vectors([-45, 60], 8, 1/2)
# S = sig.gen_signal(2, 100, 15)
#
# X = np.matmul(A, S)

# angular_power, angles = est.conventional_beamformer(X, 2, 1/2)
#
# plt.plot(angles, angular_power)
# plt.show()

#positions = np.array([[-0.25, 0.25], [-0.25, 0],  [-0.25, -0.25], [-0.125, 0.125], [-0.125, -0.125], [0, 0.25], [0, 0], [0, -0.25], [0.125, 0.125], [0.125, -0.125], [0.25, 0.25], [0.25, 0], [0.25, -0.25]])*4
#A = str.generate_sparse_vectors([-45, 60], positions, 0.125)


# angular_power, angles = est.conventional_beamformer_sparse(X, 0.1, positions, 0.125)
# plt.plot(angles, angular_power)
# plt.show()

# errosAngle = []
# errosPolari = []
# index = 0
# for SNR in np.arange(5,30,5):
#     print(SNR)
#     errosAngle.append(0)
#     errosPolari.append(0)
#     for trie in range(30):
#         angles=[15, 60]
#         reflections = [27, 35]
#
#         A = str.generate_ula_vectors(angles, 15, 1/2)
#         S = sig.gen_signal(2, 100)
#
#         u = str.generate_polarization_steering(reflections, 1)
#
#         Z = str.merge_space_polarization_steering(A, u)
#         X = np.matmul(Z, S)
#         X = sig.add_noise(X, SNR)
#         DOAS2, Pols = est.SAGE_Polarization(X, 2, 0.1, 1/2, 1)
#         DOAS2.sort()
#         Pols.sort()
#         errosAngle[index] += np.mean(np.abs(np.subtract(DOAS2, angles)))
#         errosPolari[index] += np.mean(np.abs(np.subtract(Pols, reflections)))
#
#     errosAngle[index] = errosAngle[index]/100
#     errosPolari[index] = errosPolari[index]/100
#     index += 1

# ratios = []
# for angle in range(0,60):
#     u = str.generate_polarization_steering([angle, ], 1)
#     ratios.append(abs(u[0,0]/u[1,0]))
#
# plt.plot(range(0,60), ratios)
# plt.show()

#
#
# plt.plot(angles, angular_power)
# plt.show()

#
# s = sig.generate_OFDM_signal(16000, 32, [0.3, -0.5, 0, 1, 0.2, -0.3], 5, 15, True, True)

# angles = np.arange(0, 120, 1)
# results = []
#
# for angle in angles:
#     vector = str.generate_polarization_steering((angle, ), 1)
#     results.append((vector[0]/vector[1])[0])
#
# plt.plot(angles, results)
# plt.show()

A = np.array([[1, 1], [1, 1]])
u = np.atleast_2d(np.array([[2, 3],[4,5]]))
X = tls.khatri_rao([A,u])
