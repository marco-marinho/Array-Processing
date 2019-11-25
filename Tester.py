import SignalGenerator as sig
import matplotlib.pyplot as plt
import SteeringGenerator as str
import DOAEstimators as est
import numpy as np
import AlgebraLibrary as alg
import GeometryLibrary as geo

# A = str.generate_ula_vectors([-45, 60], 8, 1/2)
# S = sig.gen_signal(2, 100, 15)
#
# X = np.matmul(A, S)

# angular_power, angles = est.conventional_beamformer(X, 2, 1/2)
#
# plt.plot(angles, angular_power)
# plt.show()

# positions = [[-1, 1], [-1, 0], [-1, -1], [-0.5, 0.5], [-0.5, -0.5], [0, 1], [0, 0], [0, -1], [0.5, 0.5], [0.5, -0.5], [1, 1], [1, 0], [1, -1]]
# A = str.generate_sparse_vectors([60, ], positions, 0.125)
# S = sig.gen_signal(1, 100, 15)
# X = np.matmul(A, S)
# angular_power, angles = est.CAPON_MVDR_sparse(X, 1, positions, 0.125)
#
# plt.plot(angles, angular_power)
# plt.show()


# s = sig.generate_OFDM_signal(16000, 32, [0.3, -0.5, 0, 1, 0.2, -0.3], 5, 30, True, True)

point_1 = geo.Point(0, 0)
point_2 = geo.Point(-2, 2)
point_3 = geo.Point(0, 3)
line = geo.Line.fromPoints(point_1, point_2)
print(line.getSlope())
print(line.getIntercept())

line_1 = geo.Line.fromPoints(point_1, point_2)
line_2 = geo.Line.fromPoints(point_2, point_3)

print(geo.getLinesIntercept(line_1, line_2))
print(geo.getAngleLines(line_1, line_2, inDegree=True))

print(geo.getPointDistance(point_1, point_2)+geo.getPointDistance(point_2, point_3))

print(geo.getPointInLineGivenDistance(line_2, point_2, geo.getPointDistance(point_2, point_3)))
