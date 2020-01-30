import SignalGenerator as sig
import matplotlib.pyplot as plt
import SteeringGenerator as str
import DOAEstimators as est
import numpy as np
import GeometryLibrary as geo

transmitter = geo.Point(0, 30)
reflectors = [geo.Point(-20, 20), geo.Point(15, 10)]
receiver = geo.Point(0, 0)

angles = []
reflections = []
distances = []

for reflector in reflectors:
    angles.append(geo.getIncidenceAngle(reflector, receiver, True))
    reflections.append(geo.getReflectionAngle(transmitter, reflector, receiver, True))
    distances.append(geo.getPointDistance(transmitter, reflector)+geo.getPointDistance(reflector, receiver))

A = str.generate_ula_vectors(angles, 15, 1/2)
S = sig.gen_signal(2, 100)

u = str.generate_polarization_steering(reflections, 1)

Z = str.merge_space_polarization_steering(A, u)
X = np.matmul(Z, S)
X = sig.add_noise(X, 30)
DOAS, REFLECS = est.SAGE_Polarization(X, 2, 0.01, 1/2, 1)

possible_positions = []

for doa, reflection, distance in zip(DOAS, REFLECS, distances):
    transmissor_longinquo_doa = geo.getPointsPositiveY(geo.getPointGivenSlopeDistance(geo.incidenceToSlope(doa, True), receiver, distance))[0]
    transmissor_longinquo_reflection = geo.getPointsPositiveY(geo.getPointGivenSlopeDistance(geo.reflectionToSlope(doa, reflection, True), receiver, distance))[0]
    possible_positions.append(geo.Line.fromPoints(transmissor_longinquo_doa, transmissor_longinquo_reflection))

for position in possible_positions:
    print(position.getY(0))
    y = []
    x = np.linspace(-50, 50)
    for x_i in x:
        y.append(position.getY(x_i))
    plt.plot(x, y)

plt.show()

estimated_pos = geo.getLinesIntercept(possible_positions[0], possible_positions[1])
error = geo.getPointDistance(transmitter, estimated_pos)
print(estimated_pos)
print(error)