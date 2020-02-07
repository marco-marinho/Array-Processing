import SignalGenerator as sig
import matplotlib.pyplot as plt
import SteeringGenerator as str
import DOAEstimators as est
import numpy as np
import GeometryLibrary as geo

transmitter = geo.Point(0, 30)
reflectors = [geo.Point(-30, 20), geo.Point(25, 10), geo.Point(15, 17)]
receiver = geo.Point(0, 0)

angles = []
reflections = []
distances = []

for reflector in reflectors:
    angles.append(geo.getIncidenceAngle(reflector, receiver, True))
    reflections.append(geo.getReflectionAngle2(transmitter, reflector, receiver, True))
    distances.append(geo.getPointDistance(transmitter, reflector)+geo.getPointDistance(reflector, receiver))
    line = geo.Line.fromPoints(transmitter, reflector)

print(angles)
total_tries = 30

SNRs = np.arange(5, 30, 5)
errors = [0]*len(SNRs)
tries = np.arange(0, total_tries, 1)
index = 0

for SNR in SNRs:

    sucessos = 0
    for trie in tries:
        A = str.generate_ula_vectors_center(angles, 11, 1/2)
        S = sig.gen_signal(len(reflectors), 1000)

        u = str.generate_polarization_steering(reflections, 1)

        Z = str.merge_space_polarization_steering(A, u)
        X = np.matmul(Z, S)
        X = sig.add_noise(X, SNR)
        X = sig.doFBA_Polarization(X)

        ESPRIT = est.ESPRIT_Polarization(X, len(reflectors))
        print(ESPRIT)

        DOAS, REFLECS = est.SAGE_Polarization_Center(X, len(reflectors), 0.01, 1/2, 1)
        DIST = est.distancePairing((angles, reflections), (DOAS, REFLECS), distances)
        possible_positions = []

        for doa, reflection, distance in zip(DOAS, REFLECS, DIST):

            transmissor_longinquo_doa = geo.getPointsPositiveY(geo.getPointGivenSlopeDistance(geo.incidenceToSlope(doa, True), receiver, distance))[0]
            transmissor_longinquo_reflection = geo.getPointsPositiveY(geo.getPointGivenSlopeDistance(geo.reflectionToSlope2(doa, reflection, True), receiver, distance))[0]
            possible_positions.append(geo.Line.fromPoints(transmissor_longinquo_doa, transmissor_longinquo_reflection))

        estimated_pos = geo.getPositionEstimate(possible_positions)
        error = geo.getPointDistance(transmitter, estimated_pos)
        distance_transmitter = geo.getPointDistance(estimated_pos, receiver)
        print(error)
        if distance_transmitter > distance*1.1:
            print('fail')
        else:
            errors[index] += error
            sucessos += 1

    errors[index] = errors[index]/sucessos
    index += 1