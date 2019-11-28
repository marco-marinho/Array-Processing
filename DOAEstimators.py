import numpy as np
import SteeringGenerator as stg
import AlgebraLibrary as alg
import matplotlib.pyplot as plt


def conventional_beamformer(signal, resolution, separation = 1/2):

    angular_power = []
    angles = np.arange(-90, 90, resolution)
    elements = np.shape(signal)[0]
    R = alg.get_covariance(signal)

    for angle in angles:
        A = stg.generate_ula_vectors(angle, elements, separation)
        p = np.abs(np.matmul(np.matmul(A.conj().T, R),A))
        angular_power.append(np.squeeze(p))

    return angular_power, angles

def conventional_beamformer_sparse(signal, resolution, positions, wavenumber):

    angular_power = []
    angles = np.arange(-90, 90, resolution)
    R = alg.get_covariance(signal)

    for angle in angles:
        A = stg.generate_sparse_vectors([angle,], positions, wavenumber)
        p = np.abs(np.matmul(np.matmul(A.conj().T, R),A))
        angular_power.append(np.squeeze(p))

    return angular_power, angles

def CAPON_MVDR_sparse(signal, resolution, positions, wavenumber):

    angular_power = []
    angles = np.arange(-90, 90, resolution)
    R = alg.get_covariance(signal)
    R_inv = np.linalg.inv(R)

    for angle in angles:
        A = stg.generate_sparse_vectors([angle,], positions, wavenumber)
        p = 1/np.abs(np.matmul(np.matmul(A.conj().T, R_inv),A))
        angular_power.append(np.squeeze(p))

    return angular_power, angles


def ESPRIT(signal, model_order, separation = 1/2):

    Qs, Qn = alg.get_subspaces(signal, model_order)
    phi = np.linalg.lstsq(Qs[0:-1, :], Qs[1:, :], rcond=None)[0]
    ESPRIT_doas = np.arcsin(-np.angle(np.linalg.eigvals(phi)) / (2 * np.pi * separation)) * 180 / np.pi

    return ESPRIT_doas

def SAGE(received_signal, model_order, resolution = 0.1, separation = 1/2):

    Xn = received_signal
    angles = np.arange(-90, 90, resolution)
    print(angles)
    doas_ini = np.arange(0, 180, 180 / model_order)
    ESAGE_doas = np.zeros(model_order)
    N = np.shape(Xn)[0]

    doas_est = stg.generate_ula_vectors([doas_ini, ], N, separation)

    Ks = np.identity(model_order)

    for iter in range(30):

        for signal in range(model_order):

            Kx = Ks[signal, signal] * doas_est @ np.conj(doas_est).T + np.identity(N)

            Ky = doas_est @ Ks @ np.conj(doas_est).T + np.identity(N)

            Ry = alg.get_covariance(Xn)

            Cx = Kx @ np.linalg.inv(Ky) @ Ry @ np.linalg.inv(Ky) @ Kx + Kx - Kx @ np.linalg.inv(Ky) @ Kx

            Pmaxexp = []

            for angle in range(len(angles)):
                A = stg.generate_ula_vectors([angles[angle], ], N, separation)
                Pmaxexp.append(np.squeeze(np.abs(np.conj(A).T @ Cx @ A / np.conj(A).T @ A)))

            plt.plot(angles, Pmaxexp)
            plt.show()
            index = Pmaxexp.index(max(Pmaxexp))

            index = Pmaxexp.index(max(Pmaxexp))

            ESAGE_doas[signal] = angles[index]

            A_filter = stg.generate_ula_vectors([angles[index], ], N, separation)

            Ks[signal, signal] = np.abs(((1/(np.conj(A_filter).T@A_filter))@((np.conj(A_filter).T@Cx@A_filter)/(np.conj(A_filter).T@A_filter)))-1/N);

            doas_est[:, signal] = A_filter[:, 0]

            print(angles[index])


def SAGE_sparse(signal, model_order, positions, wavenumber, resolution):

    Xn = signal
    angles = np.arange(-90, 90, resolution)
    doas_ini = np.arange(0, 180, 180/model_order)
    doas_ini = [60, ]
    # print(doas_ini)
    # doas_ini = doas_ini[0:model_order] - 90
    # print(doas_ini)
    doas_est = stg.generate_sparse_vectors(doas_ini, positions, wavenumber)

    samples = np.shape(Xn)[1]
    N = len(positions)

    ESAGE_doas = np.zeros(model_order)

    Ks = np.identity(model_order)

    for iter in range(100):

        for signal in range(model_order):
            Kx = Ks[signal, signal] * (doas_est[:, signal] @ doas_est[:, signal].T) + np.identity(N)

            Ky = (doas_est[:, [signal]] @ Ks @ doas_est[:, [signal]].T) + np.identity(N)

            Ry = alg.get_covariance(Xn)

            Cx = (Kx @ np.linalg.pinv(Ky) @ Ry @  np.linalg.pinv(Ky) @ Kx) + Kx - (Kx @  np.linalg.pinv(Ky) @ Kx)

            Pmaxexp = []
            for angle in range(len(angles)):
                A_search = stg.generate_sparse_vectors([angles[angle], ], positions, wavenumber)
                #Pmaxexp.append(np.squeeze(np.abs((A_search.T  @ Cx @ A_search)/(A_search.T @ A_search))))
                Pmaxexp.append(np.squeeze(np.abs(A_search.T @ Cx @ A_search)))
            print(np.shape(Pmaxexp))
            plt.plot(angles, Pmaxexp)
            plt.show()
            index = Pmaxexp.index(max(Pmaxexp))

            A_filter = stg.generate_sparse_vectors([angles[index], ], positions, wavenumber)
            Ks[signal, signal] = np.abs(((1/(A_filter.T @ A_filter))*((A_filter.T @ Cx @ A_filter)/(A_filter.T @ A_filter)))-1/N)

            doas_est[:, signal] = A_filter[:, 0]

            ESAGE_doas[signal] = angles[index]

            print(angles[index])

    return ESAGE_doas
