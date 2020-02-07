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
    angles = np.arange(-90, 90+resolution, resolution)
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

def ESPRIT_Polarization(signal, model_order, separation = 1/2):
    antennas = int(np.shape(signal)[0]/2)

    Qs, Qn = alg.get_subspaces(signal, model_order)
    phi = np.linalg.lstsq(Qs[0:2*(antennas-1), :], Qs[2:2*antennas, :], rcond=None)[0]
    ESPRIT_doas = np.arcsin(-np.angle(np.linalg.eigvals(phi)) / (2 * np.pi * separation)) * 180 / np.pi

    phi_polarization = np.linalg.lstsq(Qs[np.arange(1, antennas * 2, 2), :], Qs[np.arange(0, antennas * 2, 2), :], rcond=None)[0]
    ESPRIT_Polarization = np.abs(np.linalg.eigvals(phi_polarization))

    return ESPRIT_doas, ESPRIT_Polarization

def SAGE(received_signal, model_order, resolution = 0.1, separation = 1/2):

    Xn = received_signal
    angles = np.arange(-90, 90+resolution, resolution)

    doas_ini = np.arange(0, 180, 180 / model_order)

    ESAGE_doas = np.zeros(model_order)
    N = np.shape(Xn)[0]

    doas_est = stg.generate_ula_vectors(doas_ini, N, separation)

    Ks = np.identity(model_order)
    last_DOAS = []
    for iter in range(30):

        for signal in range(model_order):

            Kx = Ks[signal, signal] * doas_est[:, [signal]] @ np.conj(doas_est[:, [signal]]).T + np.identity(N)

            Ky = doas_est @ Ks @ np.conj(doas_est).T + np.identity(N)

            Ry = alg.get_covariance(Xn)

            Cx = Kx @ np.linalg.inv(Ky) @ Ry @ np.linalg.inv(Ky) @ Kx + Kx - Kx @ np.linalg.inv(Ky) @ Kx

            Pmaxexp = []

            for angle in range(len(angles)):

                A = stg.generate_ula_vectors(angles[angle], N, separation)
                Pmaxexp.append(np.squeeze(np.abs((np.conj(A).T @ Cx @ A) / (np.conj(A).T @ A))))

            index = Pmaxexp.index(max(Pmaxexp))

            ESAGE_doas[signal] = angles[index]

            A_filter = stg.generate_ula_vectors(angles[index], N, separation)

            Ks[signal, signal] = np.abs(((1/(np.conj(A_filter).T@A_filter))@((np.conj(A_filter).T@Cx@A_filter)/(np.conj(A_filter).T@A_filter)))-1/N);

            doas_est[:, signal] = A_filter[:, 0]

        if np.array_equal(last_DOAS, ESAGE_doas):
            break
        else:
            last_DOAS = ESAGE_doas

    return ESAGE_doas

def SAGE_Polarization(received_signal, model_order, resolution = 0.1, separation = 1/2, E_ref = 1):

    Xn = received_signal
    angles = np.arange(-90, 90+resolution, resolution)
    reflections = np.arange(0, 180+resolution, resolution)

    doas_ini = np.arange(0, 180, 180 / model_order)
    reflection_ini = np.zeros(model_order)

    ESAGE_doas = np.zeros(model_order)
    ESAGE_polarization = np.zeros(model_order)
    N = int(np.shape(Xn)[0]/2)

    doas_est = stg.generate_ula_vectors(doas_ini, N, separation)
    reflections_est = stg.generate_polarization_steering(reflection_ini, E_ref)

    steering_est = stg.merge_space_polarization_steering(doas_est, reflections_est)

    Ks = np.identity(model_order)
    last_DOAS = []
    for iter in range(30):

        for signal in range(model_order):

            Kx = Ks[signal, signal] * steering_est[:, [signal]] @ np.conj(steering_est[:, [signal]]).T + np.identity(N*2)

            Ky = steering_est @ Ks @ np.conj(steering_est).T + np.identity(N*2)

            Ry = alg.get_covariance(Xn)

            Cx = Kx @ np.linalg.inv(Ky) @ Ry @ np.linalg.inv(Ky) @ Kx + Kx - Kx @ np.linalg.inv(Ky) @ Kx

            Pmaxexp = []

            u = stg.generate_polarization_steering([ESAGE_polarization[signal], ], E_ref)
            for angle in range(len(angles)):

                A_s = stg.generate_ula_vectors(angles[angle], N, separation)
                A = stg.merge_space_polarization_steering(A_s, u)
                Pmaxexp.append(np.squeeze(np.abs((np.conj(A).T @ Cx @ A) / (np.conj(A).T @ A))))

            index = Pmaxexp.index(max(Pmaxexp))

            ESAGE_doas[signal] = angles[index]
            A_e = stg.generate_ula_vectors(angles[index], N, separation)
            Pmaxexp = []

            A_s = stg.generate_ula_vectors(angles[index], N, separation)
            for angle in range(len(reflections)):

                u = stg.generate_polarization_steering([reflections[angle], ], E_ref)
                A = stg.merge_space_polarization_steering(A_s, u)
                Pmaxexp.append(np.squeeze(np.abs((np.conj(A).T @ Cx @ A) / (np.conj(A).T @ A))))

            index = Pmaxexp.index(max(Pmaxexp))

            ESAGE_polarization[signal] = reflections[index]

            u_e = stg.generate_polarization_steering([reflections[index], ], E_ref)
            A_filter = stg.merge_space_polarization_steering(A_e, u_e)

            Ks[signal, signal] = np.abs(((1/(np.conj(A_filter).T@A_filter)))@((np.conj(A_filter).T@Cx@A_filter)/(np.conj(A_filter).T@A_filter)-1/model_order))

            steering_est[:, signal] = A_filter[:, 0]

        if np.array_equal(last_DOAS, ESAGE_doas):
            break
        else:
            last_DOAS = ESAGE_doas

    return ESAGE_doas, ESAGE_polarization


def SAGE_Polarization_Center(received_signal, model_order, resolution = 0.1, separation = 1/2, E_ref = 1):

    Xn = received_signal
    angles = np.arange(-90, 90+resolution, resolution)
    reflections = np.arange(0, 180+resolution, resolution)

    doas_ini = np.arange(-90+(180/(model_order+1)), 90-(180/(model_order+1))+1, 180 / (model_order+1))
    reflection_ini = (np.ones(model_order))*60

    ESAGE_doas = np.zeros(model_order)
    ESAGE_polarization = np.zeros(model_order)
    N = int(np.shape(Xn)[0]/2)

    doas_est = stg.generate_ula_vectors_center(doas_ini, N, separation)
    reflections_est = stg.generate_polarization_steering(reflection_ini, E_ref)

    steering_est = stg.merge_space_polarization_steering(doas_est, reflections_est)

    Ks = np.identity(model_order)
    last_DOAS = []
    for iter in range(30):

        for signal in range(model_order):

            Kx = Ks[signal, signal] * steering_est[:, [signal]] @ np.conj(steering_est[:, [signal]]).T + np.identity(N*2)

            Ky = steering_est @ Ks @ np.conj(steering_est).T + np.identity(N*2)

            Ry = alg.get_covariance(Xn)

            Cx = Kx @ np.linalg.inv(Ky) @ Ry @ np.linalg.inv(Ky) @ Kx + Kx - Kx @ np.linalg.inv(Ky) @ Kx

            Pmaxexp = []

            u = stg.generate_polarization_steering([ESAGE_polarization[signal], ], E_ref)
            for angle in range(len(angles)):

                A_s = stg.generate_ula_vectors_center(angles[angle], N, separation)
                A = stg.merge_space_polarization_steering(A_s, u)
                Pmaxexp.append(np.squeeze(np.abs((np.conj(A).T @ Cx @ A) / (np.conj(A).T @ A))))

            index = Pmaxexp.index(max(Pmaxexp))

            ESAGE_doas[signal] = angles[index]
            A_e = stg.generate_ula_vectors_center(angles[index], N, separation)
            Pmaxexp = []

            A_s = stg.generate_ula_vectors_center(angles[index], N, separation)
            for angle in range(len(reflections)):

                u = stg.generate_polarization_steering([reflections[angle], ], E_ref)
                A = stg.merge_space_polarization_steering(A_s, u)
                Pmaxexp.append(np.squeeze(np.abs((np.conj(A).T @ Cx @ A) / (np.conj(A).T @ A))))

            index = Pmaxexp.index(max(Pmaxexp))

            ESAGE_polarization[signal] = reflections[index]

            u_e = stg.generate_polarization_steering([reflections[index], ], E_ref)
            A_filter = stg.merge_space_polarization_steering(A_e, u_e)

            Ks[signal, signal] = np.abs(((1/(np.conj(A_filter).T@A_filter)))@((np.conj(A_filter).T@Cx@A_filter)/(np.conj(A_filter).T@A_filter)-1/model_order))

            steering_est[:, signal] = A_filter[:, 0]

        if np.array_equal(last_DOAS, ESAGE_doas):
            break
        else:
            last_DOAS = ESAGE_doas

    return ESAGE_doas, ESAGE_polarization


def SAGE_sparse(received_signal, model_order, positions, wavenumber, resolution):

    Xn = received_signal
    angles = np.arange(-90, 90 + resolution, resolution)

    doas_ini = np.arange(0, 180, 180 / model_order)

    doas_est = stg.generate_sparse_vectors(doas_ini, positions, wavenumber)

    N = len(positions)

    ESAGE_doas = np.zeros(model_order)

    Ks = np.identity(model_order)

    for iter in range(1):

        for signal in range(model_order):

            Kx = Ks[signal, signal] * doas_est[:, [signal]] @ np.conj(doas_est[:, [signal]]).T + np.identity(N)

            Ky = doas_est @ Ks @ np.conj(doas_est).T + np.identity(N)

            Ry = alg.get_covariance(Xn)

            Cx = Kx @ np.linalg.inv(Ky) @ Ry @ np.linalg.inv(Ky) @ Kx + Kx - Kx @ np.linalg.inv(Ky) @ Kx

            Pmaxexp = []

            for angle in range(len(angles)):
                A = stg.generate_sparse_vectors([angles[angle], ], positions, wavenumber)
                Pmaxexp.append(np.squeeze(np.abs((np.conj(A).T @ Cx @ A) / (np.conj(A).T @ A))))

            plt.plot(Pmaxexp)
            plt.show()
            index = Pmaxexp.index(max(Pmaxexp))

            ESAGE_doas[signal] = angles[index]

            A_filter = stg.generate_sparse_vectors([angles[index], ], positions, wavenumber)

            Ks[signal, signal] = np.abs(((1 / (np.conj(A_filter).T @ A_filter)) @ (
                        (np.conj(A_filter).T @ Cx @ A_filter) / (np.conj(A_filter).T @ A_filter))) - 1 / N);

            doas_est[:, signal] = A_filter[:, 0]

    print(ESAGE_doas)

def distancePairing(true, estimated, pair):
    tru = np.vstack(true)
    est = np.vstack(estimated)
    paired = [0]*len(pair)

    for column_out in range(tru.shape[1]):
        error = float('inf')
        for column_in in range(est.shape[1]):
            erro = np.sum(np.abs(tru[:, column_out] - est[:, column_in]))
            if erro < error:
                error = erro
                paired[column_out] = pair[column_in]

    return paired
