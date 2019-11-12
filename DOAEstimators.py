import numpy as np
import SteeringGenerator as stg
import AlgebraLibrary as alg


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

def SAGE_sparse(signal, model_order, positions, wavenumber, resolution):

    angles = np.arange(-90, 90, resolution);
    doas_ini = np.arrange(0, 180, 180/model_order)
    doas_ini = doas_ini[1:model_order+1] - 90

    doas_est = stg.generate_sparse_vectors(doas_ini, positions, wavenumber)

    samples = np.shape(signal)[1]

    sig_est = np.zeros(model_order, samples)

    ESAGE_doas = np.zeros(model_order)

    Ks = np.identity(model_order)

    for iter in range(100):

        for signal in range(model_order):

            Kx = Ks[signal, signal]*doas_est[:, signal]*doas_est[:, signal].T + np.identity(samples)
