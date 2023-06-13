import numpy as np
import numpy.polynomial.polynomial as np_poly
from . import _steering as stg
from . import _algebra as alg


def conventional_beamformer(signal, resolution, separation=1 / 2):
    angular_power = []
    angles = np.arange(-90, 90, resolution)
    elements = np.shape(signal)[0]
    R = alg.get_covariance(signal)

    for angle in angles:
        A = stg.generate_ula_vectors(angle, elements, separation)
        p = np.abs(np.matmul(np.matmul(A.conj().T, R), A))
        angular_power.append(np.squeeze(p))

    return angular_power, angles


def CAPON_MVDR(signal, resolution, separation=1 / 2):
    angular_power = []
    angles = np.arange(-90, 90 + resolution, resolution)
    elements = np.shape(signal)[0]
    R = alg.get_covariance(signal)
    R_inv = np.linalg.inv(R)

    for angle in angles:
        A = stg.generate_ula_vectors([angle, ], elements, separation)
        p = 1 / np.abs(np.matmul(np.matmul(A.conj().T, R_inv), A))
        angular_power.append(np.squeeze(p))

    return angular_power, angles


def ESPRIT(signal, model_order, separation=1 / 2):
    Qs, _ = alg.get_subspaces(signal, model_order)
    phi = np.linalg.lstsq(Qs[0:-1, :], Qs[1:, :], rcond=None)[0]
    ESPRIT_doas = np.arcsin(-np.angle(np.linalg.eigvals(phi)) / (2 * np.pi * separation)) * 180 / np.pi

    return ESPRIT_doas


def MUSIC(signal, model_order, resolution, separation=1 / 2):
    _, Qn = alg.get_subspaces(signal, model_order)
    angular_power = []
    angles = np.arange(-90, 90, resolution)
    elements = np.shape(signal)[0]

    for angle in angles:
        A = stg.generate_ula_vectors([angle, ], elements, separation)
        p = np.abs((A.conj().T @ A) / (A.conj().T @ Qn @ Qn.conj().T @ A))
        angular_power.append(np.squeeze(p))

    return angular_power, angles


def Min_Norm(signal, model_order, resolution, separation=1 / 2):
    _, Qn = alg.get_subspaces(signal, model_order)
    angular_power = []
    angles = np.arange(-90, 90, resolution)
    elements = np.shape(signal)[0]
    Pi_n = Qn @ Qn.conj().T
    w = np.zeros(elements)
    w[0] = 1
    w = np.atleast_2d(w)
    W = w.T @ w

    for angle in angles:
        A = stg.generate_ula_vectors([angle, ], elements, separation)
        p = np.abs((A.conj().T @ A) / (A.conj().T @ Pi_n @ W @ Pi_n @ A))
        angular_power.append(np.squeeze(p))

    return angular_power, angles


def Root_MUSIC(received_signal, model_order, separation=1 / 2):
    _, Qn = alg.get_subspaces(received_signal, model_order)
    C = Qn @ Qn.conj().T
    m = C.shape[0]

    a = np.zeros(2 * m - 1, dtype=complex)
    for k in range(len(a)):
        a[k] = np.trace(C, k - m - 1)

    ra = np_poly.polyroots(a)
    uc_dist = np.abs(np.abs(ra) - 1)
    idx_ord = np.argsort(uc_dist)
    ra = ra[idx_ord]
    angle = np.arcsin(-np.angle(ra) / (2 * np.pi * separation)) * 180 / np.pi
    return angle[:model_order]


def SAGE(received_signal, model_order, resolution=0.1, separation=1 / 2):
    Xn = received_signal
    angles = np.arange(-90, 90 + resolution, resolution)

    doas_ini = np.arange(0, 180, 180 / model_order)

    ESAGE_doas = np.zeros(model_order)
    N = np.shape(Xn)[0]

    doas_est = stg.generate_ula_vectors(doas_ini, N, separation)

    Ks = np.identity(model_order)
    last_DOAS = []
    for _ in range(30):

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

            Ks[signal, signal] = np.abs(((1 / (np.conj(A_filter).T @ A_filter)) @ (
                    (np.conj(A_filter).T @ Cx @ A_filter) / (np.conj(A_filter).T @ A_filter))) - 1 / N);

            doas_est[:, signal] = A_filter[:, 0]

        if np.array_equal(last_DOAS, ESAGE_doas):
            break
        else:
            last_DOAS = ESAGE_doas

    return ESAGE_doas
