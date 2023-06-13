import numpy as np
import numpy.polynomial.polynomial as np_poly
from . import _steering as stg
from . import _algebra as alg


def beamformer(signal: np.ndarray, resolution: float, separation: float = 1 / 2) -> tuple[np.ndarray, np.ndarray]:
    """Conventional beamformer.

    Parameters
    ----------
    signal: array_like
        A MxN array containing the N received samples measured at the M antennas of the array.
    resolution: float
        The angular resolution of the angular power spectrum to be returned.
    separation: float
        The inner element separation of the array.

    Returns
    -------
    angular_power, angles:
        A two element tuple containing an array containing the angular spectrum measured by the beamformer using the
        specified resolution and the angles at which the spectrum has been estimated.

    """

    angles = np.arange(-90, 90, resolution)
    angular_power = np.zeros(len(angles))
    elements = np.shape(signal)[0]
    R = alg.get_covariance(signal)

    for idx, angle in enumerate(angles):
        A = stg.generate_ula_vectors(angle, elements, separation)
        p = np.abs(np.matmul(np.matmul(A.conj().T, R), A))
        angular_power[idx] = np.squeeze(p)

    return angular_power, angles


def CAPON_MVDR(signal: np.ndarray, resolution: float, separation: float = 1 / 2) -> tuple[np.ndarray, np.ndarray]:
    """Capon's spectrum.
    This will have worse results than the conventional beamformer if the data is highly correlated, i.e. if signal
    covariance is near singular.

    Parameters
    ----------
    signal: array_like
        A MxN array containing the N received samples measured at the M antennas of the array.
    resolution: float
        The angular resolution of the angular power spectrum to be returned.
    separation: float
        The inner element separation of the array.

    Returns
    -------
    angular_power, angles:
        A two element tuple containing an array containing the angular spectrum measured by the beamformer using the
        specified resolution and the angles at which the spectrum has been estimated.

    """

    angles = np.arange(-90, 90, resolution)
    angular_power = np.zeros(len(angles))
    elements = np.shape(signal)[0]
    R = alg.get_covariance(signal)
    R_inv = np.linalg.inv(R)

    for idx, angle in enumerate(angles):
        A = stg.generate_ula_vectors([angle, ], elements, separation)
        p = 1 / np.abs(np.matmul(np.matmul(A.conj().T, R_inv), A))
        angular_power[idx] = np.squeeze(p)

    return angular_power, angles


def ESPRIT(signal: np.ndarray, model_order: int, separation: float = 1 / 2) -> np.ndarray:
    """ESPRIT DOA estimation.
    It requires a model order estimation to properly separate the signal and noise subspaces.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal: array_like
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order: int
        The number of signals whose DOAs are to be estimated.
    separation: float
        The inner element separation of the array.

    Returns
    -------
    angles:
        An ndarray containing the estimated angles.

    """

    Qs, _ = alg.get_subspaces(signal, model_order)
    phi = np.linalg.lstsq(Qs[0:-1, :], Qs[1:, :], rcond=None)[0]
    ESPRIT_doas = np.arcsin(-np.angle(np.linalg.eigvals(phi)) / (2 * np.pi * separation)) * 180 / np.pi

    return ESPRIT_doas


def MUSIC(signal: np.ndarray, model_order: int,
          resolution: float, separation: float = 1 / 2) -> tuple[np.ndarray, np.ndarray]:
    """MUSIC angular spectrum.
    It requires a model order estimation to properly separate the signal and noise subspaces.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal: array_like
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order: int
        The number of signals whose DOAs are to be estimated.
    resolution: float
        The angular resolution of the angular power spectrum to be returned.
    separation: float
        The inner element separation of the array.

    Returns
    -------
    angular_power, angles:
        A two element tuple containing an array containing the angular spectrum measured by the beamformer using the
        specified resolution and the angles at which the spectrum has been estimated.

    """

    _, Qn = alg.get_subspaces(signal, model_order)
    angles = np.arange(-90, 90, resolution)
    angular_power = np.zeros(len(angles))
    elements = np.shape(signal)[0]

    for idx, angle in enumerate(angles):
        A = stg.generate_ula_vectors([angle, ], elements, separation)
        p = np.abs((A.conj().T @ A) / (A.conj().T @ Qn @ Qn.conj().T @ A))
        angular_power[idx] = np.squeeze(p)

    return angular_power, angles


def Min_Norm(signal: np.ndarray, model_order: int,
             resolution: float, separation: float = 1 / 2) -> tuple[np.ndarray, np.ndarray]:
    """Min-Norm angular spectrum. A weighted version of the MUSIC algorithm.
    It requires a model order estimation to properly separate the signal and noise subspaces.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal: array_like
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order: int
        The number of signals whose DOAs are to be estimated.
    resolution: float
        The angular resolution of the angular power spectrum to be returned.
    separation: float
        The inner element separation of the array.

    Returns
    -------
    angular_power, angles:
        A two element tuple containing an array containing the angular spectrum measured by the beamformer using the
        specified resolution and the angles at which the spectrum has been estimated.

    """

    _, Qn = alg.get_subspaces(signal, model_order)
    angles = np.arange(-90, 90, resolution)
    angular_power = np.zeros(len(angles))
    elements = np.shape(signal)[0]
    Pi_n = Qn @ Qn.conj().T
    w = np.zeros(elements)
    w[0] = 1
    w = np.atleast_2d(w)
    W = w.T @ w

    for idx, angle in enumerate(angles):
        A = stg.generate_ula_vectors([angle, ], elements, separation)
        p = np.abs((A.conj().T @ A) / (A.conj().T @ Pi_n @ W @ Pi_n @ A))
        angular_power[idx] = np.squeeze(p)

    return angular_power, angles


def Root_MUSIC(signal: np.ndarray, model_order: int, separation: float = 1 / 2) -> np.ndarray:
    """Root Music DOA estimation.
    It requires a model order estimation to properly separate the signal and noise subspaces.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal: array_like
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order: int
        The number of signals whose DOAs are to be estimated.
    separation: float
        The inner element separation of the array.

    Returns
    -------
    angles:
        An ndarray containing the estimated angles.

    """

    _, Qn = alg.get_subspaces(signal, model_order)
    C = Qn @ Qn.conj().T
    m = C.shape[0]

    b = np.zeros(2 * m - 1, dtype=complex)
    for k in range(len(b)):
        b[k] = np.trace(C, k - m - 1)

    rb = np_poly.polyroots(b)
    uc_dist = np.abs(np.abs(rb) - 1)
    idx_ord = np.argsort(uc_dist)
    rb = rb[idx_ord]
    angle = np.arcsin(-np.angle(rb) / (2 * np.pi * separation)) * 180 / np.pi
    return angle[:model_order]


def SAGE(signal: np.ndarray, model_order: int,
         resolution: float = 0.1, separation: float = 1 / 2) -> np.ndarray:
    """SAGE DOA estimation.
    It requires a model order estimation to properly separate the signal and noise subspaces.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal: array_like
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order: int
        The number of signals whose DOAs are to be estimated.
    resolution: float
        The angular resolution of the angular power spectrum to be returned.
    separation: float
        The inner element separation of the array.

    Returns
    -------
    angles:
        An ndarray containing the estimated angles.

    """

    Xn = signal
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
                    (np.conj(A_filter).T @ Cx @ A_filter) / (np.conj(A_filter).T @ A_filter))) - 1 / N)

            doas_est[:, signal] = A_filter[:, 0]

        if np.array_equal(last_DOAS, ESAGE_doas):
            break
        else:
            last_DOAS = ESAGE_doas

    return ESAGE_doas
