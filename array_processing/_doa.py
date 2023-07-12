import numpy as np
import numpy.polynomial.polynomial as np_poly
import itertools
from . import _steering as stg
from . import _algebra as alg


def beamformer(signal: np.ndarray, resolution: float, separation: float = 1 / 2) -> tuple[np.ndarray, np.ndarray]:
    """Conventional beamformer.

    Parameters
    ----------
    signal
        A MxN array containing the N received samples measured at the M antennas of the array.
    resolution
        The angular resolution of the angular power spectrum to be returned.
    separation
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
    R = alg.covariance(signal)

    for idx, angle in enumerate(angles):
        A = stg.ula_ch(angle, elements, separation)
        p = np.abs(np.matmul(np.matmul(A.conj().T, R), A))
        angular_power[idx] = 10 * np.log10(np.squeeze(p))

    return angular_power, angles


def CAPON_MVDR(signal: np.ndarray, resolution: float, separation: float = 1 / 2) -> tuple[np.ndarray, np.ndarray]:
    """Capon's spectrum.
    This will have worse results than the conventional beamformer if the data is highly correlated, i.e. if signal
    covariance is near singular.

    Parameters
    ----------
    signal
        A MxN array containing the N received samples measured at the M antennas of the array.
    resolution
        The angular resolution of the angular power spectrum to be returned.
    separation
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
    R = alg.covariance(signal)
    R_inv = np.linalg.inv(R)

    for idx, angle in enumerate(angles):
        A = stg.ula_ch([angle, ], elements, separation)
        p = 1 / np.abs(np.matmul(np.matmul(A.conj().T, R_inv), A))
        angular_power[idx] = 10 * np.log10(np.squeeze(p))

    return angular_power, angles


def ESPRIT(signal: np.ndarray, model_order: int, separation: float = 1 / 2, algorithm="TLS") -> tuple[float]:
    """ESPRIT DOA estimation.
    It requires a model order estimation to properly separate the signal and noise subspaces.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order
        The number of signals whose DOAs are to be estimated.
    separation
        The inner element separation of the array.
    algorithm:
        Select whether to use the TLS os LS versions of ESPRIT.

    Returns
    -------
    angles:
        An ndarray containing the estimated angles.

    """

    Qs, _ = alg.subspaces(signal, model_order)
    if algorithm == "TLS":
        U1 = Qs[0: -1, :]
        U2 = Qs[1:, :]
        C = np.r_[U1.conj().T, U2.conj().T] @ np.c_[U1, U2]
        l, V = np.linalg.eig(C)
        V = V[:, np.argsort(np.real(l))]
        V = V[:, ::-1]
        V_12 = V[:model_order, model_order:]
        V_22 = V[model_order:, model_order:]
        phi = -V_12 @ np.linalg.inv(V_22)
        ESPRIT_doas = np.arcsin(-np.angle(np.linalg.eigvals(phi)) / (2 * np.pi * separation)) * 180 / np.pi
    elif algorithm == "LS":
        phi = np.linalg.lstsq(Qs[0:-1, :], Qs[1:, :], rcond=None)[0]
        ESPRIT_doas = np.arcsin(-np.angle(np.linalg.eigvals(phi)) / (2 * np.pi * separation)) * 180 / np.pi
    else:
        raise ValueError("Invalid algorithm, only TSL and LS are supported")

    return tuple(ESPRIT_doas)


def MUSIC(signal: np.ndarray, model_order: int,
          resolution: float, separation: float = 1 / 2) -> tuple[np.ndarray, np.ndarray]:
    """MUSIC angular spectrum.
    It requires a model order estimation to properly separate the signal and noise subspaces.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order
        The number of signals whose DOAs are to be estimated.
    resolution
        The angular resolution of the angular power spectrum to be returned.
    separation
        The inner element separation of the array.

    Returns
    -------
    angular_power, angles:
        A two element tuple containing an array containing the angular spectrum measured by the beamformer using the
        specified resolution and the angles at which the spectrum has been estimated.

    """

    _, Qn = alg.subspaces(signal, model_order)
    angles = np.arange(-90, 90, resolution)
    angular_power = np.zeros(len(angles))
    elements = np.shape(signal)[0]

    for idx, angle in enumerate(angles):
        A = stg.ula_ch([angle, ], elements, separation)
        p = np.abs((A.conj().T @ A) / (A.conj().T @ Qn @ Qn.conj().T @ A))
        angular_power[idx] = 10 * np.log10(np.squeeze(p))

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
    signal
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order
        The number of signals whose DOAs are to be estimated.
    resolution
        The angular resolution of the angular power spectrum to be returned.
    separation
        The inner element separation of the array.

    Returns
    -------
    angular_power, angles:
        A two element tuple containing an array containing the angular spectrum measured by the beamformer using the
        specified resolution and the angles at which the spectrum has been estimated.

    """

    _, Qn = alg.subspaces(signal, model_order)
    angles = np.arange(-90, 90, resolution)
    angular_power = np.zeros(len(angles))
    elements = np.shape(signal)[0]
    Pi_n = Qn @ Qn.conj().T
    W = np.zeros((elements, elements))
    W[0, 0] = 1

    for idx, angle in enumerate(angles):
        A = stg.ula_ch([angle, ], elements, separation)
        p = np.abs((A.conj().T @ A) / (A.conj().T @ Pi_n @ W @ Pi_n @ A))
        angular_power[idx] = 10 * np.log10(np.squeeze(p))

    return angular_power, angles


def __Root_MUSIC(signal: np.ndarray, model_order: int, separation: float = 1 / 2, algorithm="MUSIC") -> tuple[float]:
    _, Qn = alg.subspaces(signal, model_order)
    m = np.shape(signal)[0]
    C = np.zeros([m, m], dtype=complex)
    if algorithm == "MUSIC":
        C = Qn @ Qn.conj().T
    elif algorithm == "MinNorm":
        Pi_n = Qn @ Qn.conj().T
        W = np.zeros((m, m), dtype=complex)
        W[0, 0] = 1
        C = Pi_n @ W @ Pi_n.conj().T

    b = np.zeros(2 * m - 1, dtype=complex)
    for k in range(len(b)):
        b[k] = np.trace(C, k - m + 1)

    rb = np_poly.polyroots(b)
    rb = rb[np.abs(rb) <= 1]
    uc_dist = np.abs(np.abs(rb) - 1)
    idx_ord = np.argsort(uc_dist)
    rb = rb[idx_ord]
    angle = np.arcsin(-np.angle(rb) / (2 * np.pi * separation)) * 180 / np.pi
    return tuple(angle[:model_order])


def Root_MUSIC(signal: np.ndarray, model_order: int, separation: float = 1 / 2) -> tuple[float]:
    """Root Music DOA estimation.
    It requires a model order estimation to properly separate the signal and noise subspaces.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order
        The number of signals whose DOAs are to be estimated.
    separation
        The inner element separation of the array.

    Returns
    -------
    angles:
        An ndarray containing the estimated angles.

    """
    return __Root_MUSIC(signal, model_order, separation, "MUSIC")


def Root_MinNorm(signal: np.ndarray, model_order: int, separation: float = 1 / 2) -> tuple[float]:
    """Root MinNorm DOA estimation.
    It requires a model order estimation to properly separate the signal and noise subspaces.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order
        The number of signals whose DOAs are to be estimated.
    separation
        The inner element separation of the array.

    Returns
    -------
    angles:
        An ndarray containing the estimated angles.

    """
    return __Root_MUSIC(signal, model_order, separation, "MinNorm")


def SML(signal: np.ndarray, model_order: int, resolution: float = 0.1, separation: float = 1 / 2) -> tuple[float]:
    """Stochastic Maximum Likelihood DOA estimation.
    The model order is not strictly necessary for an ML estimator, but without it the estimation will take an
    unreasonable amount of time. The estimator can be extended to return the current log-likelihood and multiple calls
    to this function can be made for varying model orders.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order
        The number of signals whose DOAs are to be estimated.
    resolution
        The angular resolution with which to perform the estimation.
    separation
        The inner element separation of the array.

    Returns
    -------
    angles:
        An ndarray containing the estimated angles.

    """
    return __ML("S", signal, model_order, resolution, separation)


def DML(signal: np.ndarray, model_order: int, resolution: float = 0.1, separation: float = 1 / 2) -> tuple[float]:
    """Deterministic Maximum Likelihood DOA estimation.
    The model order is not strictly necessary for an ML estimator, but without it the estimation will take an
    unreasonable amount of time. The estimator can be extended to return the current log-likelihood and multiple calls
    to this function can be made for varying model orders.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order
        The number of signals whose DOAs are to be estimated.
    resolution
        The angular resolution with which to perform the estimation.
    separation
        The inner element separation of the array.

    Returns
    -------
    angles:
        An ndarray containing the estimated angles.

    """
    return __ML("D", signal, model_order, resolution, separation)


def __ML(mode: str, signal: np.ndarray, model_order: int,
         resolution: float = 0.1, separation: float = 1 / 2) -> tuple[float]:
    """ This function is part of the private API and encapsulates both the stochastic and deterministic ML. Do not call
    it directly. Instead, call the respect DML and SML functions.
    """
    angles = np.arange(-90, 90, resolution)
    estimates = itertools.product(angles, repeat=model_order)
    R = alg.covariance(signal)
    M = signal.shape[0]
    best_likelihood = np.inf
    output = np.ndarray([])

    for estimate in estimates:
        A_ml = stg.ula_ch(estimate, M, separation)
        Sigma_A = A_ml @ np.linalg.pinv(A_ml)
        Ort_Sigma_A = np.eye(M) - Sigma_A

        if mode == "D":
            current_likelihood = np.trace(Ort_Sigma_A @ R)
        elif mode == "S":
            omega = (1 / (M - model_order)) * np.trace(Ort_Sigma_A @ R)
            P = np.linalg.pinv(A_ml) @ (R - omega * np.eye(M)) @ np.linalg.pinv(A_ml).conj().T
            current_likelihood = np.log(np.linalg.det(A_ml @ P @ A_ml.conj().T + omega * np.eye(M)))
        else:
            raise ValueError("Invalid estimator type")

        if current_likelihood < best_likelihood:
            output = estimate
            best_likelihood = current_likelihood

    return output


def SAGE(signal: np.ndarray, model_order: int,
         resolution: float = 0.1, separation: float = 1 / 2) -> tuple[float]:
    """SAGE DOA estimation.
    The model order could, theoretically, be a parameter estimated by SAGE. This implementation assumes the model
    order is known.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order
        The number of signals whose DOAs are to be estimated.
    resolution
        The angular resolution of the angular power spectrum to be returned.
    separation
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

    doas_est = stg.ula_steering(doas_ini, N, separation)

    Ks = np.identity(model_order)
    last_DOAS = []
    for _ in range(30):

        for signal in range(model_order):

            Kx = Ks[signal, signal] * doas_est[:, [signal]] @ np.conj(doas_est[:, [signal]]).T + np.identity(N)

            Ky = doas_est @ Ks @ np.conj(doas_est).T + np.identity(N)

            Ry = alg.covariance(Xn)

            Cx = Kx @ np.linalg.inv(Ky) @ Ry @ np.linalg.inv(Ky) @ Kx + Kx - Kx @ np.linalg.inv(Ky) @ Kx

            Pmaxexp = []

            for angle in range(len(angles)):
                A = stg.ula_steering(angles[angle], N, separation)
                Pmaxexp.append(np.squeeze(np.abs((np.conj(A).T @ Cx @ A) / (np.conj(A).T @ A))))

            index = Pmaxexp.index(max(Pmaxexp))

            ESAGE_doas[signal] = angles[index]

            A_filter = stg.ula_steering(angles[index], N, separation)

            Ks[signal, signal] = np.abs(((1 / (np.conj(A_filter).T @ A_filter)) @ (
                    (np.conj(A_filter).T @ Cx @ A_filter) / (np.conj(A_filter).T @ A_filter))) - 1 / N)

            doas_est[:, signal] = A_filter[:, 0]

        if np.array_equal(last_DOAS, ESAGE_doas):
            break
        else:
            last_DOAS = ESAGE_doas

    return tuple(ESAGE_doas)


def IQML(signal: np.ndarray, model_order: int, separation: float = 1 / 2, /,
         max_iter: int = 1000, epsilon: float = 0.01):
    """IQML DOA estimation.
    Iterative Quadratic Maximum Likelihood estimator. Requires a model order estimate.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order
        The number of signals whose DOAs are to be estimated.
    separation
        The inner element separation of the array.
    max_iter:
        The maximum number of iterations allowed.
    epsilon:
        The tolerance for change in the polynomial vector for stopping the iterations.

    Returns
    -------
    angles:
        An ndarray containing the estimated angles.

    """
    return __IQML_MODE(signal, model_order, separation, max_iter, epsilon, algorithm="IQML")


def MODE(signal: np.ndarray, model_order: int, separation: float = 1 / 2, /,
         max_iter: int = 1000, epsilon: float = 0.0001):
    """Root-WSF (Weighted Subspace Fitting) or MODE (Method of Direction Estimation) DOA estimation algorithm.
    Requires a model order estimate.

    The literature states that MODE is a two-step algorithm. However, in some cases, the estimates can benefit from
    further iterations. See Van Trees book for on Optimal Array Processing for more details.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order
        The number of signals whose DOAs are to be estimated.
    separation
        The inner element separation of the array.
    max_iter:
        The maximum number of iterations allowed.
    epsilon:
        The tolerance for change in the polynomial vector for stopping the iterations.

    Returns
    -------
    angles:
        An ndarray containing the estimated angles.

    """
    return __IQML_MODE(signal, model_order, separation, max_iter, epsilon, algorithm="MODE")


def __IQML_MODE(signal: np.ndarray, model_order: int, separation: float = 1 / 2, /,
                max_iter: int = 1000, epsilon: float = 0.0001, algorithm="MODE") -> tuple[float]:
    """IQML and MODE DOA estimation backend.
    It should not be called directly from outside its own module, call the IQML and MODE functions instead.

    See Also
    ________
    array_processing.moe

    Parameters
    ----------
    signal
        A MxN array containing the N received samples measured at the M antennas of the array.
    model_order
        The number of signals whose DOAs are to be estimated.
    separation
        The inner element separation of the array.
    max_iter:
        The maximum number of iterations allowed.
    epsilon:
        The tolerance for change in the polynomial vector for stopping the iterations.

    Returns
    -------
    angles:
        An ndarray containing the estimated angles.

    """
    if algorithm == "MODE":
        signal, _ = alg.subspaces(signal, model_order)
    elif algorithm != "IQML":
        raise ValueError("Only IQML and MODE types are supported")
    d = model_order
    N = signal.shape[0]
    K = signal.shape[1]

    As = []

    for t in range(K):
        A = np.zeros([N - d, d + 1], dtype=complex)
        x = signal[:, t]
        for i in reversed(range(d + 1)):
            for j in range(N - d):
                A[j, d - i] = x[i + j]
        As.append(A)

    T = np.zeros([d + 1, d + 1], dtype=complex)

    if d % 2 != 0:
        for i in range(((d + 1) // 2)):
            T[i, 2 * i] = 1
            T[i, 2 * i + 1] = 1j
            T[((d + 1) // 2) + i, d - 1 - 2 * i] = 1
            T[((d + 1) // 2) + i, d - 2 * i] = -1j
    else:
        for i in range(d // 2):
            T[i, 2 * i] = 1
            T[i, 2 * i + 1] = 1j
            T[d - i, 2 * i] = 1
            T[d - i, 2 * i + 1] = -1j
        T[d // 2, d // 2 + 1] = 1

    T = T / np.sqrt(2)

    b_hat = np.zeros(d + 1)
    B = np.eye(N - d)
    for it in range(max_iter):

        C = np.zeros([d + 1, d + 1], dtype=complex)
        B_proj = np.linalg.inv(B.conj().T @ B)
        for t in range(K):
            C += As[t].conj().T @ B_proj @ As[t]

        Qx = T.conj().T @ C @ T

        w, v = np.linalg.eig(np.real(Qx))
        ind = np.argsort(w)
        v = v[:, ind]

        b_hat_next = T @ v[:, 0]

        if np.linalg.norm(b_hat - b_hat_next) < epsilon:
            break

        b_hat = b_hat_next
        B = np.zeros([N, N - d], dtype=complex)
        for i in range(N - d):
            B[i: i + d + 1, i] = np.conj(b_hat[::-1])

    rb = np_poly.polyroots(np.conj(b_hat))
    angle = np.arcsin(-np.angle(rb) / (2 * np.pi * separation)) * 180 / np.pi
    return tuple(angle)
