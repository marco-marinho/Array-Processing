import numpy as np
import tensorly.tenalg as tls

#Generates steering vectors for a ULA gor the given DOAs, number of elements and inner element separation
def generate_ula_vectors(angles, elements, separation = 1/2):
    angles_rad = np.asarray(angles) * np.pi / 180
    A = np.exp(np.atleast_2d(-1j*2*np.pi*separation*np.arange(elements)).T @ np.atleast_2d(np.sin(angles_rad)))
    return A

def generate_ula_vectors_center(angles, elements, separation = 1/2):
    angles_rad = np.asarray(angles) * np.pi / 180
    if elements%2 == 0:
        A_1 = np.exp(
            np.atleast_2d(1j * 2 * np.pi * ((0.5+np.arange(elements / 2)) * separation)).T @ np.atleast_2d(
                np.sin(angles_rad)))
        A_2 = np.exp(
            np.atleast_2d(-1j * 2 * np.pi * ((0.5+np.arange(elements / 2)) * separation)).T @ np.atleast_2d(
                np.sin(angles_rad)))
        A = np.vstack((np.flipud(A_1), A_2))

    else:
        A_1 = np.exp(
            np.atleast_2d(1j * 2 * np.pi * ((1+np.arange(int(elements / 2))) * separation)).T @ np.atleast_2d(
                np.sin(angles_rad)))
        A_2 = np.exp(
            np.atleast_2d(-1j * 2 * np.pi * ((1+np.arange(int(elements / 2))) * separation)).T @ np.atleast_2d(
                np.sin(angles_rad)))
        A = np.vstack((np.flipud(A_1), np.ones((1, np.shape(A_1)[1])), A_2))

    return A

def generate_sparse_vectors(angles, positions, wavenumber):
    angles_rad = np.asarray(angles) * np.pi / 180
    A = 1j*np.zeros(shape=(len(positions), len(angles_rad)))
    for column in range(len(A.T)):
        for position in range(len(positions)):
            x_component = positions[position][0]*np.cos(angles_rad[column])
            y_component = positions[position][1]*np.sin(angles_rad[column])
            A[position, column] = np.exp(1j*wavenumber*(x_component+y_component))

    return A

def generate_polarization_steering(angles, E_ref):
    angles_rad = np.asarray(angles) * np.pi / 180
    u = np.zeros((2, len(angles)))

    for angle in range(len(angles)):

        h = np.cos(angles_rad[angle]) - np.sqrt(E_ref - np.sin(angles_rad[angle])**2) / \
        np.cos(angles_rad[angle]) + np.sqrt(E_ref - np.sin(angles_rad[angle]) ** 2)

        v = E_ref * np.cos(angles_rad[angle]) - np.sqrt(E_ref - np.sin(angles_rad[angle])**2) / \
        E_ref * np.cos(angles_rad[angle]) + np.sqrt(E_ref - np.sin(angles_rad[angle]) ** 2)

        u[:, angle] = [h, v]

    return u

def merge_space_polarization_steering(A, u):

    Z = tls.khatri_rao([A, u])

    return Z