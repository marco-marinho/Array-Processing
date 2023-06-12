import numpy as np


# Generates steering vectors for a ULA gor the given DOAs, number of elements and inner element separation
def generate_ula_vectors(angles, elements, separation=1 / 2):
    angles_rad = np.asarray(angles) * np.pi / 180
    A = np.exp(np.atleast_2d(-1j * 2 * np.pi * separation * np.arange(elements)).T @ np.atleast_2d(np.sin(angles_rad)))
    return A


def generate_ula_vectors_center(angles, elements, separation=1 / 2):
    angles_rad = np.asarray(angles) * np.pi / 180
    if elements % 2 == 0:
        A_1 = np.exp(
            np.atleast_2d(1j * 2 * np.pi * ((0.5 + np.arange(elements / 2)) * separation)).T @ np.atleast_2d(
                np.sin(angles_rad)))
        A_2 = np.exp(
            np.atleast_2d(-1j * 2 * np.pi * ((0.5 + np.arange(elements / 2)) * separation)).T @ np.atleast_2d(
                np.sin(angles_rad)))
        A = np.vstack((np.flipud(A_1), A_2))

    else:
        A_1 = np.exp(
            np.atleast_2d(1j * 2 * np.pi * ((1 + np.arange(int(elements / 2))) * separation)).T @ np.atleast_2d(
                np.sin(angles_rad)))
        A_2 = np.exp(
            np.atleast_2d(-1j * 2 * np.pi * ((1 + np.arange(int(elements / 2))) * separation)).T @ np.atleast_2d(
                np.sin(angles_rad)))
        A = np.vstack((np.flipud(A_1), np.ones((1, np.shape(A_1)[1])), A_2))

    return A
