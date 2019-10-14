import numpy as np

#Generates steering vectors for a ULA gor the given DOAs, number of elements and inner element separation
def generate_ula_vectors(angles, elements, separation = 1/2):
    angles_rad = np.asarray(angles) * np.pi / 180
    A = np.exp(np.matmul(np.atleast_2d(1j*2*np.pi*separation*np.arange(elements)).T, np.atleast_2d(np.sin(angles_rad))))
    return A