import numpy as np

#Generates steering vectors for a ULA gor the given DOAs, number of elements and inner element separation
def generate_ula_vectors(angles, elements, separation = 1/2):
    angles_rad = np.asarray(angles) * np.pi / 180
    A = np.exp(np.matmul(np.atleast_2d(1j*2*np.pi*separation*np.arange(elements)).T, np.atleast_2d(np.sin(angles_rad))))
    return A

def generate_sparse_vectors(angles, positions, wavenumber):

    angles_rad = np.asarray(angles) * np.pi / 180
    A = 1j*np.zeros(shape=(len(positions), len(angles)))
    for column in range(len(A.T)):
        for position in range(len(positions)):
            x_component = positions[position][0]*np.cos(angles_rad[column])
            y_component = positions[position][1]*np.sin(angles_rad[column])
            A[position, column] = np.exp(1j*wavenumber*(x_component+y_component))

    return A
