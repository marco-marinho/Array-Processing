import numpy as np
import math

#Generates 4 QAM symbols for given number of sources and snapshots with specified SNR
def gen_signal(sources, snapshots, SNR = np.inf):
    real = np.random.choice([1, -1], (sources, snapshots))
    imaginary = 1j*np.random.choice([1, -1], (sources, snapshots))
    symbols = np.add(real, imaginary)
    noisy_symbols = add_noise(symbols, SNR)
    return noisy_symbols

#Adds AWGN noise with specified SNR measuring clean signal power
def add_noise(signal, SNR):
    power = np.sum(np.square(np.absolute(signal)))/np.product(signal.shape)
    SNR_lin = 10**(SNR/10)
    N0 = power/SNR_lin
    noise = np.sqrt(N0/2) * np.add(np.random.randn(*signal.shape), 1j*np.random.randn(*signal.shape))
    noisy_signal = np.add(signal, noise)
    return noisy_signal
