import numpy as np
import scipy.signal as sig
import math
import matplotlib.pyplot as plt

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

def generate_OFDM_signal(snapshots, subcarriers, channel, cyclic_prefix_lenght, SNR, plot_sent = False, plot_received = False):

    #Calculate number of OFDM data frames
    frame_number = int(snapshots/subcarriers)

    #Generate random polar signaling data
    s_data = np.random.choice([1, -1], snapshots) + 1j*np.random.choice([1, -1], snapshots)
    print(np.shape(s_data))
    if plot_sent:
        X = [x.real for x in s_data]
        Y = [x.imag for x in s_data]
        plt.scatter(X, Y, color='blue')
        plt.show()

    #FIR channel in frequency domain
    hf = np.fft.fft(channel, subcarriers)

    #S/P conversion
    p_data = np.reshape(s_data, (subcarriers, frame_number))

    #Convert data to time domain
    p_td = np.fft.ifft(p_data)

    #Get cyclic prefix of given lenght
    cyclic_prefix = p_td[-cyclic_prefix_lenght:, :]

    #Data frames with cyclic prefix
    p_cyc = np.concatenate((cyclic_prefix, p_td), axis=0)

    #P/S conversion
    s_cyc = np.reshape(p_cyc, ((subcarriers+cyclic_prefix_lenght)*frame_number, 1))

    #Pass data through channel
    chs_out = sig.lfilter(channel, 1, s_cyc, axis=0)

    #Add noise
    x_out = chs_out#add_noise(chs_out, SNR)

    #P/S conversion and cyclic prefix removal
    x_para = np.reshape(x_out, (subcarriers+cyclic_prefix_lenght, frame_number))
    x_disc = x_para[cyclic_prefix_lenght:, :]

    #FFT to F domain
    x_hat_para = np.fft.fft(x_disc)

    z_data = np.matmul(np.linalg.inv(np.diag(hf)), x_hat_para)

    if plot_received:
        X = [x.real for x in z_data[10,:]]
        Y = [x.imag for x in z_data[10,:]]
        plt.scatter(X, Y, color='red')
        plt.show()
