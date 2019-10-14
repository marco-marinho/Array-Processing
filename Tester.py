import SignalGenerator as sig
import matplotlib.pyplot as plt

signal = sig.gen_signal(1, 100, 30)

fig, ax = plt.subplots()

ax.scatter(signal.real, signal.imag)

plt.show()

