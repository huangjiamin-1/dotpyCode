import numpy as np
import matplotlib.pyplot as plt

T = 10e-6  # pulse width
B = 25e6   # signal bandwidth
K = B/T   # frequency deviation
Fs = 2*B  # sampling frequency
Ts = 1/Fs # sampling interval
N = int(T/Ts)  # number of samples
t = np.linspace(-T/2, T/2, N)
f0 = 10e6  # carrier frequency
St = np.exp(1j*np.pi*K*t**2 + 1j*2*np.pi*f0*t)  # generate linear frequency modulated signal
FFTST = np.fft.fft(St,N)

plt.figure(1)
plt.plot(t, np.abs(FFTST))
plt.show()

plt.figure(2)
ref = np.exp(1j*np.pi*K*t**2 + 1j*2*np.pi*f0*t)
FFTREF = np.fft.fft(ref,N)
plt.plot(t, np.abs(FFTREF))
plt.show()

plt.figure(3)
RES = 20*np.log10(np.abs((np.fft.fftshift(np.fft.ifft(FFTREF*np.conjugate(FFTST),1024)))))
plt.plot(RES)
plt.show()


