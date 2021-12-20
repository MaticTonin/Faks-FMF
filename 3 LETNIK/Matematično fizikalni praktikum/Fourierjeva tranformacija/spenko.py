from numpy import fft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.ticker import LogLocator
import time

n = 1000

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1)) # in essence row->column transformation, k*n is then the dot product..
    #print "Vector",k
    #print "Matrix",k*n
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def IDFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1)) # in essence row->column transformation, k*n is then the dot product..
    #print "Vector",k
    #print "Matrix",k*n
    M = (np.exp(2j * np.pi * k * n / N))/N
    return np.dot(M, x)

x = np.linspace(-np.pi, np.pi, n)
f3 = np.exp(- x*x/0.015)


plt.figure()
plt.plot(x, f3)

plt.figure()
plt.plot(x, np.fft.ifftshift(f3))

fft1 = DFT_slow(f3)
#print(fft1.shape)
#print(fft1)
fft2 = DFT_slow(np.fft.ifftshift(f3))
#print(fft2.shape)

fc = 1 / (2*(2*np.pi/n))
freq = np.linspace(-fc, fc, n)
print(freq.shape)

plt.figure()
plt.plot(freq, fft1.real)
plt.figure()
plt.plot(freq, fft1.imag)

plt.figure()
plt.plot(freq, np.fft.fftshift(fft2).real)
plt.figure()
plt.plot(freq, np.fft.fftshift(fft2).imag)


ifft1 = IDFT_slow(fft1)
ifft2 = IDFT_slow(fft2)

plt.figure()
plt.plot(x, ifft1.real)
plt.figure()
plt.plot(x, ifft1.imag)

plt.figure()
plt.plot(x, ifft2.real)
plt.figure()
plt.plot(x, ifft2.imag)


plt.figure()
plt.plot(x, np.fft.fftshift(ifft2).real)
plt.figure()
plt.plot(x, np.fft.fftshift(ifft2).imag)

plt.show()

#Povzetek: če želimo dobiti "pravilne" funkcije tako v obeh prostorih moramo uporabljati fftshift.
#ifftshift preden damo v k-prostor, nato pa plotamo funkcije skupaj z fftshiftom.
#za inverz ne izvajamo dodatnih fftshiftov!

