from numpy import fft
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.ticker import LogLocator
import time

n = 1000
n1 = 10000
delta = 2*np.pi

fc = 1 / (2*(delta/n))
freq = np.linspace(-fc, fc, n)
freq1 = np.fft.fftfreq(n)
print(freq)
print(fc)
print(freq1)
print(freq1[n-1])

x = np.linspace(0, delta - (delta/n), n)
#x1 = np.linspace(0, delta - (delta/n1), n1)
#f = np.sin(x)
#f = np.cos(x)
v = 3
f = np.sin(2*np.pi*v*x) 
#f1 = np.sin(2*np.pi*v*x1)
#Spectral leakage, 
#Aliasing,

#f = np.sin(x - 2)
#Translacija cos/sin ustvari kombinacijo obeh? 
#Zakaj ne opazimo exp(2pi*j*a*w)?

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

plt.figure()
plt.title("Sin(2$\pi$" + str(round(v, 2)) + "x)")
plt.ylabel("f(x)")
txt = "Primer neperiodične funkcije na danem intervalu!"
plt.xlabel("x\n" + txt)
plt.plot(x, f)

#funkcijo shiftamo preden damo v fourierov prostor
#fft1 = np.fft.ifftshift(DFT_slow(np.fft.fftshift(f)))
#plt.figure()
#plt.plot(freq, (fft1).real)
#plt.figure()
#plt.plot(freq, (fft1).imag)

#funkcijo neshiftamo (ne rabimo ker je izhodišče že na levem robu)
fft2 = np.fft.ifftshift(DFT_slow(f))
plt.figure()
plt.title("Realna komponenta DFT")
plt.ylabel("Re|F($\omega$)|")
txt = "'Spectral leakge'. Frekvenca signala ni večkratnik osnovne frekvence!\n Dobimo tudi realno komponento v spektru, čeprav imamo opravka s sinusom."
plt.xlabel("$\omega$\n" + txt)
plt.plot(freq, (fft2).real)
plt.figure()
plt.title("Imaginarna komponenta DFT")
plt.ylabel("Im|F($\omega$)|")
txt = "'Spectral leakge'. Frekvenca signala ni večkratnik osnovne frekvence!"
plt.xlabel("$\omega$\n" + txt)
plt.plot(freq, (fft2).imag)

#POMEMBNO: Fourierova transformacija razporedi frekvence od 0 do fc, nato pa od -fc do 0.
#zato moramo vedno pri prikazovanju spektra ishiftat, da dobimo interval frekvenc od -fc do fc.

#spekter nazaj shiftamo pred ifft
ifft1 = IDFT_slow(np.fft.fftshift(fft2))
plt.figure()
plt.title("IDFT")
plt.ylabel("f(x)")
txt = "Kot vidimo smo po inverzni transformaciji dobili isti signal!"
plt.xlabel("x\n" + txt)
plt.plot(x, ((ifft1).real))
#plt.figure()
#plt.plot(x, ((ifft1).imag))

#spekter ne shiftamo nazaj pred ifft
#ifft2 = IDFT_slow(fft1)
#plt.figure()
#plt.plot(x, np.fft.ifftshift((ifft2).real))
#plt.figure()
#plt.plot(x, np.fft.ifftshift((ifft2).imag))

plt.show()