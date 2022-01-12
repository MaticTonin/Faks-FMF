from numpy import fft
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.ticker import LogLocator
import time

N=10 # št intervalov
delta=2*np.pi
fc=1/(2*delta/N)  #Nyquisova frekvenca

frekvenca=np.linspace(-fc,fc,N)
freq1 = np.fft.fftfreq(N) #python preuredi frekvence tako da imamo pravilno razdeljene za fc
print(frekvenca)
print(freq1)
x = np.linspace(0, delta - (delta/N), N) #x os po kateri printamo
#print(x)


h=np.sin(delta *x)

def Fourierjev(f, N):
    f=np.asarray(f, dtype=complex) # spremeniš v kompleksen array
    n = np.arange(N) # naredi array velikosti N
    k = n.reshape((N, 1)) # ga spremeni v vektor
    M = np.exp(2j * np.pi * k * n / N)/N #sestavi matriko koeficientov eksponenta za posamezne k in n
    return np.dot(M,f) #naredi Fouriejev koeficient
a=Fourierjev(h, N)
print(a)
B=np.fft.ifftshift(a)

plt.figure()
plt.plot(frekvenca, (B).real)
plt.title('Odvisnost lastnih energij od metode pri $\lambda$=0.5')
plt.xlabel("$N$")
plt.ylabel("$E_(n)$")
plt.legend()
plt.show()

def Fourier_coef(k, n, N):
    exp=np.exp(2*np.pi*1j*k*n/N)
    return exp

def Fourier_razvoj(k, n, N, function, t):
    Hn=0
    for i in range(len(t)):
        Hn=np.sin(t[i])*Fourier_coef(k,n,N)
    return 1100



#plt.figure()
#plt.plot(x, ((ifft1).imag))

#spekter ne shiftamo nazaj pred ifft
#ifft2 = IDFT_slow(fft1)
#plt.figure()
#plt.plot(x, np.fft.ifftshift((ifft2).real))
#plt.figure()
#plt.plot(x, np.fft.ifftshift((ifft2).imag))

plt.show()

def hn(x):
    funkcija=np.zeros(len(x))
    for i in range(N):
        funkcija[i]=np.sin(2*np.pi*x[i])
    return funkcija

def Hn(N, n):
    H_nčlen=0+0j
    for i in range(N):
        H_nčlen+= (np.exp(2j * np.pi * i * n / N))/N
    return H_nčlen

def H(N, x):
    H=np.zeros(N)
    funkcija=hn(x)
    for i in range(N):
        H[i]=(Hn(N, i))*funkcija[i]
    return H

print(H(10,x))
print(a)