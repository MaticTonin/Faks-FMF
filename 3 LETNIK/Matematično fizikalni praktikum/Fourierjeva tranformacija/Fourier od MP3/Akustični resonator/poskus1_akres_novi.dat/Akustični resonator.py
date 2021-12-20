import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import time
import os
samples=44100 # 11025
sample=4
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(THIS_FOLDER,"poskus" + str(sample) + "_akres_novi.dat")
#fname = os.path.join(THIS_FOLDER,"poskus" + str(sample) + "_akres.txt")
#f = np.fromfile(fname, dtype = float) # za DAT
f=np.loadtxt(fname, dtype = float) # za TXT
print(f)
N=len(f)
print(N)
delta=N/samples
fc =samples/2.
freq = np.linspace(-fc, fc, N)
t=np.linspace(0, delta- (delta/N),N)

def Fourier(fx, N):
    fx=np.asarray(fx,dtype=complex)
    n=np.arange(N)
    k=n.reshape((N,1))
    coef=(np.exp(-2j * np.pi * k * n / N))
    H=np.dot(coef,fx)
    return H

def Fourier2(fx,N):
    fx=np.asarray(fx,dtype=float)
    H=[]
    for k in range(N):
        print(k)
        fk=0
        for n in range(N):
            Mnk=np.exp(-2j * np.pi * k * n / N)
            fk+=Mnk*fx[n]
        H.append(fk)
    return np.asarray(H, dtype=complex)

def InverseFourier(fx, N):
    fx=np.asarray(fx,dtype=complex)
    n=np.arange(N)
    k=n.reshape((N,1))
    coef=(np.exp(2j * np.pi * k * n / N))/N
    H=np.dot(coef,fx)
    return H 


plt.plot(t,f,'-',color="lime")
plt.title('Akustični resonator, predmet v resonatorju ('+str(sample)+")")
plt.xlabel("t")
plt.ylabel("f(x)")
plt.legend()
plt.show()


start1 = time.time()
#H=np.fft.fftshift(Fourier(f,N))
H=np.fft.fftshift(np.fft.fft(f))
end1 = time.time()

print(N)
print(end1-start1)

plt.plot(freq,H.real,'-',color="lime")
plt.title("Akustični resonator ("+ str(sample)+ ") vzorčenje: "+ str(samples)+ " Hz")
plt.xlabel("$\omega$")
plt.ylabel("Re[FT($\omega$)]")
plt.legend()
plt.show()

H_abs=2*abs(H)

plt.plot(freq,H_abs.real,'-',color="lime")
plt.title("Akustični resonator ("+ str(sample)+ ") vzorčenje: "+ str(samples)+ " Hz")
plt.xlabel("$\omega$")
plt.ylabel("Re[FT($\omega$)]")
plt.legend()
plt.show()
H_abs_2=H_abs**2
plt.plot(abs(freq),H_abs_2,'-',color="lime")
plt.title("Akustični resonator ("+ str(sample)+ ") vzorčenje: "+ str(samples)+ " Hz")
plt.xlabel("$\omega$")
plt.ylabel("Re[FT($\omega$)]")
plt.legend()
plt.show()
#imaginarni


plt.plot(freq,H.imag,'-',color="lime")
plt.title("Akustični resonator ("+ str(sample)+ ") vzorčenje: "+ str(samples)+ " Hz")
plt.xlabel("$\omega$")
plt.ylabel("Im[FT($\omega$)]")
plt.legend()
plt.show()

print(N)
print(end1-start1)