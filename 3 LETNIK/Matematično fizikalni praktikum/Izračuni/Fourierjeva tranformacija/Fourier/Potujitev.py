import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import time
start1 = time.time()
end1 = time.time()
N=1000
delta=2*np.pi
fc = 1 / (2*(delta/N))
freq = np.linspace(-fc, fc, N)
freq1 = np.fft.fftfreq(N)
t=np.linspace(0, (delta- (delta/N)),N)
sin=np.sin(2*np.pi*t)
cos=np.cos(2*np.pi*t)

def Fourier(fx, N):
    fx=np.asarray(fx,dtype=complex)
    n=np.arange(N)
    k=n.reshape((N,1))
    coef=(np.exp(-2j * np.pi * k * n / N))
    H=np.dot(coef,fx)
    return H

def InverseFourier(fx, N):
    fx=np.asarray(fx,dtype=complex)
    n=np.arange(N)
    k=n.reshape((N,1))
    coef=(np.exp(2j * np.pi * k * n / N))/N
    H=np.dot(coef,fx)
    return H 

def Fourier2(fx,N):
    fx=np.asarray(fx,dtype=float)
    H=[]
    for k in range(N):
        fk=0
        for n in range(N):
            Mnk=np.exp(-2j * np.pi * k * n / N)
            fk+=Mnk*fx[n]
        H.append(fk)
    return np.asarray(H, dtype=complex)

def InverseFourier2(fx,N):
    fx=np.asarray(fx,dtype=float)
    H=[]
    for k in range(N):
        fk=0
        for n in range(N):
            Mnk=np.exp(-2j * np.pi * k * n / N)/N
            fk+=Mnk*fx[n]
        H.append(fk)
    return np.asarray(H, dtype=complex)

k=100
točke=0
int1=[]
int2=[]
velikost=[]
while k<2000:
    t1=np.linspace(0, (delta- (delta/k)),k)
    sin1=np.sin(t1)
    start1 = time.time()
    fftsin= np.fft.fftshift(Fourier(sin1,k))
    end1 = time.time()
    start2 = time.time()
    fftsin2= np.fft.fftshift(Fourier2(sin1,k))
    end2 = time.time()
    int1.append(end1-start1)
    int2.append(end2-start2)
    točke+=1
    print(točke)
    velikost.append(k)
    k+=100

točke=np.arange(točke)
plt.plot(velikost,int1,'-',color="red", label="Izdelava vektorjev")
plt.plot(velikost,int2,'-',color="blue", label="Izdelava vsote")
plt.title('Časovna zahtevnost glede na izbiro metode')
plt.xlabel("N")
plt.ylabel("t")
plt.legend()
plt.show()
start1 = time.time()
fftsin= np.fft.fftshift(Fourier(sin,N))
end1 = time.time()
start2 = time.time()
fftsin2= np.fft.fftshift(Fourier2(sin,N))
end2 = time.time()
print(end1-start1, end2-start2)
plt.plot(t,sin,'-',color="red")
plt.title('Funkcija sin(t)')
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.show()

plt.plot(t,(InverseFourier(np.fft.ifftshift(fftsin),N)).real,'-',color="red")
plt.title('Funkcija sin(t) po inverzni transformaciji')
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.show()

plt.plot(freq,(fftsin2).real,'-',color="red")
plt.title('FT za sin(x), realni del')
plt.xlabel("$\omega$")
plt.ylabel("$Re[FT]$")
plt.legend()
plt.show()

plt.plot(freq,(fftsin).real,'-',color="red")
plt.title('FT za sin(x), realni del, 2 način')
plt.xlabel("$\omega$")
plt.ylabel("$Re[FT]$")
plt.legend()
plt.show()

plt.plot(freq,(fftsin).imag, '-', color="red")
plt.title('FT za sin(x), imaginarni del')
plt.xlabel("$\omega$")
plt.ylabel("$Im[FT]$")
#plt.axis([-3,3, -0.6,0.6])
plt.legend()
plt.show()

#COSINUSNI DEL 
fftcos= np.fft.ifftshift(Fourier2(cos,N))
plt.plot(t,cos,'-',color="blue")
plt.title('Funkcija cos(t)')
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.show()

plt.plot(t,(InverseFourier(np.fft.ifftshift(fftcos),N)).real,'-',color="blue")
plt.title('Funkcija cos(t) po inverzni transformaciji')
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.show()


plt.plot(freq,(fftcos).real,'-',color="blue")
plt.title('FT za cos(x), realni del')
plt.xlabel("$\omega$")
plt.ylabel("$Re[FT]$")
#plt.axis([-3,3, -100,1400])
plt.legend()
plt.show()

plt.plot(freq,(fftcos).imag, '-', color="blue")
plt.title('FT za cos(x), imaginarni del')
plt.xlabel("$\omega$")
plt.ylabel("$Im[FT]$")
#plt.axis([-3,3, -0.6,0.6])
plt.legend()
plt.show()
