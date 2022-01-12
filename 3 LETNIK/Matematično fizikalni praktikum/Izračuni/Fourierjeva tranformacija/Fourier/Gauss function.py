import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import time
start1 = time.time()
end1 = time.time()
N=5000

fc = 1 / ((2*(2*np.pi/N)))
freq = np.linspace(-fc, fc, N)
freq1 = np.fft.fftfreq(N)
t=np.linspace(-40, 40,N)
gaussfunction=  lambda x: np.exp(-x**2 / 2) 
gaussfunction1=  lambda x: np.exp(-(x+30)**2 / 2) 
gaussfunction2=  lambda x: np.exp(-(x+5)**2 / 2) - np.exp(-(x-5)**2 / 2) 
gauss=gaussfunction(t)
gauss1=gaussfunction1(t)
gauss2=gaussfunction2(t)
gauss=np.roll(gauss, int(len(t)/2))
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
            Mnk=np.exp(-2j * np.pi * k * n / N)/N
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

start1 = time.time()
fftgauss= np.fft.ifftshift(Fourier(gauss,N))
end1 = time.time()
start2 = time.time()
#fftgauss2= np.fft.ifftshift(Fourier2(gauss,N))
end2 = time.time()
print(end1-start1, end2-start2)
plt.plot(t,gauss,'-',color="green")
plt.title('Gaussova funkcija rolled')
plt.xlabel("t")
plt.ylabel("f(t)")
#plt.axis([-10,10, -0.2,1.2])
plt.legend()
plt.show()

plt.plot(t,(InverseFourier(np.fft.ifftshift(fftgauss),N)).real,'-',color="green")
plt.title('Gaussova funkcija po inverzni transformaciji')
plt.xlabel("t")
plt.ylabel("f(t)")
plt.axis([-10,10, -0.2,1.2])
plt.legend()
plt.show()

plt.plot(t,(InverseFourier(np.fft.ifftshift(fftgauss),N)).imag,'-',color="green")
plt.title('Gaussova funkcija po inverzni transformaciji')
plt.xlabel("t")
plt.ylabel("f(t)")
#plt.axis([-10,10, -0.2,1.2])
plt.legend()
plt.show()

plt.plot(freq,(fftgauss).real,'-',color="green")
plt.title('FT za Gaussovo funkcijo N=7000, realni del')
plt.xlabel("$\omega$")
plt.ylabel("$Re[FT]$")
plt.axis([-15,15, -200,200])
plt.legend()
plt.show()


plt.plot(freq,(fftgauss).imag, '-', color="green")
plt.title('FT za Gaussovo funkcijo N=7000, imaginarni del')
plt.xlabel("$\omega$")
plt.ylabel("$Im[FT]$")
plt.axis([-10,10, -1,1])
plt.legend()
plt.show()

#Premaknjen gauss
fftgauss1= np.fft.ifftshift(Fourier(gauss1,N))
plt.plot(t,gauss1,'-',color="orange")
plt.title('Gaussova funkcija premaknjena za x=30 iz izhodišča')
plt.xlabel("t")
plt.ylabel("f(t)")
plt.axis([-20,20, -0.2,1.2])
plt.legend()
plt.show()

plt.plot(t,(InverseFourier(np.fft.ifftshift(fftgauss1),N)).real,'-',color="orange")
plt.title('Gaussova funkcija premaknjena x=30, po inverzni')
plt.xlabel("t")
plt.ylabel("f(t)")
plt.axis([-20,20, -0.2,1.2])
plt.legend()
plt.show()

plt.plot(t,(InverseFourier(np.fft.ifftshift(fftgauss1),N)).imag,'-',color="orange")
plt.title('Gaussova funkcija premaknjena x=30, po inverzni')
plt.xlabel("t")
plt.ylabel("f(t)")
#plt.axis([-20,20, -0.2,1.2])
plt.legend()
plt.show()

plt.plot(freq,(fftgauss1).real,'-',color="orange")
plt.title('FT za premaknjeno Gaussovo funkcijo (x=30), realni del')
plt.xlabel("$\omega$")
plt.ylabel("$Re[FT]$")
plt.axis([-10,10, -200,200])
plt.legend()
plt.show()


plt.plot(freq,(fftgauss1).imag, '-', color="gray")
plt.title('FT za premaknjeno Gaussovo funkcijo (x=30), imaginarni del')
plt.xlabel("$\omega$")
plt.ylabel("$Im[FT]$")
plt.axis([-10,10, -200,200])
plt.legend()
plt.show()

#Dva gaussa
fftgauss2= np.fft.ifftshift(Fourier(gauss2,N))
plt.plot(t,gauss2,'-',color="gray")
plt.title('Dve Gaussovi funkciji')
plt.xlabel("t")
plt.ylabel("f(t)")
plt.axis([-20,20, -1.2,1.2])
plt.legend()
plt.show()

plt.plot(t,(InverseFourier(np.fft.ifftshift(fftgauss2),N)).real,'-',color="gray")
plt.title('Dve Gaussovi funkciji po inverzni')
plt.xlabel("t")
plt.ylabel("f(t)")
plt.axis([-20,20, -1.2,1.2])
plt.legend()
plt.show()

plt.plot(t,(InverseFourier(np.fft.ifftshift(fftgauss2),N)).imag,'-',color="gray")
plt.title('Dve Gaussovi funkciji po inverzni imaginarni')
plt.xlabel("t")
plt.ylabel("f(t)")
#plt.axis([-20,20, -1.2,1.2])
plt.legend()
plt.show()


plt.plot(freq,(fftgauss2).real,'-',color="gray")
plt.title('FT za dve Gaussovi funkciji, realni del')
plt.xlabel("$\omega$")
plt.ylabel("$Re[FT]$")
plt.axis([-10,10, -2,2])
plt.legend()
plt.show()


plt.plot(freq,(fftgauss2).imag, '-', color="gray")
plt.title('FT za dve Gaussovi funkciji, imaginarni del')
plt.xlabel("$\omega$")
plt.ylabel("$Im[FT]$")
plt.axis([-10,10, -350,350])
plt.legend()
plt.show()