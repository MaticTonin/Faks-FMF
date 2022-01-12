import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import time
import os
def Autocorelation1(Inverse): #naredi autokorelacijo
    N=len(Inverse)
    Auto=[]
    F=np.fft.ifftshift(Inverse)
    for i in range(N):
        Auto.append((1/N-i)*F[i])
    Auto=Auto[:-1]
    return Auto

def Podaljšanje(f):  #število podaljšanj
    f=np.asarray(f,dtype=float)
    N=len(f)
    h=np.zeros(N*2)
    for i in range(N):
            h[i]=f[i]
    return h

def Autocorelation1(Inverse): #naredi autokorelacijo
    N=len(Inverse)
    Auto=[]
    F=Inverse
    #F=np.fft.ifftshift(Inverse)
    for i in range(N):
        Auto.append((1/N-i)*F[i])
    return Auto

samples=44100
n=1000
N=500000
mesto=[]
cas1=[]
cas2=[]
while n<N:
    mesto.append(n)
    #f1=np.random.randint(1000, size=(n))
    x=np.linspace(0, 2*np.pi, n)
    f1=np.sin(x)
    start1 = time.time()
    f=Podaljšanje(f1) #podaljša podatke, da so dolžine 2N 
    H=np.fft.fft(f)  #da naredi
    H_2=np.absolute(H)**2
    F_inverse=np.fft.ifft(H_2)
    Auto=Autocorelation1(F_inverse)
    Spekt=np.absolute(np.fft.ifftshift(np.fft.fft(Auto)))**2
    end1 = time.time()
    start2 = time.time()
    H1=np.fft.fft(f1)  #da naredi
    H_21=np.fft.ifftshift((np.absolute(H1)**2))
    end2 = time.time()
    cas1.append(end1-start1)
    cas2.append(end2-start2)
    n+=1000




plt.plot(mesto,cas1,'.',color="red", label="Čas avtokorelacije")
plt.plot(mesto,cas2,'.',color="blue", label="Čas izdelave Fourierove transformacije")
plt.title("Hitrost izdelave autokorelacije v odvisnosti od točk sinusa")
plt.xlabel("n")
plt.ylabel("cas[s]")
plt.legend()
plt.show()