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
N=2000
mesto=[]
cas=[]
while n<N:
    mesto.append(n)
    f1=np.random.randint((-100,100), size=(n))
    f=Podaljšanje(f1) #podaljša podatke, da so dolžine 2N 
    H=np.fft.fft(f)  #da naredi
    H_2=np.absolute(H)**2
    start1 = time.time()
    F_inverse=np.fft.ifft(H_2)
    Auto=Autocorelation1(F_inverse)
    Spekt=np.absolute(np.fft.ifftshift(np.fft.fft(Auto)))**2
    end1 = time.time()
    cas.append(end1-start1)
    n+=1000
print(mesto)



plt.plot(mesto,cas,'-',color="red")
#plt.plot(t,Auto/max(Auto),'-',color="blue")
plt.title("Podatki za posnetek, 41100 Hz")
plt.xlabel("t")
plt.ylabel("f(x)")
plt.legend()
plt.show()