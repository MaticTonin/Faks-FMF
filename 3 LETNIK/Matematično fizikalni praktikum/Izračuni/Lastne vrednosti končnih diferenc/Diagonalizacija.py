import numpy as np
from numpy.linalg.linalg import norm
from numpy import sin, cos, pi, array
import numpy as np
import scipy.integrate as integrate
import matplotlib
import scipy
import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import time
import os
def matrix_A(N):
    A=np.zeros((N,N))
    for i in range(1,N-1):
        A[i,i-1]=1
        A[i,i]=-2
        A[i,i+1]=1
    #Ker more biti matrika v vsakem primeru tridiagonalna za neskončno jamo
    A[0,0] = -2
    A[0, 1] = 1

    A[N-1, N-2] = 1
    A[N-1, N-1] = -2
    return A

a=-np.pi/2
b=np.pi/2
y_a=0
y_b=0
N=2000
E=1
x=np.linspace(a,b,N)
icons=[a,b,y_a,y_b]
h=(b-a)/N

eig1, eig2 = np.linalg.eigh(matrix_A(N))
eig2 = -np.transpose(eig2)
eig1=eig1[::-1]/h**2
print(eig1) 
psi=eig2[len(eig2)-E-1]   
psi = psi/max(psi)

psi_main0=np.sin((1+E)*x+np.pi/2)
psi_main1=np.sin((1+E)*x)
psi_main2=np.sin((1+E)*x+np.pi/2)
psi_main3=-np.sin((1+E)*x)
psi_main4=np.sin((1+E)*x+np.pi/2)
plt.plot(x,psi,'-', label=r"$\psi_0$")
plt.plot(x,psi_main3,'-', label=r"$\psi_{main}$")
plt.title(r'Prikaz približkov Diagonalizacijo iteracijo za $\psi(x)_0$')
plt.xlabel("t")
plt.ylabel(r"$\psi (t)$ ")
plt.legend()
plt.show()

def Error(Y,value):
    Error=[]
    for i in range(len(Y)):
        Error.append(abs(abs(Y[i])-abs(value[i])))
    return Error

n=0
"""while n<6:
    if n%2==0:
        eig1, eig2 = np.linalg.eigh(matrix_A(N))
        eig2 = -np.transpose(eig2)  
        psi=eig2[len(eig2)-n-1]  
        psi = psi/max(psi)
        psi_main=np.sin((1+n)*x+np.pi/2)
        plt.plot(x,np.log(Error(psi,psi_main)),'-', label=r"$\psi$"+str(n))
    if n%2==1:
        eig1, eig2 = np.linalg.eigh(matrix_A(N))
        eig2 = -np.transpose(eig2)  
        psi=eig2[len(eig2)-n-1]  
        psi = psi/max(psi)
        psi_main=np.sin((1+n)*x)
        plt.plot(x,np.log(Error(psi,psi_main)),'-', label=r"$\psi$"+str(n))
    n+=1

plt.title(r'Prikaz napake Diagonalizacije za $\psi(x)$ pri različnih N')
plt.xlabel("x")
plt.ylabel(r"Error($\psi (x)$) ")
plt.legend()
plt.show()

n=0
while n<6:
    if n%2==0:
        eig1, eig2 = np.linalg.eigh(matrix_A(N))
        eig2 = -np.transpose(eig2)
        eig1=-eig1[::-1]/h**2 
        lamb=[]
        for i in range(N):
            lamb.append(i)
        psi=eig2[len(eig2)-n-1]  
        psi = psi/max(psi)
        psi_main=np.sin((1+n)*x+np.pi/2)
        plt.plot(lamb,np.log(abs((1+n)**2-abs(eig1))),'-', label=r"$\psi$"+str(n))
    if n%2==1:
        eig1, eig2 = np.linalg.eigh(matrix_A(N))
        eig2 = -np.transpose(eig2) 
        lamb=[]
        for i in range(N):
            lamb.append(i)
        eig1=-eig1[::-1]/h**2 
        psi=eig2[len(eig2)-n-1]  
        psi = psi/max(psi)
        psi_main=np.sin((1+n)*x)
        plt.plot(lamb,np.log(abs((1+n)**2-abs(eig1))),'-', label=r"$\psi$"+str(n))
    n+=1

plt.title(r'Prikaz napake Diagonalizacije za $E(x)$ pri različnih N')
plt.xlabel(r"$\lambda$")
plt.ylabel(r"Error($E (x)$) ")
plt.legend()
plt.show()

n=0
while n<6:
    if n%2==0:
        eig1, eig2 = np.linalg.eigh(matrix_A(N))
        eig2 = -np.transpose(eig2)  
        psi=eig2[len(eig2)-n-1] 
        eig1=-eig1[::-1]/h**2  
        psi = psi/max(psi)
        psi_main=np.sin((1+n)*x+np.pi/2)
        plt.plot(x,psi+eig1[n],'-', label=r"$\psi$"+str(n))
    if n%2==1:
        eig1, eig2 = np.linalg.eigh(matrix_A(N))
        eig2 = -np.transpose(eig2) 
        eig1=-eig1[::-1]/h**2 
        psi=eig2[len(eig2)-n-1]  
        psi = psi/max(psi)
        psi_main=np.sin((1+n)*x)
        plt.plot(x,psi+eig1[n],'-', label=r"$\psi$"+str(n))
    n+=1

plt.title(r'Prikaz Diagonalizacije za $\psi(x)$ pri različnih N')
plt.xlabel("x")
plt.ylabel(r"$\psi (x)$")
plt.legend()
plt.show()"""
N=200
n=0
lamb=[]
vrednost=[]
cas=[]
while N<4000:
        start1 = time.time()
        eig1, eig2 = np.linalg.eigh(matrix_A(N))
        eig2 = -np.transpose(eig2) 
        lamb.append(N)
        eig1=-eig1[::-1]/h**2 
        psi=eig2[len(eig2)-n-1]  
        psi = psi/max(psi)
        vrednost.append(np.log(abs((1+n)**2-abs(eig1[0]))))
        psi_main=np.sin((1+n)*x)
        end1 = time.time()
        cas.append(end1-start1)
        N+=200
plt.plot(lamb,vrednost,'.', label=r"$\psi$"+str(n))
plt.title(r'Prikaz napake Diagonalizacije za $E(x)_0$ pri različnih N')
plt.xlabel(r"$N$")
plt.ylabel(r"Error($E (x)$) ")
plt.legend()
plt.show()

plt.plot(lamb,cas,'.', label=r"$\psi$"+str(n))
plt.title(r'Prikaz časovne zahtevnosti Diagonalizacije za $E(x)_0$ pri različnih N')
plt.xlabel(r"$N$")
plt.ylabel(r"t")
plt.legend()
plt.show()