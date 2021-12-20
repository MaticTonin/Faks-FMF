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
N=1000
E=1
x=np.linspace(a,b,N)
icons=[a,b,y_a,y_b]
h=(b-a)/N
A=matrix_A(N)
def Y(x):
    return x-x**3/6

def f(t, y, E):
    return -E*y

def vector_V(N, y_a, y_b, a, b, E, Y_i):
    h=(b-a)/N
    h_2=h**2
    w=np.zeros(N)
    w[0]= h_2*f(a, Y_i[0], E)-y_a
    for i in range(1,N-2):
        w[i]=h_2*f(a+ i*h, Y_i[i], E)
    w[N-1]= h_2*f(a, Y_i[N-1], E)-y_b
    return w

def Piccard(A,N,icons,E, eps):
    a=icons[0]
    b=icons[1]
    y_a=icons[2]
    y_b=icons[3]
    x=np.linspace(a,b,N)
    Y_i=Y(x)
    Y_old=np.full(N, 1000)
    index=0
    while (norm(Y_old-Y_i)>=eps):
        Y_old=Y_i
        V_i=vector_V(N,y_a,y_b,a,b,E,Y_old)
        A_1=np.linalg.inv(A)
        Y_i=np.dot(V_i,A_1)
        if norm(Y_i[0])>10000 or index>3000:
            if norm(Y_i[0])>10000:
                print("DIVERGIRA")
            if index>300:
                print("KOKRAKI")
            print("You are retarded, divergira.")
            break
        index+=1
    return Y_i, index
Piacrd=Piccard(A,N,icons, E, 10**(-6))
psi=np.dot(A,Piacrd[0])
psi=psi/max(abs(psi))
print(Piacrd[1])
eig1, eig2 = np.linalg.eigh(A)
eig2 = -np.transpose(eig2)  
psi_main=eig2[len(eig2)-E]   
psi_main = psi_main/max(psi_main)
def Error(Y,value):
    Error=[]
    for i in range(len(Y)):
        Error.append(abs(Y[i]-value[i]))
    return Error
plt.plot(x,psi,'-', label=r"$\psi(x)$")
plt.plot(x,psi_main,'-', label=r"$\psi_{main}$")
plt.title(r'Prikaz približkov Picardovo iteracijo za $\psi(x)_0$')
plt.xlabel("t")
plt.ylabel(r"$\psi (t)$ ")
plt.legend()
plt.show()

plt.plot(x,Error(psi, psi_main),'-', label=r"Error $\psi(x)$")
plt.title(r'Prikaz napake Picardovo iteracijo za $\psi(x)_0$')
plt.xlabel("t")
plt.ylabel(r"$\psi (t)$ ")
plt.legend()
plt.show()
M=1000
while M<5000:
    Piacrd=Piccard(matrix_A(M),M,icons, E, 10**(-6))
    psi=np.dot(matrix_A(M),Piacrd[0])
    psi=psi/max(abs(psi))
    x_m=np.linspace(a,b,M)
    eig1, eig2 = np.linalg.eigh(matrix_A(M))
    eig2 = -np.transpose(eig2)  
    psi_main=eig2[len(eig2)-E]   
    psi_main = psi_main/max(psi_main)
    plt.plot(x_m,np.log(Error(psi, psi_main)),'-', label=r"N=" +str(M))
    M+=1000

plt.title(r'Prikaz napake Picardovo iteracijo za $\psi(x)_0$')
plt.xlabel("t")
plt.ylabel(r"$\psi (t)$ ")
plt.legend()
plt.show()
