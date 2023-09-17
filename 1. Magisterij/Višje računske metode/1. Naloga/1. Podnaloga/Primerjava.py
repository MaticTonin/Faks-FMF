import scipy.integrate as integrate
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.core.numeric import identity
import scipy
from scipy import special
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from tqdm import tqdm
import os
import matplotlib.cm as cm
import os
cm = plt.cm.jet
cm1 = plt.cm.winter
cm2 = plt.cm.autumn
L=10
h=0.10
tau=2*np.pi*h**2*0.1
x=np.arange(-L, L, h)
T=tau*len(x)*15
print(T)
t=np.linspace(0,T,len(x))
N=len(x)
M=N
print(N)
tau=t[1]-t[0]
alpha=1
lam=0
stanje=0
gamma=0



####### PLOTS TO SHOW #######
plot_3D=True
plot_Matrix=None
plot_animation=True
plot_lambda=None

def Potencial(x,lambd):
    return 1/2*x**2+lambd*x**4

def A(M,tau,h,lambd):
    A=np.zeros((int(M),int(N)), dtype=complex)
    A[0,0]=-5/2
    A[1,1]=-5/2
    A[0,1]=4/3
    A[1,0]=4/3
    for m in range(2,M):
        A[m,m]=-5/2-2*h**2*(Potencial(x[m],lambd))
        A[m,m-1]=4/3
        A[m-1,m]=4/3
        A[m,m-2]=-1/12
        A[m-2,m]=-1/12
    A_plus=np.identity(M)-(1j*tau)/(4*h**2)*A
    A_minus=np.identity(M)+(1j*tau)/(4*h**2)*A
    return A_plus, A_minus
def H_tri(M,tau,h,lambd):
    H=np.zeros((int(M),int(N)))
    H[0,0]=1
    for m in range(1,M):
        H[m,m]=1+h**2*(Potencial(x[m],lambd))
        H[m,m-1]=-1/2
        H[m-1,m]=-1/2
    H=1/(h**2)*H
    return H
def H_pet(M,tau,h,lambd):
    H=np.zeros((int(M),int(N)), dtype=complex)
    H[0,0]=-5/2
    H[1,1]=-5/2
    H[0,1]=4/3
    H[1,0]=4/3
    for m in range(2,M):
        H[m,m]=-5/2-2*h**2*(Potencial(x[m],lambd))
        H[m,m-1]=4/3
        H[m-1,m]=4/3
        H[m,m-2]=-1/12
        H[m-2,m]=-1/12
    H=1/(h**2)*H
    return H

def propagator(H_matrix,psi_0):
    N=len(A_plus[0,:])
    Wawe=[]
    solution=np.array(psi_0, dtype=complex)
    for i in tqdm(range(N)):
        Wawe.append(solution.copy())
        #solution=scipy.linalg.solve_banded()
        H_k=np.identity(N)
        for k in range(2):
            k+=1
            H_k=np.dot(H_k,H_matrix)
            solution+=(-1j*tau)**k/np.math.factorial(k)*np.dot(H_k,psi_0)
        psi_0=solution
    return np.asarray(Wawe, dtype=complex)

def lastne_f(x, stanje):
    funkcija_m=[]
    for i in range(len(x)):
        #funkcija_m.append((alpha/(np.pi)**(1/2))**(1/2)*np.exp(-alpha**2*(x[i]-gamma)**2/2)) 
        funkcija_m.append(alpha**(1/2)*(2. ** stanje  * np.math.factorial(stanje) * np.pi ** 0.5) ** (-0.5) *np.exp(-alpha**2*(x[i])**(2)/2) * special.hermite(stanje,0)(x[i]))
    return funkcija_m

def main(A_plus, A_minus,psi_0):
    N=len(A_plus[0,:])
    Wawe=[]
    solution=np.array(psi_0, dtype=complex)
    for i in range(N):
        Wawe.append(solution)
        #solution=scipy.linalg.solve_banded()
        solution=np.linalg.solve(A_plus,np.dot(A_minus,psi_0))
        psi_0=solution
    return np.asarray(Wawe, dtype=complex)


A_plus, A_minus=A(M,tau,h,lam)
H_matrix=H_pet(M,tau,h,lam)
Wawe0=main(A_plus, A_minus,lastne_f(x,0))
Wawe2=propagator(H_matrix,lastne_f(x,0))
psi_2=[]
psi_2_prop=[]
for i in range(N):
    psi_2.append(max(Wawe0[i].real))
    psi_2_prop.append(max(Wawe2[i].real))
plt.title(r"max$(\Psi(x,t))$, $\lambda=%.2f, \tau=%.4f, h=%.3f$" %(lam,tau,h))
plt.plot(t,psi_2, label="Implicitna")
#plt.plot(t,psi_2_prop, label="Propagator")
plt.xlabel("t")
plt.ylabel(r"log(max($\Psi$))")
#plt.yscale("log")
plt.legend()
plt.show()

psi_2=[]
psi_2_prop=[]
middle=[]
for i in range(N):
    psi_2.append(max(abs(Wawe0[i])))
    psi_2_prop.append(abs((Wawe2[i])))
    k=0
    for j in range(N):
        if abs(x[j])<0.001 and k==0:
            middle.append(Wawe0[i,j])
            k+=1
middle=np.array(middle)
fig = plt.figure()
ax1 = fig.add_subplot(211)
fig.suptitle(r"$\lambda=%.2f, \tau=%.4f, h=%.3f$" %(lam,tau,h))
ax1.plot(t,psi_2, label=r"max$(|\Psi(x,t)|$)")
ax2 = fig.add_subplot(212)
ax2.plot(t,middle.imag, label=r"max(Imag($\Psi$))")
ax2.plot(t,middle.real, label=r"max(Real($\Psi$))")
#plt.plot(t,psi_2_prop, label="Propagator")
ax1.set_xlabel("t")
ax2.set_xlabel("t")
ax1.set_ylabel(r"log(max($\Psi$))")
ax1.set_yscale("log")
ax2.set_ylabel(r"max($\Psi$)")
ax1.legend()
ax2.legend()
plt.show()


"""
fig = plt.figure()
fig.suptitle(r"\tau=%.4f, h=%.3f$" %(tau,h))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224)
lambda_list=[0,0.2,0.5,1]
color_index=0
for lam in lambda_list:
    A_plus, A_minus=A(M,tau,h,lam)
    H_matrix=H_pet(M,tau,h,lam)
    Wawe0=main(A_plus, A_minus,lastne_f(x,0))
    Wawe2=propagator(H_matrix,lastne_f(x,0))
    psi_2=[]
    psi_2_prop=[]
    middle=[]
    for i in range(N):
        psi_2.append(max(abs(Wawe0[i])))
        psi_2_prop.append(abs((Wawe2[i])))
        k=0
        for j in range(N):
            if abs(x[j])<0.001 and k==0:
                middle.append(Wawe0[i,j])
                k+=1
    middle=np.array(middle)
    ax1.plot(t,psi_2, label=r"$\lambda=%.2f$" %(lam))
    ax2.plot(t,middle.imag, label=r"$\lambda=%.2f$" %(lam), color=cm2(color_index/(len(lambda_list))))
    ax3.plot(t,middle.real, label=r"$\lambda=%.2f$" %(lam),  color=cm1(color_index/(len(lambda_list))))
    color_index+=1
ax1.set_xlabel("t")
ax2.set_xlabel("t")
ax1.set_title(r"max($|\Psi(x,t)$|")
ax2.set_title(r"max(Real($\Psi))$")
ax3.set_title(r"max(Imag($\Psi))$")
ax1.set_ylabel(r"log(max($\Psi$))")
ax1.set_yscale("log")
ax2.set_ylabel(r"max($\Psi$)")
ax3.set_ylabel(r"max($\Psi$)")
ax1.legend()
ax2.legend()
ax3.legend()
plt.show() """



fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224)
lambda_list=[0,1,2,3]
color_index=0

fig.suptitle(r"$\lambda=%.2f, \tau=%.4f, h=%.3f$" %(1,tau,h))
for lam in lambda_list:
    A_plus, A_minus=A(M,tau,h,1)
    H_matrix=H_pet(M,tau,h,1)
    Wawe0=main(A_plus, A_minus,lastne_f(x,lam))
    Wawe2=propagator(H_matrix,lastne_f(x,lam))
    psi_2=[]
    psi_2_prop=[]
    middle=[]
    for i in range(N):
        psi_2.append(max(abs(Wawe0[i])))
        psi_2_prop.append(abs((Wawe2[i])))
        k=0
        for j in range(N):
            if abs(x[j])<0.001 and k==0:
                middle.append(Wawe0[i,j])
                k+=1
    middle=np.array(middle)
    ax1.plot(t,psi_2, label=r"$\Psi_{%.i}$" %(lam))
    ax2.plot(t,middle.imag, label=r"$\Psi_{%.i}$" %(lam), color=cm2(color_index/(len(lambda_list))))
    ax3.plot(t,middle.real, label=r"$\Psi_{%.i}$" %(lam),  color=cm1(color_index/(len(lambda_list))))
    color_index+=1
ax1.set_xlabel("t")
ax2.set_xlabel("t")
ax1.set_title(r"max($|\Psi(x,t)$|")
ax2.set_title(r"max(Real($\Psi))$")
ax3.set_title(r"max(Imag($\Psi))$")
ax1.set_ylabel(r"log(max($\Psi$))")
ax1.set_yscale("log")
ax2.set_ylabel(r"max($\Psi$)")
ax3.set_ylabel(r"max($\Psi$)")
ax1.legend()
ax2.legend()
ax3.legend()
plt.show()