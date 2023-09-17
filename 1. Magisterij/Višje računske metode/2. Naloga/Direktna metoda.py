import numpy as np
import matplotlib.pyplot as plt
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

from scipy.integrate import simpson
from scipy.special import  factorial, hermite
##za sortiranje lastnih vrednosti in pripadajocih vektorjev:
# import numpy as np
# import numpy.linalg as linalg
#
# A = np.random.random((3,3))
# eigenValues,eigenVectors = linalg.eig(A)
#
# idx = eigenValues.argsort()[::-1]
# eigenValues = eigenValues[idx]
# eigenVectors = eigenVectors[:,idx]
##

##n vedno vecji od 5


#x=np.linspace(-50,50,500)
#Osnovna chi=psi_0+1/2psi_1
alpha=1
L=20
x=np.linspace(-L,L,10000)
a=0
def lastne_f(x, stanje):
    funkcija_m=hermite(stanje)(x-a) * np.exp(-(x-a)**2/2)/(np.sqrt(2**stanje*factorial(stanje,exact=True) * np.sqrt(np.pi)))
    funkcija_m /= np.sqrt(simpson(funkcija_m**2,x=x))
    return funkcija_m
N=50
H=np.zeros((N,N))
def matrix_H(N, lamb,x):
    for j in tqdm(range(N)):
        for k in range(j,N):
            if j==k:
                H[j,k]+=1/2+k
                H[j,k]+=lamb*integrate.simpson(lastne_f(x,j)*x**4*lastne_f(x,k),x)
            else:
                H[j,k]=lamb*integrate.simpson(lastne_f(x,j)*x**4*lastne_f(x,k),x)
                H[k,j]=H[j,k]
    return H
lamb_imshow="N"
if lamb_imshow=="Yes":
    lamb_list=[0.01,1]
    fig = plt.figure()
    fig2 = plt.figure()
    for i in range(len(lamb_list)):
        ax = fig.add_subplot(1,2,i+1)
        fig.suptitle(r"Matrika H, $L=%i$" %(L))
        ax.set_title(r"$\lambda=%.2f$" %(lamb_list[i]))
        H=matrix_H(N,lamb_list[i],x)
        im=ax.imshow(H, cmap="coolwarm")
        fig.colorbar(im)

        ax2 =fig2.add_subplot(1,2,i+1)
        ax2.set_title(r"$\lambda=%.2f$" %(lamb_list[i]))
        fig2.suptitle(r"Lastni vektorji, $L=%i$" %(L))
        E,vecorji=np.linalg.eig(H)
        idx = E.argsort()   
        eigenValues = E[idx]
        eigenVectors = vecorji[:,idx]
        im2=ax2.imshow(abs(eigenVectors),cmap="coolwarm")
        ax2.set_xlabel(r"n-ti lastni vektor")
        ax2.set_ylabel(r"$\Psi_{HO,n}$")
        fig2.colorbar(im2)
    plt.show()


from scipy.integrate import simpson
from scipy.special import  factorial, hermite
def generate_psi(L:float,N:int,n:int,a=0.,zero_pad=1):
    '''N is the number of intervals and n is the order of the Hermite polynomial
    L is the half-width of the interval
    zero_pad is an integer that fixes the dirichlet boundary conditions, fixing psi=0 at -L,L
    due to different stencils zero padding has to be adjusted
    a is the shift of the initial position'''

    x = np.linspace(-L,L,N)
    psi = hermite(n)(x-a) * np.exp(-(x-a)**2/2)/(np.sqrt(2**n*factorial(n,exact=True) * np.sqrt(np.pi)))
    psi[-zero_pad] = 0
    psi[zero_pad] = 0

    psi /= np.sqrt(simpson(psi**2,x=x))
    psi = np.array(psi,dtype=np.complex128) # set the type to complex for further calcualtions

    return psi,x

def calc_braket(x,psi1,psi2):

    return simpson(np.conjugate(psi1)*psi2,x=x)

def direct2(N:int,n_int:int,lbda:float,L):

    basis = np.zeros((n_int,N),dtype=np.complex128)
    # create a basis
    for i in range(N):
        basis[:,i] = generate_psi(L,n_int,i,zero_pad=0)[0]

    _,x = generate_psi(L,n_int,0,zero_pad=0) # just to get the x values

    H = np.zeros((N,N),dtype=np.complex128) # initialize the hamiltonian
    # construct the hamiltonian
    for i in range(N):
        for j in range(i,N):
            if i == j: # diagonal components
                H[i,j] += i + 0.5
                H[i,j] += lbda*calc_braket(x,basis[:,i],x**4*basis[:,j])
            
            else: # nondiagonal components
                H[i,j] = lbda*calc_braket(x,basis[:,i],x**4*basis[:,j])
                H[j,i] = np.conjugate(H[i,j])

    eigval, eigvec = scipy.linalg.eig(H) # get the eigenvalues and eigenvectores
    sort_mat = np.argsort(eigval) # a matrix of indices to sort the eigenvalues from lowest to highest
    return eigval[sort_mat],eigvec[:,sort_mat],basis, H






def lastne_za_x(N, x, n, H):
    funkcija=0
    for i in range(N):
        funkcija+=hermite(n)(x-a) * np.exp(-(x-a)**2/2)/(np.sqrt(2**n*factorial(n,exact=True) * np.sqrt(np.pi))) * (H[i,n])
    #funkcija/= np.sqrt(simpson(funkcija**2,x=x))
    return funkcija



plot_nicelna="N"
if plot_nicelna=="Yes":
    lamb_list=[0.01,1]
    fig = plt.figure()
    k=1
    for i in tqdm(range(3)):
        ylim=0
        for j in range(len(lamb_list)):
            ax = fig.add_subplot(3,2,k)
            fig.suptitle(r"$\Psi(0), L=%i$" %(L))
            ax.set_title(r"$\lambda=%.2f, n=%i$" %(lamb_list[j],i))
            eigval,eigvec,basis,H=direct2(N,10000, lamb_list[j], L)
            ax.plot(x,abs(lastne_za_x(N,x,i,eigvec)))
            ax.set_xlabel("x")
            if j%2==0:
                ylim=max(abs(lastne_za_x(N,x,i,eigvec)))
            ax.set_ylim(-0.05,ylim+0.1)
            ax.set_ylabel(r"$\Psi_{%i}$"%(i))
            k+=1
    plt.show()



plt_elipse="N"
if plt_elipse=="Yes":
    lambd=0
    eigval,eigvec,basis=direct2(N,10000, lambd, L)
    x_plot=[]
    y_plot=[]
    stanje=5
    lamb_list=[0,0.01,0.1,1,10]
    plotting=1
    fig = plt.figure()
    fig2 = plt.figure()
    ax2= fig2.add_subplot(1,1,1)
    for stanje in [5,10,20,40]:
        ax = fig.add_subplot(2,2,plotting)
        k=0
        plostina=[]
        for lam in lamb_list:
            index_l=0
            index_r=0
            x_plot=[]
            y_plot=[]
            for i in range(len(x)):
                Det=1+4*(stanje+1/2)*lam
                y_test=2*(stanje+1/2)-x[i]**2-2*lam*x[i]**4
                if y_test>=0:
                    x_plot.append(x[i])
                    y_plot.append(np.sqrt(2*(stanje+1/2)-x[i]**2-2*lam*x[i]**4))
                if y_test<0 and 0<i<len(x)-1 and index_l==0 and (2*(stanje+1/2)-x[i+1]**2-2*lam*x[i+1]**4)>0:
                    x_plot.append(x[i])
                    y_plot.append(0)
                    index_l+=1
                if y_test<0 and 0<i<len(x)-1 and index_r==0 and  (2*(stanje+1/2)-x[i-1]**2-2*lam*x[i-1]**4)>0:
                    x_plot.append(x[i])
                    y_plot.append(0)
                    index_r+=1
            x_plot=np.array(x_plot)
            y_plot=np.array(y_plot)
            ax.set_title("n=$%i$" %(stanje))
            ax.plot(x_plot,y_plot, color=plt.cm.coolwarm(k/len(lamb_list)), label=r"$\lambda=%.2f$" %(lam))
            ax.plot(x_plot,-y_plot, color=plt.cm.coolwarm(k/len(lamb_list)))
            ax.fill_between(x_plot, -y_plot, y_plot, color=plt.cm.coolwarm(k/len(lamb_list)), alpha=.05)
            ax.set_xlabel("x")
            ax.set_ylabel("p")
            k+=1
        plotting+=1
    #plt.plot(x,-y,color="blue")
    ax.legend()
    plt.show()


plot_r="N"
if plot_r=="Yes":
    lamb_list=np.linspace(0,20,1000)
    plotting=1
    fig = plt.figure()
    ax= fig.add_subplot(1,1,1)
    stanja=[0,1,2,3,5,10,20,50,100]
    for stanje in stanja:
        k=0
        plostina=[]
        for lam in lamb_list:
            index_l=0
            index_r=0
            x_plot=[]
            y_plot=[]
            for i in range(len(x)):
                Det=1+4*(stanje+1/2)*lam
                y_test=2*(stanje+1/2)-x[i]**2-2*lam*x[i]**4
                if y_test>=0:
                    x_plot.append(x[i])
                    y_plot.append(np.sqrt(2*(stanje+1/2)-x[i]**2-2*lam*x[i]**4))
                if y_test<0 and 0<i<len(x)-1 and index_l==0 and (2*(stanje+1/2)-x[i+1]**2-2*lam*x[i+1]**4)>0:
                    x_plot.append(x[i])
                    y_plot.append(0)
                    index_l+=1
                if y_test<0 and 0<i<len(x)-1 and index_r==0 and  (2*(stanje+1/2)-x[i-1]**2-2*lam*x[i-1]**4)>0:
                    x_plot.append(x[i])
                    y_plot.append(0)
                    index_r+=1
            x_plot=np.array(x_plot)
            y_plot=np.array(y_plot)

            plostina.append(scipy.integrate.quad(lambda x: 2*np.sqrt(2*(stanje+1/2)-x**2-2*lam*x**4),min(x_plot)+0.05,max(x_plot)-0.05)[0]/(np.pi*(2*(stanje+1/2))))
        fig.suptitle("Prikaz $r$ v odvisnosti od $\lambda$")
        from scipy.optimize import curve_fit
        def func(x, a,b, c):
            return (x+a)**(-b) + c
        popt, pcov = curve_fit(func, lamb_list, plostina)
        print(popt)
        ax.plot(lamb_list,plostina,color=plt.cm.coolwarm(plotting/len(stanja)), label="$n=%i$" %(stanje))
        ax.plot(lamb_list, func(lamb_list, *popt), "--",color=plt.cm.coolwarm(plotting/len(stanja)), label="Fit $c=%.2f$, $n=%i$" %(popt[2],stanje))
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$r=\frac{S_n}{2\pi}$")
        plotting+=1
        ax.legend()
    plt.show()

plot_r="N"
if plot_r=="Yes":
    lamb_list=np.linspace(0,20,1000)
    plotting=1
    fig = plt.figure()
    ax= fig.add_subplot(1,1,1)
    stanja=[0,1,2,3,5,10,20,50,100]
    for stanje in stanja:
        k=0
        plostina=[]
        for lam in lamb_list:
            index_l=0
            index_r=0
            x_plot=[]
            y_plot=[]
            for i in range(len(x)):
                Det=1+4*(stanje+1/2)*lam
                y_test=2*(stanje+1/2)-x[i]**2-2*lam*x[i]**4
                if y_test>=0:
                    x_plot.append(x[i])
                    y_plot.append(np.sqrt(2*(stanje+1/2)-x[i]**2-2*lam*x[i]**4))
                if y_test<0 and 0<i<len(x)-1 and index_l==0 and (2*(stanje+1/2)-x[i+1]**2-2*lam*x[i+1]**4)>0:
                    x_plot.append(x[i])
                    y_plot.append(0)
                    index_l+=1
                if y_test<0 and 0<i<len(x)-1 and index_r==0 and  (2*(stanje+1/2)-x[i-1]**2-2*lam*x[i-1]**4)>0:
                    x_plot.append(x[i])
                    y_plot.append(0)
                    index_r+=1
            x_plot=np.array(x_plot)
            y_plot=np.array(y_plot)

            plostina.append(scipy.integrate.quad(lambda x: 2*np.sqrt(2*(stanje+1/2)-x**2-2*lam*x**4),min(x_plot)+0.05,max(x_plot)-0.05)[0]/(np.pi*(2*(stanje+1/2))))
        fig.suptitle("Prikaz $r$ v odvisnosti od $\lambda$")
        from scipy.optimize import curve_fit
        def func(x, a,b, c):
            return (x+a)**(-b) + c
        popt, pcov = curve_fit(func, lamb_list, plostina)
        print(popt)
        ax.plot(lamb_list,plostina,color=plt.cm.coolwarm(plotting/len(stanja)), label="$n=%i$" %(stanje))
        ax.plot(lamb_list, func(lamb_list, *popt), "--",color=plt.cm.coolwarm(plotting/len(stanja)), label="Fit $c=%.2f$, $n=%i$" %(popt[2],stanje))
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$r=\frac{S_n}{2\pi}$")
        plotting+=1
        ax.legend()
    plt.show()


plot_energije="N"
if plot_energije=="Yes":
    n_space=np.linspace(0,N,N)
    lamb_list=[0,0.01,0.1,0.5,1,10]
    fig = plt.figure()
    l=1
    ylim=0
    stanje=N
    ax = fig.add_subplot(1,1,1)
    for j in range(len(lamb_list)):
        fig.suptitle(r"Energije za diskretno pri različnih $\lambda$, $L=%i$" %(L))
        index_l=0
        index_r=0
        x_plot=[]
        y_plot=[]
        for i in range(len(x)):
            Det=1+4*(stanje+1/2)*lamb_list[j]
            y_test=2*(stanje+1/2)-x[i]**2-2*lamb_list[j]*x[i]**4
            if y_test>=0:
                x_plot.append(x[i])
                y_plot.append(np.sqrt(2*(stanje+1/2)-x[i]**2-2*lamb_list[j]*x[i]**4))
            if y_test<0 and 0<i<len(x)-1 and index_l==0 and (2*(stanje+1/2)-x[i+1]**2-2*lamb_list[j]*x[i+1]**4)>0:
                x_plot.append(x[i])
                y_plot.append(0)
                index_l+=1
            if y_test<0 and 0<i<len(x)-1 and index_r==0 and  (2*(stanje+1/2)-x[i-1]**2-2*lamb_list[j]*x[i-1]**4)>0:
                x_plot.append(x[i])
                y_plot.append(0)
                index_r+=1
        x_plot=np.array(x_plot)
        y_plot=np.array(y_plot)
        plostina=(scipy.integrate.quad(lambda x: 2*np.sqrt(2*(stanje+1/2)-x**2-2*lamb_list[j]*x**4),min(x_plot)+0.005,max(x_plot)-0.005)[0]/(np.pi*(2*(stanje+1/2))))
        eigval,eigvec,basis,H=direct2(N,100, lamb_list[j], L)
        ax.axvline(x=plostina*N, ymin=0, ymax=350000, color=plt.cm.coolwarm(j/len(lamb_list)),ls='--', label=r"$r\cdot N=%.i$" %(plostina*N+1))
        ax.plot(n_space,np.sort(abs(eigval)), "-", color=plt.cm.coolwarm(j/len(lamb_list)),label="$\lambda=%.2f$" %(lamb_list[j]))
        ax.set_xlabel("n")
        ax.set_ylabel(r"$E_n$")
        ax.set_yscale("log")
        l+=1
        ax.legend()
    plt.show()


L=50
h=2
tau=2*np.pi*h**2*0.001
x=np.arange(-L, L, h)
T=tau*len(x)
print(T)
t=np.linspace(0,T,len(x))
N=len(x)
M=N
print(N)
tau=t[1]-t[0]
alpha=1
lam=1
stanje=0
gamma=0
####### PLOTS TO SHOW #######
plot_3D=None
plot_Matrix=None
plot_animation=None
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

A_plus, A_minus=A(M,tau,h,lam)
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

H_matrix=H_pet(M,tau,h,lam)
#A_plus, A_minus=A(M,tau,h,lam)
Wawe0=main(A_plus, A_minus,lastne_f(x,0))
plot_casovni="no"
if plot_casovni=="Yes":
    lamb_list=[0.01,1]
    fig = plt.figure()
    k=1
    for i in tqdm(range(3)):
        ylim=0
        for j in range(len(lamb_list)):
            ax = fig.add_subplot(3,2,k)
            fig.suptitle(r"$\Psi(0), L=%i$" %(L))
            ax.set_title(r"$\lambda=%.2f, n=%i$" %(lamb_list[j],i))
            A_plus, A_minus=A(N,tau,h,lamb_list[j])
            Wawe0=main(A_plus, A_minus,lastne_f(x,i))
            ax.plot(x,abs(Wawe0[0]), label="Prejšnja")
            eigval,eigvec,basis,H=direct2(N,10000, lamb_list[j], L)
            ax.plot(x,abs(lastne_za_x(N,x,i,eigvec)),"--",label="Diskretne")
            ax.set_xlabel("x")
            if j%2==0:
                ylim=max(abs(lastne_za_x(N,x,i,eigvec)))
            ax.set_ylim(-0.05,ylim+0.1)
            ax.set_ylabel(r"$\Psi_{%i}$"%(i))
            k+=1
            ax.legend()
    plt.show()

import time
def koef_of_hermite(n):
	fac = 1
	for i in range(0, n):
		fac *= np.sqrt(2*(i+1))
	return fac

def phi_n_1(x, n=0):
	H = hermite(n, monic=False)
	return np.exp(-1*x**2/2) * H(x) / (np.pi**(1/4) * koef_of_hermite(n))

def time_evol(koef, base_vectors, energies, t_s, spacepoints):
	start_time = time.time()
	N = len(koef)
	psi_t_s = np.zeros((len(t_s), len(spacepoints)), dtype=np.complex_)
	for i in range(N):
		if koef[i] == 0:
			continue
		for j in range(N):
			psi_part = np.zeros_like(spacepoints, dtype=np.complex_)
			for k in range(N):
				psi_part += koef[i] * base_vectors[i, j] * base_vectors[k, j] * phi_n_1(spacepoints, n=k)
			for t_ind in range(len(t_s)):
				psi_t_s[t_ind] += psi_part * np.exp(-1j*energies[j]*t_s[t_ind])
	print(time.time()-start_time)
	return psi_t_s
L=50
T=2*np.pi
x=np.arange(-L, L, h)
t=np.linspace(0,T,len(x))
tau=t[1]-t[0]
N=len(x)
M=N
eigval,eigvec,basis,H=direct2(N,1000, 1, L)
psi_0=lastne_f(x,0)
psi_t=lastne_f(x,0)
Wawe1=[]
lam=0
A_plus, A_minus=A(M,tau,h,lam)
H_matrix=H_pet(M,tau,h,lam)
Wawe0=main(A_plus, A_minus,lastne_f(x,0))
from tqdm import tqdm
for tau in tqdm(t):
    psi_t=np.zeros(len(lastne_f(x,0)), dtype=complex)
    for i in range(len(eigval)):
        psi_t+=np.array(lastne_f(x,i),dtype=complex)*calc_braket(x,lastne_f(x,i),eigvec[i])*np.exp(-1j*eigval[i]*tau)
    #psi_0=psi_t.copy()
    Wawe1.append(psi_t)
Wawe1=np.array(Wawe1, dtype=complex)
Wawe0=np.array(Wawe0, dtype=complex)
plot_3D="nO"
if plot_3D==True:
    X,Y=np.meshgrid(x,t)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(r" |$\Psi(x,t)^2_{nova}|$, $\lambda=%.2f, \tau=%.4f, h=%.4f$" %(lam,tau,h))
    im=ax.plot_surface(X, Y, abs(Wawe1), rstride=1, cstride=1,
                    cmap='seismic', edgecolor='none')
    #im=ax.plot_surface(X, Y, abs(Wawe0), rstride=1, cstride=1, cmap='seismic', edgecolor='none')
    cbar = ax.figure.colorbar(im, label="Amplituda")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.legend()
    plt.show()


    X,Y=np.meshgrid(x,t)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(r" |$\Psi(x,t)^2_{prej}-\Psi(x,t)^2_{nova}|$, $\lambda=%.2f, \tau=%.4f, h=%.4f$" %(lam,tau,h))
    im=ax.plot_surface(X, Y, abs(Wawe1)-abs(Wawe0), rstride=1, cstride=1,
                    cmap='seismic', edgecolor='none')
    #im=ax.plot_surface(X, Y, abs(Wawe0), rstride=1, cstride=1, cmap='seismic', edgecolor='none')
    cbar = ax.figure.colorbar(im, label="Amplituda")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.legend()
    plt.show()


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
plot_animation=True
if plot_animation==True:
    #Wawe1=main(A_plus, A_minus,lastne_f(x,1))

    fig, ax = plt.subplots()

    #line2, = ax.plot(x, x/40, color="white")
    room, =ax.plot([-10,10], [0.8,-0.8], color="white")
    line3, = ax.plot(x, abs(Wawe0[0,:N]), color="blue", label=r"$|\Psi(x)|^2$, implicitna")
    line4, = ax.plot(x, abs(Wawe1[0,:N]), color="black", label=r"$|\Psi(x)|^2$, propagator")
    #line2, = ax.plot(x, Wawe0[0,:N].imag, color="black", label="Lastno stanje 0, imag")
    #line3, = ax.plot(x, abs(Wawe0[0,:N]), color="green", label="Lastno stanje 0, abs")
    #line3, = ax.plot(x, Wawe1[0,:N].real, color="blue", label="Lastno stanje 1")





    def animate(i):
        room.set_data([-10,10], [0.8,-0.8])
        line3.set_ydata(abs(Wawe0[i,:N]))
        line4.set_ydata(abs(Wawe1[i,:N]))
        legend = plt.legend(loc='lower right')
        plt.title(r"Primerjava metod za $\lambda=%.2f, \tau=%.4f, h=%.4f$" %(lam,tau,h))
        #line3.set_ydata(abs(Wawe0[i,:N]))
        #line3.set_ydata(Wawe1[i,:N].imag)  # update the data.
        return  room, line3, line4


    ani = animation.FuncAnimation(
        fig, animate, interval=20, frames=N, blit=True, save_count=50)

    # To save the animation, use e.g.
    #
    #ani.save(THIS_FOLDER +"\\Prikaz razvoja lam="+str(lam)+".gif")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)
    #ani.save("N=1000,Gauss.mp4")
    h_t="{:.4f}".format(tau)
    plt.xlabel(r"x")
    plt.ylabel(r"$\psi$ (x)")
    plt.title(r"Prikaz časovnega razvoja Gaussovega paketa za $\tau= %.3f$ in $h=%.4f$" %(tau,h))
    plt.legend()
    plt.show()