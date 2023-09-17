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
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.integrate import simpson
from scipy.linalg import eig_banded
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import quad
import math

from scipy.integrate import simpson
from scipy.special import  factorial, hermite
L=10
a=0

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


def lastne_f(x, stanje):
    funkcija_m=hermite(stanje)(x-a) * np.exp(-(x-a)**2/2)/(np.sqrt(2**stanje*factorial(stanje,exact=True) * np.sqrt(np.pi)))
    funkcija_m /= np.sqrt(simpson(funkcija_m**2,x=x))
    return funkcija_m

def lastne_za_x(N, x, n, H):
    funkcija=0
    for i in range(N):
        funkcija+=hermite(n)(x-a) * np.exp(-(x-a)**2/2)/(np.sqrt(2**n*factorial(n,exact=True) * np.sqrt(np.pi))) * (H[i,n]) 
    #funkcija/= np.sqrt(simpson(funkcija**2,x=x))
    return funkcija

def Potencial(x,lambd):
    return 1/2*x**2+lambd*x**4

def H_pet(M,h,lambd):
    H=np.zeros((int(M),int(M)), dtype=complex)
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

def H_3(x, lamb):
	N = len(x)
	h = x[1] - x[0]
	V_s = Potencial(x, lamb)
	M = np.diag(V_s) - 1/(2*h**2) * (-2*np.diag(np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1))
	return M

def Lanczos(H,psi):
    Lv=np.zeros((len(psi),len(psi)), dtype=complex) #Creates matrix for Lanczos vectors
    Hk=np.zeros((len(psi),len(psi)), dtype=complex)
    #Creates matrix for the Hamiltonian in Krylov subspace
    Lv[0]=psi/np.linalg.norm(psi) #Creates the first Lanczos vector as the normalized guess vector(psi
     
    #Performs the first iteration step of the Lanczos algorithm
    w=np.dot(H,Lv[0]) 
    a=np.dot(np.conj(w),Lv[0])
    w=w-a*Lv[0]
    Hk[0,0]=a
     
    #Performs the iterative steps of the Lanczos algorithm
    for j in range(1,len(psi)):
        b=(np.dot(np.conj(w),np.transpose(w)))**0.5
        Lv[j]=w/b
         
        w=np.dot(H,Lv[j])
        a=np.dot(np.conj(w),Lv[j])
        w=w-a*Lv[j]-b*Lv[j-1]
        
        #Creates tridiagonal matrix Hk using a and b values
        Hk[j,j]=a
        Hk[j-1,j]=b
        Hk[j,j-1]=np.conj(b)
    eigval, eigvec = scipy.linalg.eigh(Hk)
    sort_mat = np.argsort(eigval)
    eigval, eigvec=eigval[sort_mat],eigvec[:,sort_mat]
    return (Hk,Lv, eigval, eigvec)
def phi_n_1(x, n=0):
	H = hermite(n, monic=False)
	return np.exp(-1*x**2/2) * H(x) / (np.pi**(1/4) * koef_of_hermite(n))

def phi_n(x, n=0):
	H = hermite(n, monic=False)
	return np.exp(-1*x**2/2) * H(x) / (np.pi**(1/4) * np.sqrt(2**n * math.factorial(n)))

def koef_of_hermite(n):
	fac = 1
	for i in range(0, n):
		fac *= np.sqrt(2*(i+1))
	return fac

def zero_bound(y):
	y[0], y[-1] = 0, 0
	#y[1], y[-2] = y[2]/2, y[-3]/2
	return y
def sine(x):
	psi_0 = np.zeros_like(x)
	for i in range(len(x)):
		if x[i] > -np.pi and abs(x[i]) < np.pi:
			psi_0[i] = 1+np.sin(x[i]+np.pi/2)
	return psi_0

def normalize(psi, x):
	norm = simpson(psi**2, x)
	return psi / np.sqrt(norm)

def lanczos_base(N, x, lamb):
	lanczos_states = np.zeros((N, len(x)))
	H = H_3(x, lamb)
	lan_diag_0 = np.zeros(N-1)
	lan_diag_1 = np.zeros(N-2)
	# initial state and first step
	phi_0 = np.zeros_like(x)
	for i in range(1):
		phi_0 += phi_n_1(x, n=i)
	lanczos_states[0] = zero_bound(normalize(phi_0, x)) # začetni pogoj primeren za majhne vrednosti lambda
	#lanczos_states[0] = zero_bound(normalize(sine(x+0.5), x)) #zacetni pogoj za velike vrednosti lambda
	#plt.plot(lanczos_states[0])
	#plt.show()
	# second step
	old_vec = lanczos_states[0]
	new_vec = zero_bound(H.dot(old_vec))
	alpha = simpson(old_vec*new_vec, x)
	lan_diag_0[0] = alpha
	lanczos_states[1] = normalize(new_vec - alpha*old_vec, x)
	# n-th step
    
	for i in range(2, N):
		old_vec = lanczos_states[i-1]
		oldold_vec = lanczos_states[i-2]
		new_vec = zero_bound(H.dot(old_vec))
		alpha = simpson(old_vec*new_vec, x)
		beta = simpson(oldold_vec*new_vec, x)
		lan_diag_0[i-1] = alpha
		lan_diag_1[i-2] = beta
		lanczos_states[i] = normalize(new_vec - alpha*old_vec - beta*oldold_vec, x)
                
	return lanczos_states, lan_diag_0, lan_diag_1

def lancozos_jon(N,N_basis, lbda,x,L=5.):
    basis= np.zeros((N,N_basis), dtype=np.complex128)

    psi0,x= generate_psi(L,N,0,zero_pad=0,a=0)
    psi1,_=generate_psi(L,N,1,a=0, zero_pad=0)
    psi0 +=psi1*0.5
    psi0 /= np.sqrt(calc_braket(x,psi0,psi0))
    H=H_3(x,lbda)

    diag0= np.zeros(N_basis, dtype=np.complex128)
    diag1=np.zeros(N_basis-1, dtype=np.complex128)
    basis[:,0]=psi0
    partial=H@basis[:,0]
    diag0[0]=calc_braket(x,basis[:,0], partial)
    basis[:,1]=partial-basis[:,0]*diag0[0]
    basis[:,1]/=np.sqrt(simpson(basis[:,1]*np.conjugate(basis[:,1]),x=x))

    for i in range(2,N_basis):
        partial= H@basis[:,i-1]
        diag0[i-1]=calc_braket(x,basis[:,i-1],partial)
        diag1[i-2]=calc_braket(x,basis[:,i-2],partial)
        basis[:,i]=partial-basis[:,i-1]*diag0[i-1]-basis[:,i-2]*diag1[i-2]
        basis[:,i]/=np.sqrt(simpson(basis[:,i]*np.conjugate(basis[:,i]),x=x))
    
    partial =H@basis[:,-1]
    diag0[-1]=calc_braket(x,basis[:,-1],partial)
    diag1[-1]=calc_braket(x,basis[:,-2],partial)

    a_band =np.zeros((2,N_basis),dtype=np.complex128)
    a_band[0,1:]= diag1
    a_band[1] =diag0
    eigval, eigvec =eig_banded(a_band)
    sort_mat=np.argsort(eigval)
    return eigval[sort_mat], eigvec[sort_mat],basis

def calc_braket(x,psi1,psi2):

    return simpson(np.conjugate(psi1)*psi2,x=x)


N=100
x=np.linspace(-50,50,500)
h=x[1]-x[0]


lamb_imshow="No"
if lamb_imshow=="Yes":
    lamb_list=[0.01,1]
    fig = plt.figure()
    fig2 = plt.figure()
    for i in range(len(lamb_list)):
        ax = fig.add_subplot(1,2,i+1)
        fig.suptitle(r"Matrika H, $L=%i$" %(L))
        ax.set_title(r"$\lambda=%.2f$" %(lamb_list[i]))
        psi=lastne_f(x,0)
        lanczos_states, lan_diag_0, lan_diag_1=lanczos_base(N, x, lamb_list[i])
        Hk,Lv, eigval, eigvec=Lanczos(H_pet(N,h,lamb_list[i]),psi)
        lanczos_states, lan_diag_0, lan_diag_1=lanczos_base(N, x, lamb_list[i])
        eigval, eigvec = eigh_tridiagonal(lan_diag_0, lan_diag_1)
        sort_mat = np.argsort(eigval)
        eigval, eigvec=eigval[sort_mat],eigvec[:,sort_mat]
        im=ax.imshow(abs(Hk), cmap="coolwarm")
        eigval, eigvec = scipy.linalg.eigh(Hk)
        sort_mat = np.argsort(eigval)
        eigval, eigvec=eigval[sort_mat],eigvec[:,sort_mat]
        fig.colorbar(im)

        ax2 =fig2.add_subplot(1,2,i+1)
        ax2.set_title(r"$\lambda=%.2f$" %(lamb_list[i]))
        fig2.suptitle(r"Lastni vektorji, $L=%i$" %(L))
        im2=ax2.imshow(abs(eigvec),cmap="coolwarm")
        ax2.set_xlabel(r"n-ti lastni vektor")
        ax2.set_ylabel(r"$\Psi_{HO,n}$")
        fig2.colorbar(im2)
    plt.show()

plot_nicelna="No"
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
            psi=lastne_f(x,0)
            H,Lv, eigval, eigvec=Lanczos(H_pet(len(psi),h, lamb_list[j]),psi)
            ax.plot(x,abs(lastne_za_x(N,x,i,eigvec)))
            ax.set_xlabel("x")
            if j%2==0:
                ylim=max(abs(lastne_za_x(N,x,i,eigvec)))
            ax.set_ylim(-0.05,ylim+0.1)
            ax.set_ylabel(r"$\Psi_{%i}$"%(i))
            k+=1
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
    return eigval[sort_mat],eigvec[:,sort_mat],basis

plot_nicelna="No"
if plot_nicelna=="Yes":
    lamb_list=[0.01,1]
    fig = plt.figure()
    k=1
    for i in tqdm([0,1,10]):
        ylim=0
        for j in range(len(lamb_list)):
            ax = fig.add_subplot(3,2,k)
            fig.suptitle(r"$\Psi(0), L=%i$" %(L))
            ax.set_title(r"$\lambda=%.2f, n=%i$" %(lamb_list[j],i))
            lanczos_states, lan_diag_0, lan_diag_1=lanczos_base(N, x, lamb_list[j])
            eigval, eigvec = eigh_tridiagonal(lan_diag_0, lan_diag_1)
            sort_mat = np.argsort(eigval)
            #eigval, eigvec=eigval[sort_mat],eigvec[:,sort_mat]
            ax.plot(x[:N-1],abs(lastne_za_x(N-1,x[:N-1],i,eigvec)), label="Lanczos")
            eigval,eigvec,basis=direct2(N,10000, lamb_list[j], L)
            ax.plot(x,abs(lastne_za_x(N,x,i,eigvec)),"--", label="Diskretna")
            ax.set_xlabel("x")
            if j%2==0:
                ylim=max(abs(lastne_za_x(N,x,i,eigvec)))
            ax.set_ylim(-0.05,ylim+0.1)
            ax.set_ylabel(r"$\Psi_{%i}$"%(i))
            if k==5:
                 ax.legend()
            k+=1
    plt.show()

N=100
L=50
x=np.linspace(-L,L,500)
h=x[1]-x[0]
plot_energije="Yes"
if plot_energije=="Yes":
    n_space=np.linspace(0,N,N)
    lamb_list=[0.01,0.1,0.25,0.5,0.75,1]
    fig = plt.figure()
    l=1
    ylim=0
    ax = fig.add_subplot(1,2,1)
    i=1
    for j in range(len(lamb_list)):
        if l%4==0:
            i+=1
            ax = fig.add_subplot(1,2,i)
        fig.suptitle(r"Energije za oba primera pri različnih $\lambda$, $L=%i$" %(L))
        lanczos_states, lan_diag_0, lan_diag_1=lanczos_base(N, x, lamb_list[j])
        eigval, eigvec = eigh_tridiagonal(lan_diag_0, lan_diag_1)
        sort_mat = np.argsort(eigval)
        eigval, eigvec=eigval[sort_mat],eigvec[:,sort_mat]
        eigval, eigvec, basis=lancozos_jon(500,N, lamb_list[j],x,L)
        ax.plot(n_space,np.sort(abs(eigval)), ".-", label=r"Lanczsos $\lambda=%.2f$" %(lamb_list[j]), color=plt.cm.brg(l/len(lamb_list)))
        eigval,eigvec,basis=direct2(N,100, lamb_list[j], L)
        ax.plot(n_space,np.sort(abs(eigval)), "x-", label=r"Diskretna $\lambda=%.2f$" %(lamb_list[j]), color=plt.cm.brg(l/len(lamb_list)))
        ax.set_xlabel("n-to stanje")
        ax.set_yscale("log")
        ax.set_ylabel(r"$E_n$")
        l+=1
        ax.legend()
    plt.show()


plot_energije="No"
if plot_energije=="Yes":
    n_space=np.linspace(0,N,N)
    lamb_list=[0.01,0.1,0.25,0.5,0.75,1]
    fig = plt.figure()
    l=1
    ylim=0
    ax = fig.add_subplot(2,2,1)
    ax1 = fig.add_subplot(2,2,3)
    i=1
    for j in range(len(lamb_list)):
        if l%4==0:
            i+=1
            ax = fig.add_subplot(2,2,i)
            ax1 = fig.add_subplot(2,2,2+i)
        fig.suptitle(r"Energije za oba primera pri različnih $\lambda$, $L=%i$" %(L))
        lanczos_states, lan_diag_0, lan_diag_1=lanczos_base(N, x, lamb_list[j])
        eigval, eigvec = eigh_tridiagonal(lan_diag_0, lan_diag_1)
        sort_mat = np.argsort(eigval)
        eigval, eigvec=eigval[sort_mat],eigvec[:,sort_mat]
        eigval, eigvec, basis=lancozos_jon(500,N, lamb_list[j],x,L)
        ax.plot(n_space,np.sort(abs(eigval)), ".-", label=r"Lanczsos $\lambda=%.2f$" %(lamb_list[j]), color=plt.cm.brg(l/len(lamb_list)))
        ax1.plot(n_space[:30],np.sort(abs(eigval))[:30], ".-", label=r"Lanczsos $\lambda=%.2f$" %(lamb_list[j]), color=plt.cm.brg(l/len(lamb_list)))
        eigval,eigvec,basis=direct2(N,100, lamb_list[j], L)
        ax.plot(n_space,np.sort(abs(eigval)), "x-", label=r"Diskretna $\lambda=%.2f$" %(lamb_list[j]), color=plt.cm.brg(l/len(lamb_list)))
        ax1.plot(n_space[:30],np.sort(abs(eigval))[:30], "x-", label=r"Diskretna $\lambda=%.2f$" %(lamb_list[j]), color=plt.cm.brg(l/len(lamb_list)))
        ax.set_xlabel("n-to stanje")
        ax.set_yscale("log")
        ax.set_ylabel(r"$E_n$")
        ax1.set_xlabel("n-to stanje")
        ax1.set_yscale("log")
        ax1.set_ylabel(r"$E_n$")
        ax1.legend()
        l+=1
        ax.legend()
    plt.show()

N=150
plot_energije_konvergenca="Yes"
import timeit
start = timeit.timeit()
if plot_energije_konvergenca=="Yes":
    epsilon=0.1
    n_space=np.linspace(0,N,N)
    lamb_list=[0.01,0.1,0.25,0.5,0.75,1]
    fig = plt.figure()
    fig1 = plt.figure()
    l=1
    ylim=0
    i=1
    N_list=np.linspace(20,100,150)
    ax = fig.add_subplot(1,1,1)
    ax1 = fig1.add_subplot(1,1,1)
    fig.suptitle(r"Divergenca energije za $|E_{dir}-E_{Lan}|>%.3f$, $L=%i$" %(epsilon,L))
    fig1.suptitle(r"Prikaz časovne zahtevnosti programa v odvisnosti od baze, $L=%i$" %(L))
    for j in tqdm(range(len(lamb_list))):
        N_plot=[]
        N_razlika=[]
        time_plot=[]
        for N in range(20,len(N_list)):
            start = timeit.timeit()
            N_plot.append(N)
            eigval_lan, eigvec_lan, basis_lan=lancozos_jon(500,N, lamb_list[j],x,L)
            eigval,eigvec,basis=direct2(N,100, lamb_list[j], L)
            test=0
            for pi in range(len(eigval_lan)):
                if abs(eigval[pi]-eigval_lan[pi])/eigval[pi]>epsilon and test==0:
                    test+=1
                    N_razlika.append(pi)
            end = timeit.timeit()
            time_plot.append(abs(end-start))
            if j==0:
                 ax1.plot(N_plot,time_plot)
        ax.plot(N_plot,N_razlika, "x-", label=r"$\lambda=%.2f$" %(lamb_list[j]), color=plt.cm.brg(j/len(lamb_list)))
    ax.set_xlabel("n velikost baze")
    ax.set_ylabel(r"n_ta divergenca")
    ax.legend()
    ax1.set_xlabel("n velikost baze")
    ax1.set_ylabel("t[s]")
    ax1.legend()
    plt.show()