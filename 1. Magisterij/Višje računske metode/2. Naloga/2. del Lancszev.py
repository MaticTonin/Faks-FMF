from scipy.integrate import simpson
from scipy.special import  factorial, hermite
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad
import numpy as np
import scipy
from scipy.linalg import eig_banded
from tqdm import tqdm
from scipy import special
alpha=1
L=20
x=np.linspace(-L,L,10000)
a=0
N=50

def Psi(L,N,n,a=0,zero_pad=1):
    x=np.linspace(-L,L,N)
    psi=1/(np.sqrt(2**n*factorial(n,exact=True) * np.sqrt(np.pi)))*np.exp(-(x-a)**2/2)*hermite(n)(x-a)
    psi[-zero_pad]=0
    psi[zero_pad]=0
    psi=psi/np.sqrt(simpson(psi**2,x=x))
    psi=np.array(psi,dtype=np.complex)
    return psi,x

def calc_braket(x,psi1,psi2):
    return simpson(np.conjugate(psi1)*psi2,x=x)

def Eigval(N,n_int,lam,L):
    baza=np.zeros((n_int,N),dtype=np.complex)
    for i in range(N):
        baza[:,i]=Psi(L,n_int,i,zero_pad=0)[0]
    _,x=Psi(L,n_int,0,zero_pad=0) 
    H = np.zeros((N,N),dtype=np.complex) 
    for i in range(N):
        for j in range(i,N):
            if i==j:
                H[i,j]+=i+0.5
                H[i,j]+=lam*calc_braket(x,baza[:,i],x**4*baza[:,j])
            else:
                H[i,j]=lam*calc_braket(x,baza[:,i],x**4*baza[:,j])
                H[j,i]=np.conjugate(H[i,j])
    eigval,eigvec=scipy.linalg.eig(H)
    sort=np.argsort(eigval) 
    return eigval[sort],eigvec[:,sort],baza, H

def lastne_za_x(N, x, n, H):
    funkcija=0
    for i in range(N):
        funkcija+=hermite(n)(x) * np.exp(-(x)**2/2)/(np.sqrt(2**n*factorial(n,exact=True) * np.sqrt(np.pi))) * (H[i,n])
    funkcija/=np.sqrt(simpson(funkcija**2,x=x))
    return funkcija

def Potencial(x,lambd):
    return 1/2*x**2+lambd*x**4

def H_3(x, lamb):
	N=len(x)
	h=x[1] - x[0]
	V_s=Potencial(x, lamb)
	H=np.diag(V_s)-1/(2*h**2)*(-2*np.diag(np.ones(N))+np.diag(np.ones(N-1),1)+np.diag(np.ones(N-1),-1))
	return H


def Lancozos(N,M, lam,x,L=5.):
    baza=np.zeros((N,M), dtype=np.complex)
    psi0,x=Psi(L,N,0,zero_pad=0,a=0)
    psi1,_=Psi(L,N,1,a=0, zero_pad=0)
    psi0+=psi1*0.5
    psi0/=np.sqrt(calc_braket(x,psi0,psi0))
    H=H_3(x,lam)
    diag0= np.zeros(M, dtype=np.complex)
    diag1=np.zeros(M-1, dtype=np.complex)
    baza[:,0]=psi0
    partial=H@baza[:,0]
    diag0[0]=calc_braket(x,baza[:,0], partial)
    baza[:,1]=partial-baza[:,0]*diag0[0]
    baza[:,1]/=np.sqrt(simpson(baza[:,1]*np.conjugate(baza[:,1]),x=x))
    for i in range(2,M):
        partial=H@baza[:,i-1]
        diag0[i-1]=calc_braket(x,baza[:,i-1],partial)
        diag1[i-2]=calc_braket(x,baza[:,i-2],partial)
        baza[:,i]=partial-baza[:,i-1]*diag0[i-1]-baza[:,i-2]*diag1[i-2]
        baza[:,i]/=np.sqrt(simpson(baza[:,i]*np.conjugate(baza[:,i]),x=x))
    partial=H@baza[:,-1]
    diag0[-1]=calc_braket(x,baza[:,-1],partial)
    diag1[-1]=calc_braket(x,baza[:,-2],partial)
    a_band =np.zeros((2,M),dtype=np.complex)
    a_band[0,1:]= diag1
    a_band[1] =diag0
    eigval, eigvec =eig_banded(a_band)
    sort_mat=np.argsort(eigval)
    return eigval[sort_mat], eigvec[sort_mat],baza

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
            eigval, eigvec, basis=Lancozos(500,N, lamb_list[j],x,L)
            ax.plot(x[:N-1],abs(lastne_za_x(N-1,x[:N-1],i,eigvec)), label="Lanczos")
            eigval,eigvec,basis=Eigval(N,10000, lamb_list[j], L)
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
        eigval, eigvec, basis=Lancozos(500,N, lamb_list[j],x,L)
        ax.plot(n_space,np.sort(abs(eigval)), ".-", label=r"Lanczsos $\lambda=%.2f$" %(lamb_list[j]), color=plt.cm.brg(l/len(lamb_list)))
        eigval,eigvec,basis, H=Eigval(N,100, lamb_list[j], L)
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
        eigval, eigvec, basis=Lancozos(500,N, lamb_list[j],x,L)
        ax.plot(n_space,np.sort(abs(eigval)), ".-", label=r"Lanczsos $\lambda=%.2f$" %(lamb_list[j]), color=plt.cm.brg(l/len(lamb_list)))
        ax1.plot(n_space[:30],np.sort(abs(eigval))[:30], ".-", label=r"Lanczsos $\lambda=%.2f$" %(lamb_list[j]), color=plt.cm.brg(l/len(lamb_list)))
        eigval,eigvec,basis, H=Eigval(N,100, lamb_list[j], L)
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
            eigval_lan, eigvec_lan, basis_lan=Lancozos(500,N, lamb_list[j],x,L)
            eigval,eigvec,basis, H=Eigval(N,100, lamb_list[j], L)
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
    N=len(H_matrix[0,:])
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

L=5
x=np.linspace(-5,5,100)
h=x[1]-x[0]

def time_propagation(vector,basis,eigval,x,t):
    psi_t=np.zeros(basis.shape[0],dtype=np.complex)
    for i in range(basis.shape[1]):
        psi_t+=basis[:,i]*calc_braket(x,basis[:,i],vector)*np.exp(-1j*eigval[i]*t)
    return psi_t


lam=0
lamb_list=[0,0.01,0.1,0.5,0.75,1]
fig=plt.figure()
fig.suptitle(r"Prikaz $|\Psi(x,t)_0|$")
L=20
x=np.linspace(-L,L,10000)
N=50
h=x[1]-x[0]
plt.show()
#tau=t[1]-t[0]
fig=plt.figure()
fig1=plt.figure()
h=x[1]-x[0]
fig.suptitle(r"Časovni razvoj za funkcijo $|\Psi(x,t)_0|$")
fig1.suptitle(r"Razlika funkcij za funkcijo $|\Psi(x,t)_0|$")
L=5
N=100
t=np.linspace(0,2*np.pi,100)
tau=t[1]-t[0]
x=np.linspace(-L,L,100)
for j in tqdm(range(len(lamb_list))):
    ax = fig.add_subplot(2,3,j+1)
    ax1 = fig1.add_subplot(2,3,j+1)
    t=np.linspace(0,2*np.pi,100)
    tau=t[1]-t[0]
    H_matrix=H_pet(N,tau,h,lamb_list[j])
    A_plus, A_minus=A(N,tau,h,lamb_list[j])
    Wawe0=main(A_plus, A_minus,lastne_f(x,0))
    eigval,eigvec,basis, H=Eigval(N,100, lamb_list[j], L)
    psi_0=lastne_za_x(N, x, 0, eigvec)
    #psi_0=lastne_f(x,0)
    Wawe1=[]
    for i in range(len(t)):
        solution=time_propagation(psi_0,basis,eigval,x,t[i])
        Wawe1.append(solution)
    Wawe1=np.array(Wawe1, dtype=np.complex128)
    ax.set_title(r"$\lambda=%.2f$" %(lamb_list[j]))
    ax1.set_title(r"$\lambda=%.2f$" %(lamb_list[j]))
    barva=0
    for i in range(len(Wawe1)):
        if i%20==0:
            ax.plot(x,abs(Wawe1[i]), "-",label="$t=%.2f$" %(t[i]),color=plt.cm.coolwarm(i/len(t)))
            ax1.plot(x,abs(abs(Wawe1[i])-abs(Wawe0[i])),label="$t=%.2f$" %(t[i]),color=plt.cm.coolwarm(i/len(t)))
        ax.set_xlabel("x")
        ax.set_ylabel(r"$|\Psi(x,t)_0|$")
        ax.legend()
        ax1.set_xlabel("x")
        ax1.set_ylabel(r"$abs(|\Psi_{0,2DN}|-|\Psi_{0,1DN}|)$",fontsize=10)
        ax1.set_yscale("log")
        ax1.legend()
plt.show()


import matplotlib.ticker as mticker
def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"
plot_3D=True
if plot_3D==True:
    X,Y=np.meshgrid(x,t)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(r" |$\Psi(x,t)^2_{nova}|$, $\lambda=%.2f, \tau=%.4f, h=%.4f$" %(lamb_list[j],tau,h))
    im=ax.plot_surface(X, Y, abs(Wawe1), rstride=1, cstride=1,
                    cmap='seismic', edgecolor='none')
    #im=ax.plot_surface(X, Y, abs(Wawe0), rstride=1, cstride=1, cmap='seismic', edgecolor='none')
    cbar = ax.figure.colorbar(im, label="Amplituda")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel(r"$|\Psi(x,t)_0|$")
    ax.legend()
    plt.show()


    X,Y=np.meshgrid(x,t)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(r" |$\Psi(x,t)^2_{prej}-\Psi(x,t)^2_{nova}|$, $\lambda=%.2f, \tau=%.4f, h=%.4f$" %(lamb_list[j],tau,h))
    im=ax.plot_surface(X, Y, np.log10(abs(abs(Wawe1)-abs(Wawe0))), rstride=1, cstride=1,
                    cmap='seismic', edgecolor='none')
    #im=ax.plot_surface(X, Y, abs(Wawe0), rstride=1, cstride=1, cmap='seismic', edgecolor='none')
    cbar = ax.figure.colorbar(im, label="Amplituda")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel(r"$abs(|\Psi_{0,2DN}|-|\Psi_{0,1DN}|)$")
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend()
    plt.show()