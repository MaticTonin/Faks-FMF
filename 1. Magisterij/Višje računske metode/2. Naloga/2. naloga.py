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
N=100
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

    eigval, eigvec = scipy.linalg.eig(H) # get the eigenvalues and eigenvectores
    sort_mat = np.argsort(eigval) # a matrix of indices to sort the eigenvalues from lowest to highest
    return eigval[sort_mat],eigvec[:,sort_mat],basis
lamb_imshow="No"
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







def calc_braket(x,psi1,psi2):

    return simpson(np.conjugate(psi1)*psi2,x=x)

def lastne_za_x(N, x, i, H):
    funkcija=0
    for n in range(N):
        funkcija+=hermite(n)(x-a) * np.exp(-(x-a)**2/2)/(np.sqrt(2**n*factorial(n,exact=True) * np.sqrt(np.pi))) * (H[i,n]) 
    return funkcija

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

    #psi /= np.sqrt(simpson(psi**2,x=x))
    psi = np.array(psi,dtype=np.complex128) # set the type to complex for further calcualtions

    return psi,x

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

    eigval, eigvec = scipy.linalg.eigh(H) # get the eigenvalues and eigenvectores
    sort_mat = np.argsort(eigval) # a matrix of indices to sort the eigenvalues from lowest to highest
    return eigval[sort_mat],eigvec[:,sort_mat],basis[:,sort_mat], H

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
            eigval,eigvec,basis=matrix_H(N,10000, lamb_list[j], L)
            ax.plot(x,abs(lastne_za_x(N,x,i,eigvec)))
            ax.set_xlabel("x")
            if j%2==0:
                ylim=max(abs(lastne_za_x(N,x,i,eigvec)))
            ax.set_ylim(-0.05,ylim+0.1)
            ax.set_ylabel(r"$\Psi_{%i}$"%(i))
            k+=1
    plt.show()



plt_elipse="No"
if plt_elipse=="Yes":
    lambd=0
    eigval,eigvec,basis=matrix_H(N,10000, lambd, L)
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


plot_r="No"
if plot_r=="Yes":
    lamb_list=np.linspace(0,2,100)
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
        ax.plot(lamb_list,plostina,color=plt.cm.coolwarm(plotting/len(stanja)), label="$n=%i$" %(stanje))
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$r=\frac{S_n}{2\pi}$")
        plotting+=1
        ax.legend()
    plt.show()


plot_3D="No"
if plot_3D=="Yes":
    x_plot=[]
    y_plot=[]
    stanje=5
    lambd_matrix=[]
    lamb_list=np.linspace(0,10,40)
    plotting=1
    x_matrix=[]
    y_matrix=[]
    fig = plt.figure()
    stanja=[0,2,5,10]
    for stanje in stanja:
        ax = fig.add_subplot(2,2,plotting, projection='3d')
        k=0
        for lam in lamb_list:
            index_l=0
            index_r=0
            x_plot=[]
            y_plot=[]
            lambda_plot=[]
            for i in range(len(x)):
                Det=1+4*(stanje+1/2)*lam
                y_test=2*(stanje+1/2)-x[i]**2-2*lam*x[i]**4
                if y_test>=0:
                    x_plot.append(x[i])
                    lambda_plot.append(lam)
                    y_plot.append(np.sqrt(2*(stanje+1/2)-x[i]**2-2*lam*x[i]**4))
                if y_test<0 and 0<i<len(x)-1 and index_l==0 and (2*(stanje+1/2)-x[i+1]**2-2*lam*x[i+1]**4)>0:
                    x_plot.append(x[i])
                    y_plot.append(0)
                    lambda_plot.append(lam)
                    index_l+=1
                if y_test<0 and 0<i<len(x)-1 and index_r==0 and  (2*(stanje+1/2)-x[i-1]**2-2*lam*x[i-1]**4)>0:
                    x_plot.append(x[i])
                    y_plot.append(0)
                    lambda_plot.append(lam)
                    index_r+=1
            lambda_plot=np.array(lambda_plot)
            x_matrix.append(x_plot)
            y_matrix.append(y_plot)
            lambd_matrix.append(lambda_plot)
            ax.set_title(r"$n=%i$" %(stanje))
            ax.plot3D(x_plot, y_plot, lambda_plot, color=plt.cm.coolwarm(k/len(lamb_list)), label=r"$\lambda=%.2f$" %(lam))
            ax.plot3D(x_plot, -np.array(y_plot,dtype=float), lambda_plot, color=plt.cm.coolwarm(k/len(lamb_list)))
            ax.set_xlabel("x")
            ax.set_ylabel("p")
            ax.set_zlabel(r"$\lambda$")
            k+=1
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plotting+=1
    plt.show()


x=np.linspace(-L,L,N)
h=x[1]-x[0]


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

def Lanczos(H,psi):
    Lv=np.zeros((len(psi),len(psi)), dtype=complex) #Creates matrix for Lanczos vectors
    Hk=np.zeros((len(psi),len(psi)), dtype=complex)
    print(Hk) #Creates matrix for the Hamiltonian in Krylov subspace
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
        
    return (Hk,Lv)

N=len(x)
M=N

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
    out=np.zeros(basis.shape[0],dtype=np.complex128)
    for i in range(basis.shape[1]):
        out+=basis[:,i]*calc_braket(x,basis[:,i],vector)*np.exp(-1j*eigval[i]*t)
    return out
lam=0
lamb_list=[0,0.01,0.1,0.5,0.75,1]
fig=plt.figure()
fig.suptitle(r"Prikaz $|\Psi(x,t)_0|$")
L=20
x=np.linspace(-L,L,10000)
N=50
h=x[1]-x[0]
#for i in range(len(lamb_list)):
#    ax = fig.add_subplot(2,3,i+1)
#    eigval,eigvec,basis,H=direct2(N,10000, lamb_list[i], L)
#    t=np.linspace(0,2*np.pi,10000)
#    psi_0=lastne_za_x(N, x, 0, eigvec)
#    psi_0=lastne_f(x,0)
#    Wawe1=[]
#    for tau in t:
#        Wawe1.append(time_propagation(psi_0,basis,eigval,x,tau))
#    Wawe1=np.array(Wawe1, dtype=complex)
#    ax.set_title(r"$\lambda=%.2f$" %(lamb_list[i]))
#    ax.imshow(abs(Wawe1),cmap="coolwarm")
#    ax.set_xlabel("x")
#    ax.set_ylabel("t")
#    ax.set_xticks([0, N/2, N-1], [-L,0,L])
#    ax.set_yticks([0, N/2, N-1], [0,"$%.2f$"%(max(t)/2), "$%.2f$"%(max(t))])
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
    eigval,eigvec,basis, H=direct2(N,100, lamb_list[j], L)
    psi_0=lastne_za_x(N, x, 0, eigvec)
    #psi_0=lastne_f(x,0)
    Wawe1=[]
    for i in range(len(t)):
        #Wawe0[i]/=np.sqrt(simpson(Wawe0[i]**2,x=x)) 
        solution=time_propagation(psi_0,basis,eigval,x,t[i])
        #solution/= np.sqrt(simpson(solution**2,x=x))
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