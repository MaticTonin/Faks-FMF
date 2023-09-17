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
L=10
h=0.25
tau=2*np.pi*h**2*0.5
x=np.arange(-L, L, h)
T=tau*len(x)

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
K=2



####### PLOTS TO SHOW #######
plot_3D=True
plot_Matrix=True
plot_animation=True
plot_lambda=True

def Potencial(x,lambd):
    return 1/2*x**2+lambd*x**4

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
    for i in range(N):
        Wawe.append(solution.copy())
        #solution=scipy.linalg.solve_banded()
        H_k=np.identity(N)
        for k in range(K):
            k+=1
            H_k=np.dot(H_k,H_matrix)
            solution+=(-1j*tau)**k/np.math.factorial(k)*np.dot(H_k,psi_0)
        psi_0=solution
    return np.asarray(Wawe, dtype=complex)

def lastne_f(x, stanje):
    funkcija_m=[]
    for i in range(len(x)):
        funkcija_m.append((alpha/(np.pi)**(1/2))**(1/2)*np.exp(-alpha**2*(x[i]-gamma)**2/2)) 
        #funkcija_m.append(alpha**(1/2)*(2. ** stanje  * np.math.factorial(stanje) * np.pi ** 0.5) ** (-0.5) *np.exp(-alpha**2*(x[i])**(2)/2) * special.hermite(stanje,0)(x[i]))
    return funkcija_m

H_matrix=H_pet(M,tau,h,lam)
Wawe0=propagator(H_matrix,lastne_f(x,0))
lambda_N=N
list_lambda=np.linspace(0,0.02,lambda_N)
psi_lambda=[]
j=0
for i in tqdm(list_lambda):
    H_matrix=H_pet(M,tau,h,i)
    psi_lambda.append(propagator(H_matrix,lastne_f(x,0))[2])
    j+=1
psi_lambda=np.array(psi_lambda)
if plot_lambda==True:
    X,Y=np.meshgrid(x,list_lambda)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(r" Real($\Psi(x,2\tau)), \tau=%.4f, h=%.4f$" %(tau,h))
    im=ax.plot_surface(X, Y, psi_lambda.real, rstride=1, cstride=1,
                    cmap='seismic', edgecolor='none')
    cbar = ax.figure.colorbar(im, label="Amplituda")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\lambda$")
    ax.legend()
    plt.show()

    X,Y=np.meshgrid(x,list_lambda)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(r" Imag($\Psi(x,2\tau)), \tau=%.4f, h=%.4f$" %(tau,h))
    im=ax.plot_surface(X, Y, psi_lambda.imag,
                    cmap='seismic', edgecolor='none')
    cbar = ax.figure.colorbar(im, label="Amplituda")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\lambda$")
    ax.legend()
    plt.show()

    X,Y=np.meshgrid(x,list_lambda)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(r"Abs($\Psi(x,2\tau)), \tau=%.4f, h=%.4f$" %(tau,h))
    im=ax.plot_surface(X, Y, abs(psi_lambda)**2,
                    cmap='seismic', edgecolor='none')
    cbar = ax.figure.colorbar(im, label="Amplituda")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\lambda$")
    ax.legend()
    plt.show()

    plt.subplot(2,2,1)
    plt.suptitle(r"Real($\Psi(x,\lambda))[t_0], \tau=%.4f,h=%.4f, K=%i$," %(tau,h,K))
    plt.title(r"$t_0=%.4f$" %t[1])
    psi_lambda_1=[]
    for i in tqdm(list_lambda):
        H_matrix=H_pet(M,tau,h,i)
        psi_lambda_1.append(propagator(H_matrix,lastne_f(x,0))[1])
    psi_lambda_1=np.array(psi_lambda_1)
    plt.imshow(psi_lambda_1.real)
    plt.xlabel("x")
    plt.ylabel(r"$\lambda$")
    plt.xticks([0, M/2, M-1], [-L,0,L])
    plt.yticks([0,lambda_N/2, lambda_N], [0,max(list_lambda)/2, max(list_lambda)])

    plt.subplot(2,2,2)
    plt.title(r"$t_0=%.4f$" %t[2])
    psi_lambda_1=[]
    for i in tqdm(list_lambda):
        H_matrix=H_pet(M,tau,h,i)
        psi_lambda_1.append(propagator(H_matrix,lastne_f(x,0))[2])
    psi_lambda_1=np.array(psi_lambda_1)
    plt.imshow(psi_lambda_1.real)
    plt.xlabel("x")
    plt.ylabel(r"$\lambda$")
    plt.xticks([0, M/2, M-1], [-L,0,L])
    plt.yticks([0,lambda_N/2, lambda_N], [0,max(list_lambda)/2, max(list_lambda)])

    plt.subplot(2,2,3)
    plt.title(r"$t_0=%.4f$" %t[5])
    psi_lambda_1=[]
    for i in tqdm(list_lambda):
        H_matrix=H_pet(M,tau,h,i)
        psi_lambda_1.append(propagator(H_matrix,lastne_f(x,0))[5])
    psi_lambda_1=np.array(psi_lambda_1)
    plt.imshow(psi_lambda_1.real)
    plt.xlabel("x")
    plt.ylabel(r"$\lambda$")
    plt.xticks([0, M/2, M-1], [-L,0,L])
    plt.yticks([0,lambda_N/2, lambda_N], [0,max(list_lambda)/2, max(list_lambda)])

    plt.subplot(2,2,4)
    plt.title(r"$t_0=%.4f$" %t[10])
    psi_lambda_1=[]
    for i in tqdm(list_lambda):
        H_matrix=H_pet(M,tau,h,i)
        psi_lambda_1.append(propagator(H_matrix,lastne_f(x,0))[10])
    psi_lambda_1=np.array(psi_lambda_1)
    plt.imshow(psi_lambda_1.real)
    plt.xlabel("x")
    plt.ylabel(r"$\lambda$")
    plt.xticks([0, M/2, M-1], [-L,0,L])
    plt.yticks([0,lambda_N/2, lambda_N], [0,max(list_lambda)/2, max(list_lambda)])
    plt.show()
print(Wawe0)
H_matrix=H_pet(M,tau,h,lam)
Wawe0=propagator(H_matrix,lastne_f(x,0))
print(Wawe0)
Test=N
if plot_Matrix==True:
    plt.subplot(2,1,1)
    plt.suptitle("Prikaz časovnega razvoja valovne funkcije, $\lambda=%.2f, \tau=%.4f, h=%.4f, K=%i$," %(lam,tau,h,K))
    plt.title(r"$|\Psi(x)|^2$")
    plt.xlabel("x")
    plt.ylabel("t")
    #plt.xticks([0, M/2, M-1], [-L,0,L])
    #plt.yticks([0, M/2, M-1], [0,max(t)/2, max(t)])
    plt.imshow(abs(Wawe0[:Test][:Test]))
    plt.subplot(2,2,3)
    plt.title(r"Real($\Psi(x)$)")
    plt.xlabel("x")
    plt.ylabel("t")
    #plt.xticks([0, M/2, M-1], [-L,0,L])
    #plt.yticks([0, M/2, M-1], [0,max(t)/2, max(t)])
    plt.imshow(Wawe0[:Test][:Test].real)
    plt.subplot(2,2,4)
    plt.title(r"Imag($\Psi(x)$)")
    plt.xlabel("x")
    plt.ylabel("t")
    #plt.xticks([0, M/2, M-1], [-L,0,L])
    #plt.yticks([0, M/2, M-1], [0,max(t)/2, max(t)])
    plt.imshow(Wawe0[:Test][:Test].imag)
    plt.show()


if plot_3D==True:
    X,Y=np.meshgrid(x,t)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(r" Real($\Psi(x,t)), \lambda=%.2f, \tau=%.4f, h=%.4f$" %(lam,tau,h))
    im=ax.plot_surface(X, Y, Wawe0.real, rstride=1, cstride=1,
                    cmap='seismic', edgecolor='none')
    cbar = ax.figure.colorbar(im, label="Amplituda")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(r" Imag($\Psi(x,t)), \lambda=%.2f \tau=%.4f, h=%.4f$" %(lam,tau,h))
    im=ax.plot_surface(X, Y, Wawe0.imag, rstride=1, cstride=1,
                    cmap='seismic', edgecolor='none')
    cbar = ax.figure.colorbar(im, label="Amplituda")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.legend()
    plt.show()
if plot_animation==True:
    #Wawe1=propagator(H_matrix,lastne_f(x,1))

    fig, ax = plt.subplots()

    #line2, = ax.plot(x, x/40, color="white")
    line1, = ax.plot(x, Wawe0[0,:N].real, color="red", label="Lastno stanje 0")
    line3, = ax.plot(x, abs(Wawe0[0,:N]), color="blue", label="ABS Lastno stanje 0")
    #line2, = ax.plot(x, Wawe0[0,:N].imag, color="black", label="Lastno stanje 0, imag")
    #line3, = ax.plot(x, abs(Wawe0[0,:N]), color="green", label="Lastno stanje 0, abs")
    #line3, = ax.plot(x, Wawe1[0,:N].real, color="blue", label="Lastno stanje 1")





    def animate(i):
        line1.set_ydata(Wawe0[i,:N].real)
        line3.set_ydata(abs(Wawe0[i,:N]))
        #line3.set_ydata(abs(Wawe0[i,:N]))
        #line3.set_ydata(Wawe1[i,:N].imag)  # update the data.
        return  line1 #,line3


    ani = animation.FuncAnimation(
        fig, animate, interval=20, frames=N, blit=True, save_count=50)

    # To save the animation, use e.g.
    #
    # ani.save("movie.mp4")
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