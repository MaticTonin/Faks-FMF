from scipy.integrate import simpson
from scipy.special import  factorial, hermite
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad
import numpy as np
import scipy



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
from tqdm import tqdm 

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
            eigval,eigvec,basis,H=Eigval(N,10000, lamb_list[j], L)
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
    eigval,eigvec,basis=Eigval(N,10000, lambd, L)
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
        eigval,eigvec,basis,H=Eigval(N,100, lamb_list[j], L)
        ax.axvline(x=plostina*N, ymin=0, ymax=350000, color=plt.cm.coolwarm(j/len(lamb_list)),ls='--', label=r"$r\cdot N=%.i$" %(plostina*N+1))
        ax.plot(n_space,np.sort(abs(eigval)), "-", color=plt.cm.coolwarm(j/len(lamb_list)),label="$\lambda=%.2f$" %(lamb_list[j]))
        ax.set_xlabel("n")
        ax.set_ylabel(r"$E_n$")
        ax.set_yscale("log")
        l+=1
        ax.legend()
    plt.show()