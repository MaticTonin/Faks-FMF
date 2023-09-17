import numpy as np
import matplotlib.pyplot as plt
import time 
from numba import njit
from scipy.stats import norm
import matplotlib.mlab as mlab
import scipy
def U_matrix(z):
    return np.exp(-z)*np.array([[np.exp(2*z),0,0,0],[0,np.cosh(2*z),np.sinh(2*z),0],[0,np.sinh(2*z),np.cosh(2*z),0], [0,0,0,np.exp(2*z)]], dtype=np.complex128)

DEC_INT = np.array([[int(f'0b{i}{j}',2) for j in range(2)] for i in range(2)])


def new_psi(n):
    psi = np.random.randn(2**n)/(2**n)
    norm=np.sum(psi**2)
    psi/=np.sqrt(norm)
    psi=psi.astype(np.complex128)

    return psi

@njit
def get_b_index(i,j):
    i=i//2**j 
    return i%2
 

@njit
def n_index_change(x,d,n):
    d0=get_b_index(x,n)
    if d0==d: return x
    bn=2**n
    x-=d0*bn
    x+=d*bn

    return x


@njit
def cal_psi_j(psi, U, j_mat, N):
    psi_new= np.zeros(2**N, dtype=np.complex128)
    jp1=(j_mat+1)%N
    for i in range(2**N):
        bj=get_b_index(i, j_mat)
        bjp1=get_b_index(i,jp1)
        row=bj*2+bjp1
        for j in range(2):
            for k in range(2):
                new_ind=n_index_change(i,j,j_mat)
                new_ind=n_index_change(new_ind,k,jp1)
                psi_new[i]+= U[row, DEC_INT[j,k]]*psi[new_ind]
    return psi_new

def exp_2j(psi,z,n):
    U=U_matrix(z)
    for j in range(int(n/2)):
        psi=cal_psi_j(psi,U,2*j,n)
    return psi


def exp_2j_1(psi,z,n):
    U=U_matrix(z)
    for j in range(int(n/2)):
        psi=cal_psi_j(psi,U,2*j+1,n)
    return psi


def S_2(psi,z,n):
    psi=exp_2j(psi,z/2,n)
    psi=exp_2j_1(psi,z,n)
    psi=exp_2j(psi,z/2,n)
    return psi



def S_4(psi, z,n):
    x_0=-2**(1/3)/(2-2**(1/3))
    x_1=1/(2-2**(1/3))
    psi=S_2(psi,x_1*z,n)
    psi=S_2(psi,x_0*z,n)
    psi=S_2(psi,x_1*z,n)
    return psi


def generator_Z(psi,beta,tau,n):
    K=int((beta/tau)/2)
    beta_list=np.zeros(K+1)
    Z_generator=np.zeros(K+1)
    
    beta_list[0]=0
    Z_generator[0]=np.sum(np.abs(psi)**2)

    for i in range(K):
        psi = S_4(psi,-tau,n)
        beta_list[i+1] = beta_list[i] + tau
        Z=np.sum(abs(psi)**2)
        Z_generator[i+1] = Z

    return beta_list*2,Z_generator

def making_Z(generator,beta,tau, n):
    psi_0_list=[]
    for i in range(generator):
        psi_0_list.append(new_psi(n))
    psi_0_list=np.array(psi_0_list)
    beta_plot=[]
    Z_list=[]
    jndex=0
    for psi in psi_0_list:
        K=int((beta/tau)/2)
        beta_list=np.zeros(K+1)
        Z_generator=np.zeros(K+1)
    
        beta_list[0]=0
        Z_generator[0]=np.sum(np.abs(psi)**2)

        for i in range(K):
            psi = S_4(psi,-tau,n)
            beta_list[i+1] = beta_list[i] + tau
            Z=np.sum(abs(psi)**2)
            Z_generator[i+1] = Z
        if jndex==0:
            beta_plot=beta_list*2
            jndex+=1
        Z_list.append(Z_generator)
    Z_list=np.array(Z_list)
    Z_plot=np.average(Z_list,axis=0)

    return beta_plot, Z_plot

def H_generator(psi,n):
    psi_new=np.zeros(2**n, dtype=np.complex128)
    for j in range(n):
        jp1=(j+1)%n

        for i in range(len(psi)):
            bj=get_b_index(i,j)
            bjp1=get_b_index(i,jp1)
            bdiff = bj - bjp1
            psi_new[i] += (-1)**bdiff*psi[i]

            if bdiff !=0:
                new_ind=i+bjp1**(jp1)-bj**j
                psi_new += 2*psi[new_ind]
    return psi_new

def making_H(generator,beta,tau, n):
    psi_0_list=[]
    for i in range(generator):
        psi_0_list.append(new_psi(n))
    psi_0_list=np.array(psi_0_list)
    beta_plot=[]
    Z_list=[]
    H_list=[]
    jndex=0
    for psi in psi_0_list:
        K=int((beta/tau)/2)
        beta_list=np.zeros(K+1)
        H_g=np.zeros(K+1)
        Z_g=np.zeros(K+1)
    
        beta_list[0]=0
        Z_g[0]=np.sum(np.abs(psi)**2)
        H_g[0] = np.sum(np.conjugate(psi)*H_generator(psi,n)).real

        for i in range(K):
            psi = S_4(psi,-tau,n)
            beta_list[i+1] = beta_list[i] + tau
            H_g[i+1] = np.sum(np.conjugate(psi)*H_generator(psi,n))
            Z=np.sum(abs(psi)**2)
            Z_g[i+1] = Z
        if jndex==0:
            beta_plot=beta_list*2
            jndex+=1
        H_list.append(H_g)
        Z_list.append(Z_g)
    Z_list=np.array(Z_list)
    Z_plot=np.average(Z_list,axis=0)
    H_list=np.array(H_list)
    H_plot=np.average(H_list,axis=0)

    H_plot /= Z_plot

    return beta_plot, Z_plot, H_plot

def sigma_z(psi):
    psi[1::2]*=-1
    return psi


def making_sigma_z(generator,t,tau, n):
    psi_0_list=[]
    for i in range(generator):
        psi_0_list.append(new_psi(n))
    psi_0_list=np.array(psi_0_list)
    t_plot=[]
    korel_list=[]
    jndex=0
    for psi in psi_0_list:
        K=int((t/tau))
        t_list=np.zeros(K+1)
        korel_generated=np.zeros(K+1,dtype=np.complex128)
        chi =sigma_z(psi.copy())
        t_list[0]=0
        korel_generated[0]=np.sum(np.conjugate(psi)*sigma_z(chi.copy()))

        for i in range(K):
            psi = S_4(psi,-1j*tau,n)
            chi = S_4(chi,-1j*tau,n)
            t_list[i+1] = t_list[i] + tau
            korel_generated[i+1] = np.sum(np.conjugate(psi)*sigma_z(chi.copy()))

        if jndex==0:
            t_plot=t_list
            jndex+=1
        korel_list.append(korel_generated)
    korel_list=np.array(korel_list)
    korel_plot=np.average(korel_list,axis=0)

    return t_plot, korel_plot

def J_operator(psi,n,j):
    psi_new=np.zeros(2**n, dtype=np.complex128)
    jp1=(j+1)%n
    for i in range(len(psi)):
        bj=get_b_index(i,j)
        bjp1=get_b_index(i,jp1)
        new_ind = n_index_change(i,bjp1,j)
        new_ind = n_index_change(new_ind,bj,jp1)
        psi_new[i] = -2*1j*(-1)**bj*abs(bj-bjp1)*psi[new_ind]
    return psi_new

def making_J_r(generator, t, tau, N,j):
    psi_0_list=[]
    for i in range(generator):
        psi_0_list.append(new_psi(N))
    psi_0_list=np.array(psi_0_list, dtype=np.complex128)
    t_plot=[]
    J_korel_list=[]
    jndex=0
    for psi in psi_0_list:
        K=int((t/tau))
        t_list=np.zeros(K+1)
        J_korel_generated=np.zeros(K+1,dtype=np.complex128)
        chi =J_operator(psi,N,1)
        t_list[0]=0
        J_korel_generated[0]=np.sum(np.conjugate(psi)*J_operator(chi,N,j))
        for i in range(K):
            psi = S_4(psi,-1j*tau,N)
            chi = S_4(chi,-1j*tau,N)
            t_list[i+1] = t_list[i] + tau
            J_korel_generated[i+1] = np.sum(np.conjugate(psi)*J_operator(chi,N,j))
        if jndex==0:
            t_plot=t_list
            jndex+=1
        J_korel_list.append(J_korel_generated)
    J_korel_list=np.array(J_korel_list)
    J_korel_plot=np.average(J_korel_list,axis=0)
    return t_plot, J_korel_plot

def making_J(generator, t, tau, N, R):
    K=int((t/tau))
    J_t=np.zeros(K+1, dtype=np.complex128)
    for r in tqdm(range(R)):
        t_plot,J_r=making_J_r(generator,t,tau,N,r)
        J_t+=J_r
    return t_plot, J_t


from tqdm import tqdm

if __name__ == "__main__":
    J_plot="No"
    if J_plot=="Yes":
        N_list=[2,4,6,8,10]
        index=0
        tau=10**(-2)
        generator=40
        beta=5
        fig3 = plt.figure()
        fig3.suptitle(r"Prikaz Real($\langle J(t)J(0) \rangle_\beta)$ pri $\tau=%.3f$, $N_\Psi=%i$" %(tau,generator))
        ax3 = fig3.add_subplot(1,1,1)
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig2.suptitle(r"Prikaz Imag($\langle J(t)J(0) \rangle_\beta)$ pri $\tau=%.3f$, $N_\Psi=%i$" %(tau,generator))
        ax.set_title(r"Abs($\langle J(t)J(0) \rangle_\beta$)")
        ax1.set_title(r"Real($\langle J(t)J(0) \rangle_\beta$)")
        for N in tqdm(N_list):
            t_plot, korel_plot=making_J(generator,beta,tau,N,N)
            ax.plot(t_plot,abs(korel_plot), label="n=$%i$" %N, color=plt.cm.autumn(index/len(N_list)))
            ax1.plot(t_plot,korel_plot.real, label="n=$%i$" %N, color=plt.cm.winter(index/len(N_list)))
            ax2.plot(t_plot,korel_plot.imag, label="n=$%i$" %N, color=plt.cm.brg(index/len(N_list)))
            ax3.plot(t_plot,(korel_plot.real), label="n=$%i$" %N, color=plt.cm.brg(index/len(N_list)))
            ax.set_xlabel(r'$t$')
            ax.set_ylabel(r'$\langle J(t)J(0)  \rangle_\beta$')
            ax1.set_xlabel(r'$t$')
            ax1.set_ylabel(r'$Real(\langle J(t)J(0)  \rangle_\beta)$')
            ax2.set_xlabel(r'$t$')
            ax2.set_ylabel(r'$Imag(\langle J(t)J(0)  \rangle_\beta)$')
            ax3.set_xlabel(r'$t$')
            ax3.set_ylabel(r'$Real(\langle J(t)J(0)  \rangle_\beta)$')
            index+=1
        ax.grid()
        ax.legend()
        ax1.grid()
        ax1.legend()
        ax2.grid()
        ax2.legend()
        ax3.grid()
        ax3.legend()
        plt.show()
        plt.close()

    j_z_plot="Y"
    if j_z_plot=="Yes":
        N_list=[8,10,12,14,16]
        index=3
        tau=10**(-1)
        generator=3
        beta=4
        fig = plt.figure()
        slika=1
        fig.suptitle(r"Prikaz $(\langle J(t)J(0)  \rangle_\beta)$ pri $\tau=%.3f$, $N_\Psi=%i$" %(tau,generator))
        ax = fig.add_subplot(2,1,1)
        ax1= fig.add_subplot(2,2,3)
        ax2= fig.add_subplot(2,2,4)
        fig1=plt.figure()
        fig1.suptitle(r"Prikaz realne in imaginarne $(\langle J(t)J(0)  \rangle_\beta)$ pri $\tau=%.3f$, $N_\Psi=%i$" %(tau,generator))
        ax11 = fig1.add_subplot(2,1,1)
        ax12 = fig1.add_subplot(2,1,2)
        ax.set_title(r"Abs($(\langle J(t)J(0)  \rangle_\beta)$)")
        ax1.set_title(r"Real($(\langle J(t)J(0)  \rangle_\beta)$)")
        ax2.set_title(r"Imag($(\langle J(t)J(0)  \rangle_\beta)$)")
        TIME=[]
        t_plot_ref, korel_plot_ref = making_J(generator,beta,tau,16,16)
        for N in tqdm(N_list):
            if 3<index-2<=6 and slika!=2:
                slika+=1 
                index=6       
            start_time = time.time()
            t_plot, korel_plot = making_J(generator,beta,tau,N,N)
            TIME.append(time.time() -start_time)
            ax.plot(t_plot,abs(korel_plot), label="n=$%i$" %N, color=plt.cm.Blues(index/(len(N_list)+3)))
            ax1.plot(t_plot,korel_plot.real, label="n=$%i$" %N, color=plt.cm.Greens(index/(len(N_list)+3)))
            ax2.plot(t_plot,korel_plot.imag, label="n=$%i$" %N, color=plt.cm.Reds(index/(len(N_list)+3)))
            ax11.plot(t_plot,korel_plot.real, label="Real, n=$%i$" %N, color=plt.cm.Greens(index/(len(N_list)+3)))
            ax11.plot(t_plot,korel_plot.imag, label="Imag, n=$%i$" %N, color=plt.cm.Reds(index/(len(N_list)+3)))
            ax12.plot(t_plot,abs(korel_plot.real-korel_plot_ref.real), label="n=$%i$" %N, color=plt.cm.Greens(index/(len(N_list)+3)))
            ax.set_xlabel(r'$t$')
            ax.set_ylabel(r'$(\langle J(t)J(0)  \rangle_\beta)$')
            ax1.set_xlabel(r'$t$')
            ax1.set_ylabel(r'$Real((\langle J(t)J(0)  \rangle_\beta))$')
            ax2.set_xlabel(r'$t$')
            ax2.set_ylabel(r'$Imag((\langle J(t)J(0)  \rangle_\beta))$')
            ax11.set_xlabel(r'$t$')
            ax11.set_ylabel(r'$(\langle J(t)J(0)  \rangle_\beta)$')
            ax11.grid()
            ax11.legend()
            ax12.set_xlabel(r'$t$')
            ax12.set_ylabel(r'$|(\langle J(t)J(0)  \rangle_\beta)_n-(\langle J(t)J(0)  \rangle_\beta)_{12}|$')
            ax12.grid()
            ax12.legend()
            ax12.set_yscale("log")  
            index+=1
        ax.grid()
        ax.legend()
        ax1.grid()
        ax1.legend()
        ax2.grid()
        ax2.legend()
        ax11.grid()
        ax12.grid()
        plt.show()
        plt.title(r"Prikaz časovne zahtevnosti programa za $\langle J(t)J(0) \rangle_\beta)(n)$ od izbire n $\tau=%.3f$, $N_\Psi=%i$" %(tau,generator))
        plt.plot(N_list,TIME)
        plt.xlabel("n")
        plt.ylabel("t[s]")
        plt.grid()
        plt.show()
        plt.close()

    sigma_z_plot="Y"
    if sigma_z_plot=="Yes":
        N_list=[2,6,10,14]
        index=3
        tau=10**(-2)
        generator=5
        beta=2
        fig = plt.figure()
        slika=1
        fig.suptitle(r"Prikaz $\langle \sigma_1^z(t)\sigma_{1}^z(0) \rangle_\beta$ pri $\tau=%.3f$, $N_\Psi=%i$" %(tau,generator))
        ax = fig.add_subplot(2,1,1)
        ax1= fig.add_subplot(2,2,3)
        ax2= fig.add_subplot(2,2,4)
        fig1=plt.figure()
        fig1.suptitle(r"Prikaz realne in imaginarne $\langle \sigma_1^z(t)\sigma_{1}^z(0) \rangle_\beta/N$ pri $\tau=%.3f$, $N_\Psi=%i$" %(tau,generator))
        ax11 = fig1.add_subplot(2,1,1)
        ax12 = fig1.add_subplot(2,1,2)
        ax.set_title(r"Abs($\langle  \sigma_1^z(t)\sigma_{1}^z(0) \rangle_\beta$)")
        ax1.set_title(r"Real($\langle  \sigma_1^z(t)\sigma_{1}^z(0) \rangle_\beta$)")
        ax2.set_title(r"Imag($\langle  \sigma_1^z(t)\sigma_{1}^z(0) \rangle_\beta$)")
        TIME=[]
        t_plot_ref, korel_plot_ref = making_sigma_z(generator,beta,tau,16)
        for N in tqdm(N_list):
            if 3<index-2<=6 and slika!=2:
                slika+=1 
                index=6       
            start_time = time.time()
            t_plot, korel_plot = making_sigma_z(generator,beta,tau,N)
            TIME.append(time.time() -start_time)
            ax.plot(t_plot,abs(korel_plot), label="n=$%i$" %N, color=plt.cm.Blues(index/(len(N_list)+3)))
            ax1.plot(t_plot,korel_plot.real, label="n=$%i$" %N, color=plt.cm.Greens(index/(len(N_list)+3)))
            ax2.plot(t_plot,korel_plot.imag, label="n=$%i$" %N, color=plt.cm.Reds(index/(len(N_list)+3)))
            ax11.plot(t_plot,korel_plot.real, label="Real, n=$%i$" %N, color=plt.cm.Greens(index/(len(N_list)+3)))
            ax11.plot(t_plot,korel_plot.imag, label="Imag, n=$%i$" %N, color=plt.cm.Reds(index/(len(N_list)+3)))
            ax12.plot(t_plot,abs(korel_plot.real-korel_plot_ref.real), label="n=$%i$" %N, color=plt.cm.Greens(index/(len(N_list)+3)))
            ax.set_xlabel(r'$t$')
            ax.set_ylabel(r'$\langle \sigma_1^z(t)\sigma_{1}^z(0) \rangle_\beta$')
            ax1.set_xlabel(r'$t$')
            ax1.set_ylabel(r'$Real(\langle \sigma_1^z(t)\sigma_{1}^z(0) \rangle_\beta)$')
            ax2.set_xlabel(r'$t$')
            ax2.set_ylabel(r'$Imag(\langle \sigma_1^z(t)\sigma_{1}^z(0) \rangle_\beta)$')
            ax11.set_xlabel(r'$t$')
            ax11.set_ylabel(r'$\langle \sigma_1^z(t)\sigma_{1}^z(0) \rangle_\beta$')
            ax11.grid()
            ax11.legend()
            ax12.set_xlabel(r'$t$')
            ax12.set_ylabel(r'$|C_{\beta}(n)-C_{\beta}(16)|$')
            ax12.grid()
            ax12.legend()
            ax12.set_yscale("log")  
            index+=1
        ax.grid()
        ax.legend()
        ax1.grid()
        ax1.legend()
        ax2.grid()
        ax2.legend()
        ax11.grid()
        ax12.grid()
        plt.show()
        plt.title(r"Prikaz časovne zahtevnosti programa za izračun $C_{\beta}(n)$ od izbire n $\tau=%.3f$, $N_\Psi=%i$" %(tau,generator))
        plt.plot(N_list,TIME)
        plt.xlabel("n")
        plt.ylabel("t[s]")
        plt.grid()
        plt.show()
        plt.close()

    Fn_plot="No"
    if Fn_plot=="Yes":    
        N_list=[2,4,6,8,10,12]
        index=0
        tau=10**(-2)
        generator=40
        beta=5
        plt.title(r"Prikaz proste energije $F(\beta)/n$ pri $\tau=%.3f$, $N_\Psi=%i$" %(tau,generator))
        for N in tqdm(N_list):
            beta_plot,Z_plot = making_Z(generator,beta,tau,N)
            #plt.plot(beta_plot[1:],Z_plot[1:], label="n=$%i$" %N, color=plt.cm.coolwarm(index/len(N_list)))
            plt.plot(beta_plot[1:],-np.log(Z_plot[1:])/beta_plot[1:]*1/N, label="n=$%i$" %N, color=plt.cm.coolwarm(index/len(N_list)))
            plt.xlabel(r'$\beta$')
            plt.ylabel(r'$F(\beta)/n$')
            #plt.yscale("log")
            index+=1
        plt.grid()
        plt.legend()
        plt.show()
        plt.close()
    Hn_plot="Yes"
    if Hn_plot=="Yes":
        N_list=[2,4,6,8,10,12]
        index=0
        tau=10**(-2)
        generator=10
        beta=4
        plt.title(r"Prikaz $\langle H \rangle_\beta/n$ pri $\tau=%.3f$, $N_\Psi=%i$" %(tau,generator))
        for N in tqdm(N_list):
            beta_plot,Z_plot, H_plot = making_H(generator,beta,tau,N)
            plt.plot(beta_plot,H_plot/N, label="n=$%i$" %N, color=plt.cm.coolwarm(index/len(N_list)))
            plt.xlabel(r'$\beta$')
            plt.ylabel(r'$\langle H \rangle_\beta/n$')
            index+=1
        plt.grid()
        plt.legend()
        plt.show()
        plt.close()
    H_plot="No"
    if H_plot=="Yes":
        N_list=[2,4,6,8,10,12]
        index=0
        tau=10**(-2)
        generator=10
        beta=4
        plt.title(r"Prikaz $\langle H \rangle_\beta$ pri $\tau=%.3f$, $N_\Psi=%i$" %(tau,generator))
        for N in tqdm(N_list):
            beta_plot,Z_plot, H_plot = making_H(generator,beta,tau,N)
            plt.plot(beta_plot,H_plot, label="n=$%i$" %N, color=plt.cm.coolwarm(index/len(N_list)))
            plt.xlabel(r'$\beta$')
            plt.ylabel(r'$\langle H \rangle_\beta$')
            index+=1
        plt.grid()
        plt.legend()
        plt.show()
        plt.close()

    F_dis_plot="No"
    if F_dis_plot=="Yes":
        N_list=[2,4,6,8,10]
        index=0
        tau=10**(-2)
        #generator_list=[2,5,10,15,20,30,40]
        generator_list=[2,5,10,20,40,50,60]
        generator_0=50
        beta=10
        slika=1
        fig = plt.figure()
        fig.suptitle(r"Prikaz razporeditve $F(\beta)$ pri $\tau=%.3f$, za več generiranih stanj" %(tau))
        fig2 = plt.figure()
        fig2.suptitle(r"Prikaz povprečnih $F(10)$ pri $\tau=%.3f$" %(tau))
        ax2 = fig2.add_subplot(1,1,1)
        repeat=2
        import time
        time_list_generator=[]
        time_list_n=[]
        index=0
        mu_list=[]
        sigma_list=[]
        for N in tqdm(N_list):
            Z_list=[]
            n_list=[]
            ax = fig.add_subplot(5,2,2*index+1)
            ax1 = fig.add_subplot(5,2,2*index+2)
            ax.set_title("$n=%i$" %N)
            ax1.set_title("$n=%i$" %N)
            for l in range(repeat):
                time_list_generator=[]
                start_time = time.time()
                beta_plot,Z_plot = making_Z(generator,beta,tau,N)
                Z_plot=np.array(Z_plot)
                time_list_generator.append(time.time() - start_time)
                F=-np.log(Z_plot[len(Z_plot)-1])/beta_plot[len(Z_plot)-1]
                Z_list.append(F)
                n_list.append(l)
            (mu, sigma) = norm.fit(Z_list)
            mu_list.append(mu)
            sigma_list.append(sigma)
            ax.scatter(n_list,Z_list,color=plt.cm.brg(index/len(generator_list)), label="n=%i" %N)
            n, bins, patches = ax1.hist(Z_list,bins=10,alpha=0.5 ,color=plt.cm.brg(index/len(generator_list)), orientation="horizontal", density=True)
            y = scipy.stats.norm.pdf( bins, mu, sigma)
            ax1.plot(y, bins, '--', label=r"$\mu=%.2f, \sigma=%.3f$" %(mu,sigma),linewidth=2,color=plt.cm.brg(index/len(generator_list)))
            ax.set_xlabel("Repetition")
            ax.set_ylabel(r"$F(\beta)$")
            ax1.set_xlabel("$N$")
            ax1.set_ylabel(r"$F(\beta)$")
            ax.grid()
            ax1.grid()
            ax1.legend()
            index+=1
        mu_list=np.array(mu_list)
        sigma_list=np.array(sigma_list)
        print(mu_list-sigma_list, mu_list+sigma_list)
        ax2.plot(N_list,mu_list, color="red")
        ax2.fill_between(N_list,mu_list-sigma_list,mu_list+sigma_list,color="red", alpha=0.7)
        ax2.grid()
        plt.show()
        plt.close()
    H_dis_plot="No"
    if H_dis_plot=="Yes":
        N_list=[2,4,6,8,10]
        index=0
        tau=10**(-2)
        #generator_list=[2,5,10,15,20,30,40]
        generator_list=[2,5,10,20,40,50,60]
        generator_0=40
        beta=2
        slika=1
        fig = plt.figure()
        fig.suptitle(r"Prikaz razporeditve $\langle H \rangle_\beta$ pri $\tau=%.3f$, za več generiranih stanj" %(tau))
        fig2 = plt.figure()
        fig2.suptitle(r"Prikaz povprečnih $\langle H \rangle_{2}$ pri $\tau=%.3f$" %(tau))
        ax2 = fig2.add_subplot(1,1,1)
        repeat=40
        import time
        time_list_generator=[]
        time_list_n=[]
        index=0
        mu_list=[]
        sigma_list=[]
        for N in tqdm(N_list):
            Z_list=[]
            n_list=[]
            ax = fig.add_subplot(4,2,2*index+1)
            ax1 = fig.add_subplot(4,2,2*index+2)
            ax.set_title("$n=%i$" %N)
            ax1.set_title("$n=%i$" %N)
            for l in range(repeat):
                time_list_generator=[]
                start_time = time.time()
                beta_plot,Z_plot, H_plot= making_H(generator,beta,tau,N)
                Z_plot=np.array(Z_plot)
                time_list_generator.append(time.time() - start_time)
                F=-np.log(Z_plot[len(Z_plot)-1])/beta_plot[len(Z_plot)-1]
                Z_list.append(H_plot[len(Z_plot)-1])
                n_list.append(l)
            (mu, sigma) = norm.fit(Z_list)
            mu_list.append(mu)
            sigma_list.append(sigma)
            ax.scatter(n_list,Z_list,color=plt.cm.brg(index/len(generator_list)), label="n=%i" %N)
            n, bins, patches = ax1.hist(Z_list,bins=15,alpha=0.5 ,color=plt.cm.brg(index/len(generator_list)), orientation="horizontal", density=True)
            y = scipy.stats.norm.pdf( bins, mu, sigma)
            ax1.plot(y, bins, '--', label=r"$\mu=%.2f, \sigma=%.5f$" %(mu,sigma),linewidth=2,color=plt.cm.brg(index/len(generator_list)))
            ax.set_xlabel("Repetition")
            ax.set_ylabel(r"$\langle H \rangle_\beta$")
            ax1.set_xlabel("$N$")
            ax1.set_ylabel(r"$\langle H \rangle_\beta$")
            ax.grid()
            ax1.grid()
            ax1.legend()
            index+=1
        mu_list=np.array(mu_list)
        sigma_list=np.array(sigma_list)
        ax2.plot(N_list,mu_list, color="red")
        ax2.fill_between(N_list,mu_list-sigma_list,mu_list+sigma_list,color="red", alpha=0.7)
        ax2.grid()
        plt.show()
        plt.close()
    pov_F_plot="N"
    if pov_F_plot=="Yes":
        N_list=[2,3,4,5,6]
        index=0
        tau=10**(-2)
        #generator_list=[2,5,10,15,20,30,40]
        generator_list=[2,5,10,20,40,50,60]
        generator_0=40
        beta=4
        slika=1
        fig = plt.figure()
        fig.suptitle(r"Prikaz povprečnih $F(2)$ pri $\tau=%.3f$" %(tau))
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure()
        fig2.suptitle(r"Prikaz povprečnih $\langle H \rangle_{2}$ pri $\tau=%.3f$" %(tau))
        ax2 = fig2.add_subplot(1,1,1)
        repeat=40
        import time
        time_list_generator=[]
        time_list_n=[]
        index=0
        mu_list_Z=[]
        sigma_list_Z=[]
        mu_list_H=[]
        sigma_list_H=[]
        for N in tqdm(N_list):
            Z_list=[]
            n_list=[]
            H_list=[]
            n_list=[]
            for l in range(repeat):
                time_list_generator=[]
                start_time = time.time()
                beta_plot,Z_plot, H_plot= making_H(generator,beta,tau,N)
                Z_plot=np.array(Z_plot)
                time_list_generator.append(time.time() - start_time)
                F=-np.log(Z_plot[len(Z_plot)-1])/beta_plot[len(Z_plot)-1]
                H_list.append(H_plot[len(Z_plot)-1])
                Z_list.append(F)
                n_list.append(l)
            (mu, sigma) = norm.fit(Z_list)
            mu_list_Z.append(mu)
            sigma_list_Z.append(sigma)
            (mu, sigma) = norm.fit(H_list)
            mu_list_H.append(mu)
            sigma_list_H.append(sigma)
            index+=1
        mu_list_H=np.array(mu_list_H)
        sigma_list_H=np.array(sigma_list_H)
        mu_list_Z=np.array(mu_list_Z)
        sigma_list_Z=np.array(sigma_list_Z)
        ax2.plot(N_list,mu_list_H, color="red")
        ax2.fill_between(N_list,mu_list_H-sigma_list_H,mu_list_H+sigma_list_H,color="red", alpha=0.7)
        ax2.grid()
        ax.plot(N_list,mu_list_Z, color="red")
        ax.fill_between(N_list,mu_list_Z-sigma_list_Z,mu_list_Z+sigma_list_Z,color="red", alpha=0.7)
        ax2.set_xlabel(r"$N$")
        ax2.set_ylabel(r"$\langle H \rangle_4$")
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"$F(4)$")
        ax.grid()
        plt.show()
        plt.close()


        generator_list=[2,5,10,20,40]
        repeat=40
        import time
        time_list_generator=[]
        time_list_n=[]
        index=0
        mu_list=[]
        sigma_list=[]
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        N_list=[2,4,6,8,10,12,14]
        tau=10**(-2)
        generator=40
        beta=6
        jndex=0
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig2.suptitle(r"Prikaz spreminjanja z generatorjem $F(\beta)$ pri $\tau=%.3f$" %(tau))
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)
        fig3.suptitle(r"Prikaz povprečnih $F(\beta)$ pri $\tau=%.3f$" %(tau))
        mu_together=[]
        sigma_together=[]
        for N in tqdm(N_list):
            fig = plt.figure()
            fig.suptitle(r"Prikaz razporeditve $F(\beta)$ pri $\tau=%.3f$ pri $n=%i$" %(tau,N))
            index=0
            mu_list=[]
            sigma_list=[]
            for generator in tqdm(generator_list):
                Z_list=[]
                n_list=[]
                ax = fig.add_subplot(5,2,2*index+1)
                ax1 = fig.add_subplot(5,2,2*index+2)
                ax.set_title(r"$n_\psi=%i$" %generator)
                ax1.set_title(r"$n_\psi=%i$" %generator)
                for l in range(repeat):
                    time_list_generator=[]
                    start_time = time.time()
                    beta_plot,Z_plot= making_Z(generator,beta,tau,N)
                    Z_plot=np.array(Z_plot)
                    time_list_generator.append(time.time() - start_time)
                    F=-np.log(Z_plot[len(Z_plot)-1])/beta_plot[len(Z_plot)-1]
                    Z_list.append(F)
                    n_list.append(l)
                (mu, sigma) = norm.fit(Z_list)
                mu_list.append(mu)
                sigma_list.append(sigma)
                ax.scatter(n_list,Z_list,color=plt.cm.brg(index/len(generator_list)), label="n=%i" %N)
                n, bins, patches = ax1.hist(Z_list,bins=15,alpha=0.5 ,color=plt.cm.brg(index/len(generator_list)), orientation="horizontal", density=True)
                y = scipy.stats.norm.pdf( bins, mu, sigma)
                ax1.plot(y, bins, '--', label=r"$\mu=%.2f, \sigma=%.5f$" %(mu,sigma),linewidth=2,color=plt.cm.brg(index/len(generator_list)))
                ax.set_xlabel("Repetition")
                ax.set_ylabel(r"$F(\beta)$")
                ax1.set_xlabel("$N$")
                ax1.set_ylabel(r"$F(\beta)$")
                ax.grid()
                ax1.grid()
                ax1.legend()
                index+=1
            mu_list=np.array(mu_list)
            sigma_list=np.array(sigma_list)
            ax2.plot(generator_list,mu_list, color=plt.cm.brg(jndex/len(N_list)), label=r"$N=%i$" %(N))
            ax2.fill_between(generator_list,mu_list-sigma_list,mu_list+sigma_list,color=plt.cm.brg(jndex/len(N_list)), alpha=0.4)
            ax2.set_ylabel(r"$F(\beta)$")
            ax2.set_xlabel(r"$N_\psi$")
            jndex+=1
            mu_together.append(sum(mu_list)/len(mu_list))
            sigma_together.append(sum(sigma_list)/len(sigma_list))
            plt.show()
        ax2.grid()
        ax2.legend()
        mu_together=np.array(mu_together)
        sigma_together=np.array(sigma_together)
        ax3.plot(N_list,mu_together, color=plt.cm.brg(jndex/len(N_list)), label=r"$N=%i$" %(N))
        ax3.fill_between(N_list,mu_together-sigma_together,mu_together+sigma_together,color=plt.cm.brg(jndex/len(N_list)), alpha=0.4)
        ax3.set_ylabel(r"$F(\beta)$")
        ax3.set_xlabel(r"$N$")
        ax3.grid()
        ax3.legend()
        plt.close()


        generator_list=[2,5,10,20,40]
        repeat=40
        import time
        time_list_generator=[]
        time_list_n=[]
        index=0
        mu_list=[]
        sigma_list=[]
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        N_list=[2,3,4,5,6,7,8,9]
        tau=10**(-2)
        generator=40
        beta=2
        jndex=0
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        fig2.suptitle(r"Prikaz spreminjanja z generatorjem $\langle H \rangle_{2}$ pri $\tau=%.3f$" %(tau))
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)
        fig3.suptitle(r"Prikaz povprečnih $\langle H \rangle_{2}$ pri $\tau=%.3f$" %(tau))
        mu_together=[]
        sigma_together=[]
        for N in tqdm(N_list):
            #fig = plt.figure()
            #fig.suptitle(r"Prikaz razporeditve $\langle H \rangle_\beta$ pri $\tau=%.3f$ pri $n=%i$" %(tau,N))
            index=0
            mu_list=[]
            sigma_list=[]
            for generator in tqdm(generator_list):
                Z_list=[]
                n_list=[]
                #ax = fig.add_subplot(5,2,2*index+1)
                #ax1 = fig.add_subplot(5,2,2*index+2)
                #ax.set_title(r"$n_\psi=%i$" %generator)
                #ax1.set_title(r"$n_\psi=%i$" %generator)
                for l in range(repeat):
                    time_list_generator=[]
                    start_time = time.time()
                    beta_plot,Z_plot, H_plot= making_H(generator,beta,tau,N)
                    Z_plot=np.array(Z_plot)
                    time_list_generator.append(time.time() - start_time)
                    F=-np.log(Z_plot[len(Z_plot)-1])/beta_plot[len(Z_plot)-1]
                    Z_list.append(H_plot[len(Z_plot)-1])
                    n_list.append(l)
                (mu, sigma) = norm.fit(Z_list)
                mu_list.append(mu)
                sigma_list.append(sigma)
                #ax.scatter(n_list,Z_list,color=plt.cm.brg(index/len(generator_list)), label="n=%i" %N)
                #n, bins, patches = ax1.hist(Z_list,bins=15,alpha=0.5 ,color=plt.cm.brg(index/len(generator_list)), orientation="horizontal", density=True)
                #y = scipy.stats.norm.pdf( bins, mu, sigma)
                #ax1.plot(y, bins, '--', label=r"$\mu=%.2f, \sigma=%.5f$" %(mu,sigma),linewidth=2,color=plt.cm.brg(index/len(generator_list)))
                #ax.set_xlabel("Repetition")
                #ax.set_ylabel(r"$\langle H \rangle_\beta$")
                #ax1.set_xlabel("$N$")
                #ax1.set_ylabel(r"$\langle H \rangle_\beta$")
                #ax.grid()
                #ax1.grid()
                #ax1.legend()
                index+=1
            mu_list=np.array(mu_list)
            sigma_list=np.array(sigma_list)
            ax2.plot(generator_list,mu_list, color=plt.cm.brg(jndex/len(N_list)), label=r"$N=%i$" %(N))
            ax2.fill_between(generator_list,mu_list-sigma_list,mu_list+sigma_list,color=plt.cm.brg(jndex/len(N_list)), alpha=0.4)
            ax2.set_ylabel(r"$\langle H \rangle_\beta$")
            ax2.set_xlabel(r"$N_\psi$")
            jndex+=1
            mu_together.append(sum(mu_list)/len(mu_list))
            sigma_together.append(sum(sigma_list)/len(sigma_list))
        ax2.grid()
        ax2.legend()
        mu_together=np.array(mu_together)
        sigma_together=np.array(sigma_together)
        ax3.plot(N_list,mu_together, color=plt.cm.brg(jndex/len(N_list)), label=r"$N=%i$" %(N))
        ax3.fill_between(N_list,mu_together-sigma_together,mu_together+sigma_together,color=plt.cm.brg(jndex/len(N_list)), alpha=0.4)
        ax3.set_ylabel(r"$\langle H \rangle_\beta$")
        ax3.set_xlabel(r"$N$")
        ax3.grid()
        ax3.legend()
        plt.show()
        plt.close()

    
