import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.core.numeric import identity
import scipy
from scipy import special
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
from tqdm import tqdm
from numba import njit
from scipy.stats import norm

@njit
def derivate_V(k, lam, q):
    der_V=np.zeros(len(q))
    for j in range(1,len(q)-1):
        if j==k:
            der_V[j]=2*q[j]+4*q[j]**3*lam-q[j+1]
        if j+1==k:
            der_V[j+1]=q[j+1]-q[j]
    return der_V
@njit
def derivative_V(q, lam):
    dev_V=np.zeros(len(q))
    for j in range(1,len(q)-1):
        dev_V[j]=3*q[j]+4*q[j]**3*lam-q[j+1]-q[j-1]
    return dev_V


def derivates(t,state,lam, tau, T_l,T_r,n):
    derivates=np.zeros(len(state))
    #dq/dt=p
    derivates[:n]=state[n:2*n]


    #dp/dt= ...
    derivates[n]=-2*state[0]+state[1]-4*lam*state[0]**3-state[n]*state[2*n] #j=0
    derivates[n+1:2*n-1]=state[2:n]+state[:n-2]-3*state[1:n-1]-4*lam*state[1:n-1]**3
    derivates[2*n-1]=-1*(2*state[n-1]-state[n-2]+4*lam*state[n-1]**3)-state[2*n-1]*state[2*n+1]

    #dzeta_l/dt
    derivates[2*n]=1/tau*(state[n]**2-T_l)

    #dzeta_r/dt
    derivates[2*n+1]=1/tau*(state[2*n-1]**2-T_r)
    return np.array(derivates)

def integration_t(x,t):
    value=1/t[len(t)-1]*scipy.integrate.simpson(x,t)
    return value


def solving_eq(t_fin,length, particles, t_n, T_l, T_r,lam):
    a=length/particles
    q0=np.arange(-length/2,length/2,a)
    p0=np.random.randn(particles)
    norm=np.sum(p0**2)
    p0/=np.sqrt(norm)
    zeta_T0=np.random.randn(1)
    zeta_R0=np.random.randn(1)
    tau=t_fin/t_n
    t=np.linspace(0,t_fin,t_n)
    init_state=np.concatenate((q0, p0,zeta_T0,zeta_R0))
    sol=solve_ivp(derivates,(0,t_fin),init_state, method="DOP853",t_eval=np.linspace(0,t_fin,t_n),args=(lam, tau, T_l,T_r,particles,))
    q_matrix=[]
    p_matrix=[]
    zeta_T=[]
    zeta_R=[]
    for i in range(len(sol.y)):
        if i<n:
            q_matrix.append(sol.y[i])
        elif n<i<2*n:
            p_matrix.append(sol.y[i])
        elif i==2*n:
            zeta_T=sol.y[i]
        elif i==2*n+1:
            zeta_R=sol.y[i]
    return q_matrix, p_matrix, zeta_T, zeta_R,t


def initial_conditions(length, particles,T_r,T_l):
    a=length/particles
    sigma_q=0.2
    sigma_p=np.sqrt((T_r+T_l)/2)
    q0=sigma_q*np.zeros(particles)
    p0=sigma_p*np.random.randn(particles)
    zeta_T0=np.random.randn(1)
    zeta_R0=np.random.randn(1)
    init_state=np.concatenate((q0, p0,zeta_T0,zeta_R0))
    return init_state

def solving_eq_same_0(t_fin,init_state, particles, tau, T_l, T_r,lam):
    t_n=int(t_fin/tau)
    t=np.linspace(0,t_fin,t_n)
    sol=solve_ivp(derivates,(0,t_fin),init_state, method="DOP853",t_eval=np.linspace(0,t_fin,t_n),args=(lam, tau, T_l,T_r,particles,))
    n=particles
    q_matrix=[]
    p_matrix=[]
    zeta_T=[]
    zeta_R=[]
    for i in range(len(sol.y)):
        if i<n:
            q_matrix.append(sol.y[i])
        elif n<=i<2*n:
            p_matrix.append(sol.y[i])
        elif i==2*n:
            zeta_T=sol.y[i]
        elif i==2*n+1:
            zeta_R=sol.y[i]
    return q_matrix, p_matrix, zeta_T, zeta_R,t


def cal_T(p_matrix,t, T_r, T_l):
    T=[]
    Veriga=[]
    for j in range(len(p_matrix)):
        Veriga.append(j)
        T.append(integration_t(p_matrix[j]**2,t))
    T=np.array(T)
    return T, Veriga


def cal_J(q,p,t,a):
    J=np.zeros(len(q))
    Veriga=[0]
    for j in range(1,len(J)-1):
        Veriga.append(j)
        J[j]=integration_t(-1/2*p[j]*(q[j+1]-q[j-1]+2*a),t)
    Veriga.append(len(q))
    return J, Veriga

def plot_Tj(T_l,T_r,n,lam):
    if lam!=0:
        pj=np.zeros(n)
        for j in range(n):
            pj[j]=T_l+(j)/(n)*(T_r-T_l)
            label_text=r"$T_L+\frac{j-1}{N-1}(T_r-T_l)$"
    if lam==0:
        pj=np.zeros(n)
        for j in range(n):
            pj[j]=(T_l+T_r)/2
            label_text=r"$\frac{T_l+T_r}{2}$"
    return pj, label_text

def fit_j(x,kappa,C,delta_T):
    return kappa*delta_T/x+C

t_fin=10
n=20
N=100
lam=0
tau=t_fin/N
T_l=1
T_r=2
a=1

T_plot="No"
if T_plot=="Yes":
    lam=1
    t_list=[50,100,1000,5000,10000,25000,50000,75000]
    index=0
    length=1
    init_state=initial_conditions(length, n,T_r,T_l)
    tau=5
    q_matrix0, p_matrix0, zeta_T0, zeta_R0,t0=solving_eq_same_0(5*10**2, init_state, n, tau, T_l, T_r,lam)
    T0, Veriga0=cal_T(p_matrix0,t0, T_r,T_l)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig1.suptitle(r"Prikaz $T$ v odvisnosti od končnega časa pri $\lambda=%.2f$ in $\tau=%.2f$" %(lam, tau))
    fig2.suptitle(r"Prikaz spremembe $T$ v odvisnosti od končnega časa pri $\lambda=%.2f$ in $\tau=%.2f$" %(lam, tau))
    for t_fin in tqdm(t_list):
        q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0(t_fin, init_state, n, tau, T_l, T_r,lam)
        T, Veriga=cal_T(p_matrix,t, T_r,T_l)
        ax1.plot(Veriga,T, label=r"$t_{max}=%i$" %t_fin,color=plt.cm.coolwarm(index/len(t_list)))
        ax1.legend()
        ax2.plot(Veriga,abs(T-T0)/T0, label=r"$t_{max}=%i$" %t_fin,color=plt.cm.coolwarm(index/len(t_list)))
        ax2.legend()
        index+=1
    pj, label_text=plot_Tj(T_l,T_r,n,lam)
    ax1.plot(Veriga,pj,"--", alpha=0.8, label=label_text,color="black")
    ax1.grid()
    ax1.legend()
    ax1.set_xlabel("N")
    ax1.set_ylabel(r"$\langle T \rangle$")
    ax2.set_xlabel("N")
    ax2.set_yscale("log")
    ax2.grid()
    ax2.set_ylabel(r"$\frac{\langle T \rangle-\langle T \rangle_{10^5}}{\langle T \rangle_{10^5}}$", fontsize=16)
    plt.show()



generirane_plot_tau_hist_VEC="No"
if generirane_plot_tau_hist_VEC=="Yes":
    #t_fin_list=[100,200,300]
    tau_list=[0.5,1,2,5,10]
    jndex=0
    lam=1
    t_fin=20000
    fig3 = plt.figure()
    fig4 = plt.figure()
    ax6 = fig4.add_subplot(1,1,1)
    fig5 = plt.figure()
    ax7 = fig5.add_subplot(1,1,1)
    mu_list=[]
    for tau in tqdm(tau_list):
        n=20
        index=0
        length=1
        init_state=initial_conditions(length, n,T_r,T_l)
        fig1, ax1 = plt.subplots()
        avg_T=[]
        T_line=[]
        fig2, ax2 = plt.subplots()
        fig1.suptitle(r"Prikaz $T$ za več generiranih primerov pri $\tau=%.2f$, $\lambda=%.2f$ in $t_{fin}=%.2f$" %(tau, lam, t_fin))
        fig2.suptitle(r"Prikaz $\langle T \rangle$ za več generiranih primerov pri $\tau=%.2f$, $\lambda=%.2f$ in $t_{fin}=%.2f$" %(tau, lam, t_fin))
        fig3.suptitle(r"Prikaz $T$ za več generiranih primerov v odvisnosti od $\tau$ pri $t_{fin}=%.2f$, $\lambda=%.2f$" %(tau, lam))
        ax3 = fig3.add_subplot(int(len(tau_list)),3,1+jndex)
        ax4 = fig3.add_subplot(int(len(tau_list)),3,2+jndex)
        ax5 = fig3.add_subplot(int(len(tau_list)),3,3+jndex)
        n_functions=np.linspace(1,5,20)
        for i in tqdm(n_functions):
            init_state=initial_conditions(length, n,T_r,T_l)
            q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0(t_fin, init_state, n, tau, T_l, T_r,lam)
            T, Veriga=cal_T(p_matrix,t, T_r,T_l)
            ax1.plot(Veriga,T,color=plt.cm.coolwarm(index/len(n_functions)))
            ax3.plot(Veriga,T,color=plt.cm.coolwarm(index/len(n_functions)))
            index+=1
            avg_T.append(sum(T/len(T)))
            T_line.append(T)
            pj, label_text=plot_Tj(T_l,T_r,n,lam)
            ax3.plot(Veriga,pj,"--", alpha=0.8,color="black")
        T_line=np.array(T_line)
        T_avg=[]
        for i in range(len(T_line.T)):
            T_avg.append(sum(T_line.T[i]/len(T_line.T[i])))
        
        ax6.plot(Veriga,T_avg, label=r"$\tau=%.2f$" %tau, color=plt.cm.coolwarm(jndex/(3*len(tau_list))))
        ax6.set_title(r"Prikaz povprečnih vrednosti $\langle T \rangle$ pri $\lambda=%.2f$, $t_{fin}=%.2f$" %(lam, t_fin))
        ax6.set_xlabel("N")
        ax6.set_ylabel(r"$\langle T \rangle$")
        ax6.grid()
        ax6.legend()
        ax2.plot(n_functions,avg_T)
        ax1.grid()
        ax1.set_xlabel("N")
        ax1.set_ylabel(r"$\langle T \rangle$")
        ax2.grid()
        ax2.set_xlabel(r"$N$-ti generator")
        ax2.set_ylabel(r"$\langle T \rangle_{avg}$")
        ax3.set_title(r"Odvisnost od delca n, $\tau=%.2f$" %tau)
        ax4.set_title(r"Povprečna temperatura kopeli od $N$, $\tau=%.2f$" %tau)
        ax4.scatter(n_functions,avg_T)
        ax5.set_title(r"Porazdelitev temperature od $N$, $\tau=%.2f$" %tau)
        #ax5.scatter(n_functions,avg_T)
        n_fir, bins, patches = ax5.hist(avg_T,bins=5,alpha=0.5 ,color="red", orientation="horizontal", density=True)
        (mu, sigma) = norm.fit(avg_T)
        mu_list.append(mu)
        y = scipy.stats.norm.pdf( bins, mu, sigma)
        ax5.plot(y, bins, '--', label=r"$\mu=%.2f, \sigma=%.3f$" %(mu,sigma),linewidth=2,color="red")
        ax5.grid()
        ax5.set_xlabel(r"$N$-ti generator")
        ax5.set_ylabel(r"$\langle T \rangle_{avg}$")
        ax3.grid()
        ax3.set_xlabel("N")
        ax3.set_ylabel(r"$\langle T \rangle$")
        ax4.grid()
        ax4.set_xlabel(r"$N$-ti generator")
        ax4.set_ylabel(r"$\langle T \rangle_{avg}$")
        ax5.legend()
        jndex+=3
    pj, label_text=plot_Tj(T_l,T_r,n,lam)
    ax6.plot(Veriga,pj,"--", alpha=0.8,color="black")
    ax7.set_title(r"Prikaz konvergence temperature od končnega časa pri $\tau=%.2f$, $\lambda=%.2f$" %(tau, lam))
    ax7.plot(tau_list,mu_list)
    ax7.set_xlabel(r"$\lambda$")
    ax7.set_ylabel(r"$\mu$")
    ax7.grid()
    plt.show()


generirane_plot_hist_VEC="nO"
if generirane_plot_hist_VEC=="Yes":
    #t_fin_list=[10,50,100,200,500,1000,2000,5000,10000,15000,20000]
    t_fin_list=[50,100,1000,5000,10000,25000]
    jndex=0
    lam=1
    t_fin=20000
    fig3 = plt.figure()
    fig4 = plt.figure()
    ax6 = fig4.add_subplot(1,1,1)
    fig5 = plt.figure()
    ax7 = fig5.add_subplot(1,1,1)
    mu_list=[]
    for t_fin in tqdm(t_fin_list):
        n=20
        index=0
        length=1
        init_state=initial_conditions(length, n,T_r,T_l)
        fig1, ax1 = plt.subplots()
        avg_T=[]
        tau=1
        T_line=[]
        fig2, ax2 = plt.subplots()
        fig1.suptitle(r"Prikaz $T$ za več generiranih primerov pri $\tau=%.2f$, $\lambda=%.2f$ in $t_{fin}=%.2f$" %(tau, lam, t_fin))
        fig2.suptitle(r"Prikaz $\langle T \rangle$ za več generiranih primerov pri $\tau=%.2f$, $\lambda=%.2f$ in $t_{fin}=%.2f$" %(tau, lam, t_fin))
        fig3.suptitle(r"Prikaz $T$ za več generiranih primerov v odvisnosti od $t_{fin}=%.2f$ pri $\tau$, $\lambda=%.2f$" %(tau, lam))
        ax3 = fig3.add_subplot(int(len(t_fin_list)),3,1+jndex)
        ax4 = fig3.add_subplot(int(len(t_fin_list)),3,2+jndex)
        ax5 = fig3.add_subplot(int(len(t_fin_list)),3,3+jndex)
        n_functions=np.linspace(1,5,20)
        for i in tqdm(n_functions):
            init_state=initial_conditions(length, n,T_r,T_l)
            q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0(t_fin, init_state, n, tau, T_l, T_r,lam)
            T, Veriga=cal_T(p_matrix,t, T_r,T_l)
            ax1.plot(Veriga,T,color=plt.cm.coolwarm(index/len(n_functions)))
            ax3.plot(Veriga,T,color=plt.cm.coolwarm(index/len(n_functions)))
            index+=1
            avg_T.append(sum(T/len(T)))
            T_line.append(T)
            pj, label_text=plot_Tj(T_l,T_r,n,lam)
            ax3.plot(Veriga,pj,"--", alpha=0.8,color="black")
        T_line=np.array(T_line)
        T_avg=[]
        for i in range(len(T_line.T)):
            T_avg.append(sum(T_line.T[i]/len(T_line.T[i])))
        
        ax6.plot(Veriga,T_avg, label=r"$t_{fin}=%.2f$" %t_fin, color=plt.cm.coolwarm(jndex/(3*len(t_fin_list))))
        ax6.set_title(r"Prikaz povprečnih vrednosti $\langle T \rangle$ pri $\lambda=%.2f$, $tau=%.2f$" %(lam, tau))
        ax6.set_xlabel("N")
        ax6.set_ylabel(r"$\langle T \rangle$")
        ax6.grid()
        ax6.legend()
        ax2.plot(n_functions,avg_T)
        ax1.grid()
        ax1.set_xlabel("N")
        ax1.set_ylabel(r"$\langle T \rangle$")
        ax2.grid()
        ax2.set_xlabel(r"$N$-ti generator")
        ax2.set_ylabel(r"$\langle T \rangle_{avg}$")
        ax3.set_title(r"Odvisnost od delca n, $t_{fin}=%.2f$" %t_fin)
        ax4.set_title(r"Povprečna temperatura kopeli od $N$, $t_{fin}=%.2f$" %t_fin)
        ax4.scatter(n_functions,avg_T)
        ax5.set_title(r"Porazdelitev temperature od $N$, $t_{fin}=%.2f$" %t_fin)
        #ax5.scatter(n_functions,avg_T)
        n_fir, bins, patches = ax5.hist(avg_T,bins=5,alpha=0.5 ,color="red", orientation="horizontal", density=True)
        (mu, sigma) = norm.fit(avg_T)
        mu_list.append(mu)
        y = scipy.stats.norm.pdf( bins, mu, sigma)
        ax5.plot(y, bins, '--', label=r"$\mu=%.2f, \sigma=%.3f$" %(mu,sigma),linewidth=2,color="red")
        ax5.grid()
        ax5.set_xlabel(r"$N$-ti generator")
        ax5.set_ylabel(r"$\langle T \rangle_{avg}$")
        ax3.grid()
        ax3.set_xlabel("N")
        ax3.set_ylabel(r"$\langle T \rangle$")
        ax4.grid()
        ax4.set_xlabel(r"$N$-ti generator")
        ax4.set_ylabel(r"$\langle T \rangle_{avg}$")
        ax5.legend()
        jndex+=3
    pj, label_text=plot_Tj(T_l,T_r,n,lam)
    ax6.plot(Veriga,pj,"--", alpha=0.8,color="black")
    ax7.set_title(r"Prikaz konvergence temperature od končnega časa pri $\tau=%.2f$, $\lambda=%.2f$" %(tau, lam))
    ax7.plot(t_fin_list,mu_list)
    ax7.set_xlabel(r"$\lambda$")
    ax7.set_ylabel(r"$\mu$")
    ax7.grid()
    plt.show()


generirane_plot_hist_VEC="No"
if generirane_plot_hist_VEC=="Yes":
    #t_fin_list=[100,200,300]
    lam_list=[0,0.25,0.5,0.75,1]
    jndex=0
    t_fin=20000
    fig3 = plt.figure()
    fig4 = plt.figure()
    ax6 = fig4.add_subplot(1,1,1)
    fig5 = plt.figure()
    ax7 = fig5.add_subplot(1,1,1)
    mu_list=[]
    for lam in tqdm(lam_list):
        n=20
        index=0
        length=1
        init_state=initial_conditions(length, n,T_r,T_l)
        tau=5
        fig1, ax1 = plt.subplots()
        avg_T=[]
        T_line=[]
        fig2, ax2 = plt.subplots()
        fig1.suptitle(r"Prikaz $T$ za več generiranih primerov pri $\tau=%.2f$, $\lambda=%.2f$ in $t_{fin}=%.2f$" %(tau, lam, t_fin))
        fig2.suptitle(r"Prikaz $\langle T \rangle$ za več generiranih primerov pri $\tau=%.2f$, $\lambda=%.2f$ in $t_{fin}=%.2f$" %(tau, lam, t_fin))
        fig3.suptitle(r"Prikaz $T$ za več generiranih primerov v odvisnosti od $\lambda$ pri $\tau=%.2f$, $z_{fin}=%.2f$" %(tau, t_fin))
        ax3 = fig3.add_subplot(int(len(lam_list)),3,1+jndex)
        ax4 = fig3.add_subplot(int(len(lam_list)),3,2+jndex)
        ax5 = fig3.add_subplot(int(len(lam_list)),3,3+jndex)
        n_functions=np.linspace(1,5,20)
        for i in tqdm(n_functions):
            init_state=initial_conditions(length, n,T_r,T_l)
            q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0(t_fin, init_state, n, tau, T_l, T_r,lam)
            T, Veriga=cal_T(p_matrix,t, T_r,T_l)
            ax1.plot(Veriga,T,color=plt.cm.coolwarm(index/len(n_functions)))
            ax3.plot(Veriga,T,color=plt.cm.coolwarm(index/len(n_functions)))
            index+=1
            avg_T.append(sum(T/len(T)))
            T_line.append(T)
            pj, label_text=plot_Tj(T_l,T_r,n,lam)
            ax3.plot(Veriga,pj,"--", alpha=0.8,color="black")
        T_line=np.array(T_line)
        T_avg=[]
        for i in range(len(T_line.T)):
            T_avg.append(sum(T_line.T[i]/len(T_line.T[i])))
        
        arr4=np.ones(len(n_functions))*(T_l+T_r)/2
        ax6.plot(Veriga,T_avg, label=r"$\lambda=%.2f$" %lam, color=plt.cm.coolwarm(jndex/(3*len(lam_list))))
        ax6.set_title(r"Prikaz povprečnih vrednosti $\langle T \rangle$ pri $\tau=%.2f$, $t_{fin}=%.2f$" %(tau, t_fin))
        ax6.set_xlabel("N")
        ax6.set_ylabel(r"$\langle T \rangle$")
        ax6.grid()
        ax6.legend()
        ax2.plot(n_functions,avg_T)
        ax1.grid()
        ax1.set_xlabel("N")
        ax1.set_ylabel(r"$\langle T \rangle$")
        ax2.grid()
        ax2.set_xlabel(r"$N$-ti generator")
        ax2.set_ylabel(r"$\langle T \rangle_{avg}$")
        ax3.set_title(r"Odvisnost od delca n, $\lambda=%.2f$" %lam)
        ax4.set_title(r"Povprečna temperatura kopeli od $N$, $\lambda=%.2f$" %lam)
        ax4.scatter(n_functions,avg_T)
        ax5.set_title(r"Porazdelitev temperature od $N$, $\lambda=%.2f$" %lam)
        #ax5.scatter(n_functions,avg_T)
        n_fit, bins, patches = ax5.hist(avg_T,bins=5,alpha=0.5 ,color="red", orientation="horizontal", density=True)
        (mu, sigma) = norm.fit(avg_T)
        mu_list.append(mu)
        y = scipy.stats.norm.pdf( bins, mu, sigma)
        ax5.plot(y, bins, '--', label=r"$\mu=%.2f, \sigma=%.3f$" %(mu,sigma),linewidth=2,color="red")
        ax5.grid()
        ax5.set_xlabel(r"$N$-ti generator")
        ax5.set_ylabel(r"$\langle T \rangle_{avg}$")
        ax3.grid()
        ax3.set_xlabel("N")
        ax3.set_ylabel(r"$\langle T \rangle$")
        ax4.grid()
        ax4.set_xlabel(r"$N$-ti generator")
        ax4.set_ylabel(r"$\langle T \rangle_{avg}$")
        ax5.legend()
        jndex+=3
    pj, label_text=plot_Tj(T_l,T_r,n,lam)
    ax6.plot(Veriga,pj,"--", alpha=0.8,color="black")
    ax7.set_title(r"Prikaz konvergence temperature od končnega časa pri $\tau=%.2f$, $\lambda=%.2f$" %(tau, lam))
    ax7.plot(lam_list,mu_list)
    ax7.set_xlabel(r"$\lambda$")
    ax7.set_ylabel(r"$\mu$")
    ax7.grid()
    plt.show()

J_plot="No"
if J_plot=="Yes":
    t_list=[100,1000,5000,10000,30000,40000]
    index=0
    length=1
    init_state=initial_conditions(length, n,T_r,T_l)
    tau=1
    q_matrix0, p_matrix0, zeta_T0, zeta_R0,t0=solving_eq_same_0(5*10**4, init_state, n, tau, T_l, T_r,lam)
    q=q_matrix0
    p=p_matrix0
    a=0
    lam=1
    J0, Veriga0=cal_J(q,p,t0,a)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig1.suptitle(r"Prikaz toka $\langle J_j \rangle$ v odvisnosti od končnega časa pri $\lambda=%.2f$ in $\tau=%.2f$" %(lam, tau))
    fig2.suptitle(r"Prikaz spremembe $\langle J_j \rangle$ v odvisnosti od končnega časa pri $\lambda=%.2f$ in $\tau=%.2f$" %(lam, tau))
    for t_fin in tqdm(t_list):
        q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0(t_fin, init_state, n, tau, T_l, T_r,lam)
        q_matrix=np.array(q_matrix)
        p_matrix=np.array(p_matrix)
        q=q_matrix
        p=p_matrix
        J, Veriga=cal_J(q,p,t, a)
        ax1.plot(Veriga,J, label=r"$t_{max}=%i$" %t_fin,color=plt.cm.coolwarm(index/len(t_list)))
        ax2.plot(Veriga,(J-J0)/J0, label=r"$t_{max}=%i$" %t_fin,color=plt.cm.coolwarm(index/len(t_list)))
        ax1.legend()
        ax2.legend()
        index+=1
    ax1.grid()
    ax1.set_xlabel("N")
    ax1.set_ylabel(r"$\langle J_j \rangle$")
    ax2.set_xlabel("N")
    ax2.set_yscale("log")
    ax2.grid()
    ax2.set_ylabel(r"$\frac{\langle J_j \rangle-\langle J_j \rangle_{10^6}}{\langle J_j \rangle_{10^6}}$")
    plt.show()

generirane_plot_hist_J_VEC="No"
if generirane_plot_hist_J_VEC=="Yes":
    lam_list=[0,0.25,0.5,0.75,1]
    jndex=0
    t_fin=20000
    fig3 = plt.figure()
    fig4 = plt.figure()
    ax6 = fig4.add_subplot(1,1,1)
    fig5 = plt.figure()
    ax7 = fig5.add_subplot(1,1,1)
    mu_list=[]
    a=0
    for lam in tqdm(lam_list):
        n=10
        index=0
        length=1
        init_state=initial_conditions(length, n,T_r,T_l)
        tau=1
        fig1, ax1 = plt.subplots()
        avg_T=[]
        T_line=[]
        fig2, ax2 = plt.subplots()
        fig1.suptitle(r"Prikaz $J$ za več generiranih primerov pri $\tau=%.2f$, $\lambda=%.2f$ in $t_{fin}=%.2f$" %(tau, lam, t_fin))
        fig2.suptitle(r"Prikaz $\langle J \rangle$ za več generiranih primerov pri $\tau=%.2f$, $\lambda=%.2f$ in $t_{fin}=%.2f$" %(tau, lam, t_fin))
        fig3.suptitle(r"Prikaz $J$ za več generiranih primerov v odvisnosti od $\lambda$ pri $\tau=%.2f$, $T_{fin}=%.2f$" %(tau, t_fin))
        ax3 = fig3.add_subplot(int(len(lam_list)),3,1+jndex)
        ax4 = fig3.add_subplot(int(len(lam_list)),3,2+jndex)
        ax5 = fig3.add_subplot(int(len(lam_list)),3,3+jndex)
        n_functions=np.linspace(1,5,100)
        for i in tqdm(n_functions):
            init_state=initial_conditions(length, n,T_r,T_l)
            q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0(t_fin, init_state, n, tau, T_l, T_r,lam)
            q=q_matrix
            p=p_matrix
            T, Veriga=cal_J(q,p,t,a)
            ax1.plot(Veriga,T,color=plt.cm.coolwarm(index/len(n_functions)))
            ax3.plot(Veriga,T,color=plt.cm.coolwarm(index/len(n_functions)))
            index+=1
            avg_T.append(sum(T/len(T)))
            T_line.append(T)
        #arr4=np.ones(len(Veriga))*(T_l+T_r)/2
        #ax3.plot(Veriga,arr4, "--", color="black", alpha=0.7)
        T_line=np.array(T_line)
        T_avg=[]
        for i in range(len(T_line.T)):
            T_avg.append(sum(T_line.T[i]/len(T_line.T[i])))
        
        #arr4=np.ones(len(n_functions))*(T_l+T_r)/2
        ax6.plot(Veriga,T_avg, label=r"$\lambda=%i$" %lam, color=plt.cm.coolwarm(jndex/(3*len(lam_list))))
        ax6.set_title(r"Prikaz povprečnih vrednosti $\langle J \rangle$ pri $\tau=%.2f$, $t_{fin}=%.2f$" %(tau, t_fin))
        ax6.set_xlabel("N")
        ax6.set_ylabel(r"$\langle J \rangle$")
        ax6.grid()
        ax6.legend()
        ax2.plot(n_functions,avg_T)
        ax1.grid()
        ax1.set_xlabel("N")
        ax1.set_ylabel(r"$\langle J \rangle$")
        ax2.grid()
        ax2.set_xlabel(r"$N$-ti generator")
        ax2.set_ylabel(r"$\langle J \rangle_{avg}$")
        ax3.set_title(r"Odvisnost od delca n, $\lambda=%.2f$" %lam)
        ax4.set_title(r"Povprečen toplotni tok kopeli od $N$, $\lambda=%.2f$" %lam)
        ax4.scatter(n_functions,avg_T)
        ax5.set_title(r"Porazdelitev toplotnega toka od $N$, $\lambda=%.2f$" %lam)
        #ax5.scatter(n_functions,avg_T)
        n, bins, patches = ax5.hist(avg_T,bins=15,alpha=0.5 ,color="red", orientation="horizontal", density=True)
        (mu, sigma) = norm.fit(avg_T)
        mu_list.append(mu)
        y = scipy.stats.norm.pdf( bins, mu, sigma)
        ax5.plot(y, bins, '--', label=r"$\mu=%.2f, \sigma=%.3f$" %(mu,sigma),linewidth=2,color="red")
        ax5.grid()
        ax5.set_xlabel(r"$N$-ti generator")
        ax5.set_ylabel(r"$\langle J \rangle_{avg}$")
        ax3.grid()
        ax3.set_xlabel("N")
        ax3.set_ylabel(r"$\langle J \rangle$")
        ax4.grid()
        ax4.set_xlabel(r"$N$-ti generator")
        ax4.set_ylabel(r"$\langle J \rangle_{avg}$")
        ax5.legend()
        jndex+=3
    plt.close()
    ax7.set_title(r"Prikaz konvergence toplotnega toka od končnega časa pri $\tau=%.2f$, $\lambda=%.2f$" %(tau, lam))
    ax7.plot(lam_list,mu_list)
    ax7.set_xlabel(r"$t_{fin}$")
    ax7.set_ylabel(r"$\mu$")
    ax7.grid()
    plt.show()

lambda_J_VEC="No"
if lambda_J_VEC=="Yes":
    lam_list=[0,0.25,0.5,0.75,1]
    jndex=0
    t_fin=100000
    fig4 = plt.figure()
    ax6 = fig4.add_subplot(1,1,1)
    mu_list=[]
    a=0
    n=20
    for lam in tqdm(lam_list):
        index=0
        length=1
        init_state=initial_conditions(length, n,T_r,T_l)
        tau=1
        avg_T=[]
        T_line=[]
        n_functions=np.linspace(1,5,3)
        for i in tqdm(n_functions):
            init_state=initial_conditions(length, n,T_r,T_l)
            q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0(t_fin, init_state, n, tau, T_l, T_r,lam)
            q=q_matrix
            p=p_matrix
            T, Veriga=cal_J(q,p,t,a)
            index+=1
            avg_T.append(sum(T/len(T)))
            T_line.append(T)
        T_line=np.array(T_line)
        T_avg=[]
        for i in range(len(T_line.T)):
            T_avg.append(sum(T_line.T[i]/len(T_line.T[i])))
        ax6.plot(Veriga,T_avg, label=r"$\lambda=%.2f$" %lam, color=plt.cm.coolwarm(jndex/(3*len(lam_list))))
        ax6.set_title(r"Prikaz povprečnih vrednosti $\langle J \rangle$ pri $\tau=%.2f$, $t_{fin}=%.2f$" %(tau, t_fin))
        ax6.set_xlabel("N")
        ax6.set_ylabel(r"$\langle J \rangle$")
        ax6.grid()
        ax6.legend()
        (mu, sigma) = norm.fit(avg_T)
        mu_list.append(mu)
        jndex+=3
    plt.show()

fit_J_VEC="Yes"
if fit_J_VEC=="Yes":
    lam_list=[5,10,15,20,25,30,35,40]
    jndex=0
    t_fin=50000
    fig4 = plt.figure()
    ax6 = fig4.add_subplot(1,1,1)
    fig5 = plt.figure()
    ax7 = fig5.add_subplot(1,1,1)
    mu_list=[]
    a=0
    for n in tqdm(lam_list):
        lam=1
        index=0
        length=1
        init_state=initial_conditions(length, n,T_r,T_l)
        tau=1
        avg_T=[]
        T_line=[]
        n_functions=np.linspace(1,5,10)
        for i in tqdm(n_functions):
            init_state=initial_conditions(length, n,T_r,T_l)
            q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0(t_fin, init_state, n, tau, T_l, T_r,lam)
            q=q_matrix
            p=p_matrix
            T, Veriga=cal_J(q,p,t,a)
            index+=1
            avg_T.append(sum(T)//len(T))
            T_line.append(T)
        T_line=np.array(T_line)
        T_avg=[]
        for i in range(len(T_line.T)):
            T_avg.append(sum(T_line.T[i])/len(T_line.T[i]))
        ax6.plot(Veriga,T_avg, label=r"$n=%i$" %n, color=plt.cm.coolwarm(jndex/(3*len(lam_list))))
        ax6.set_title(r"Prikaz povprečnih vrednosti $\langle J \rangle$ pri $\tau=%.2f$, $t_{fin}=%.2f$ in $\lambda=%.1f$" %(tau, t_fin,lam))
        ax6.set_xlabel("N")
        ax6.set_ylabel(r"$\langle J \rangle$")
        ax6.grid()
        ax6.legend()
        (mu, sigma) = norm.fit(T_avg)
        mu_list.append(mu)
        jndex+=3
    popt, pcov =scipy.optimize.curve_fit(lambda x, C,kappa: fit_j(x,kappa,C,T_r-T_l), lam_list, mu_list)
    ax7.set_title(r"Prikaz odvisnosti toplotnega toka od velikosti verige pri $\tau=%.2f$, $\lambda=%.2f$" %(tau, lam))
    ax7.plot(lam_list,-np.array(mu_list), label="Model")
    ax7.plot(lam_list,-np.array(fit_j(lam_list,popt[1],popt[0],T_r-T_l)),label=r"$Fit: \kappa=%.4f$" %(-popt[1]))
    ax7.set_xlabel(r"$n$")
    ax7.set_ylabel(r"$J$")
    ax7.grid()
    #ax7.set_yscale(r"sylog")
    ax7.legend()
    plt.show()