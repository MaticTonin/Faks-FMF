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
import random
from scipy.stats import norm

from func import solving_eq_same_0
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
#q,p,random_l, random_R

def derivates(t,state,lam, tau, T_l,T_r,n):
    derivates=np.zeros(len(state))
    #dq/dt=p
    derivates[:n]=state[n:2*n]

    #dp/dt= ...
    derivates[n]=-1*(2*state[0]-state[1]+4*lam*state[0]**3)-state[n]*state[2*n] 
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

def solving_eq_same_0_Maxwell(t_fin, step_tau,init_state, particles, tau, T_l, T_r,lam):
    t_n=int(t_fin/step_tau)
    solution=[]
    t=[]
    n=particles
    time=0
    for i in range(t_n):
        time+=step_tau
        sol=solve_ivp(derivates,(0,step_tau),init_state, method="DOP853",args=(lam, tau, T_l,T_r,particles,))
        init_state=np.array(sol.y.T[len(sol.y.T)-1])
        t.append(time)
        solution.append(init_state)
        mu=0
        sigma_p=np.sqrt((T_l))
        init_state[n]=random.gauss(mu, sigma_p)
        sigma_p=np.sqrt((T_r))
        init_state[2*n]=random.gauss(mu, sigma_p)
    solution=np.array(solution)
    q_matrix=solution.T[:n]
    p_matrix=solution.T[n:2*n]
    zeta_T=solution.T[2*n]
    zeta_R=solution.T[2*n+1]
    return q_matrix, p_matrix, zeta_T, zeta_R,t

def fit_j(x,kappa,C,delta_T):
    return kappa*delta_T/x+C

def cal_T(p_matrix,t, T_r, T_l):
    T=[T_l]
    Veriga=[0]
    for j in range(1,len(p_matrix)-1):
        Veriga.append(j)
        T.append(integration_t(p_matrix[j]**2,t))
    Veriga.append(len(p_matrix))
    T.append(T_r)
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
t_fin=10
a=0.025
q0=np.arange(-0.5,0.5,a)
n=len(q0)
p0=np.random.randn(len(q0))
zeta_T0=np.random.randn(1)
zeta_R0=np.random.randn(1)
N=100
lam=0
tau=t_fin/N
T_l=1
T_r=2
t=np.linspace(0,t_fin,N)
init_state=np.concatenate((q0, p0,zeta_T0,zeta_R0))
sol=solve_ivp(derivates,(0,t_fin),init_state, method="DOP853",t_eval=np.linspace(0,t_fin,N),args=(lam, tau, T_l,T_r,n,))
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
T=[]
Veriga=[]
for j in range(len(p_matrix)):
    Veriga.append(j)
    T.append(integration_t(p_matrix[j]**2,t))
plt.plot(Veriga,T)
plt.show()

J_plot="No"
if J_plot=="Yes":
    t_list=[100,1000,10000,30000,40000]
    index=0
    length=1
    n=10
    init_state=initial_conditions(length, n,T_r,T_l)
    tau=1
    step_tau=0.5
    q_matrix0, p_matrix0, zeta_T0, zeta_R0,t0=solving_eq_same_0_Maxwell(t_fin, step_tau,init_state, n, tau, T_l, T_r,lam)
    q=q_matrix0
    p=p_matrix0
    a=0
    J0, Veriga0=cal_J(q,p,t0,a)
    lam=1
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,2,1)
    ax2 = fig1.add_subplot(1,2,2)
    fig1.suptitle(r"Prikaz toka $\langle J_j \rangle$ v odvisnosti od končnega časa pri $\lambda=%.2f$ in $\tau=%.2f$" %(lam, tau))
    for t_fin in tqdm(t_list):
        q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0_Maxwell(t_fin, step_tau,init_state, n, tau, T_l, T_r,lam)
        q_matrix=np.array(q_matrix)
        p_matrix=np.array(p_matrix)
        q=q_matrix
        p=p_matrix
        J, Veriga=cal_J(q,p,t, a)
        ax2.set_title(r"Maxwell")
        ax2.plot(Veriga,J, label=r"$t_{max}=%i$" %t_fin,color=plt.cm.coolwarm(index/len(t_list)))
        q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0(t_fin, init_state, n, tau, T_l, T_r,lam)
        q_matrix=np.array(q_matrix)
        p_matrix=np.array(p_matrix)
        q=q_matrix
        p=p_matrix
        J, Veriga=cal_J(q,p,t, a)
        ax1.set_title(r"Nose-Hoover")
        ax1.plot(Veriga,J, label=r"$t_{max}=%i$" %t_fin,color=plt.cm.coolwarm(index/len(t_list)))
        ax1.legend()
        ax2.legend()
        index+=1
    ax1.grid()
    ax1.set_xlabel("N")
    ax1.set_ylabel(r"$\langle J_j \rangle$")
    ax2.set_xlabel("N")
    ax2.grid()
    ax2.set_ylabel(r"$\langle J_j \rangle$")
    plt.show()

T_plot="No"
if T_plot=="Yes":
    t_list=[100,1000,10000,30000,40000]
    index=0
    length=1
    tau=1
    n=10
    step_tau=0.5
    lam=0
    q_matrix0, p_matrix0, zeta_T0, zeta_R0,t0=solving_eq_same_0_Maxwell(t_fin, step_tau,init_state, n, tau, T_l, T_r,lam)
    T0, Veriga0=cal_T(p_matrix0,t0, T_r,T_l)
    init_state=initial_conditions(length, n,T_r,T_l)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,2,1)
    ax2 = fig1.add_subplot(1,2,2)
    fig1.suptitle(r"Prikaz $T$ v odvisnosti od končnega časa pri $\lambda=%.2f$ in $\tau=%.2f$" %(lam, tau))
    for t_fin in tqdm(t_list):
        q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0_Maxwell(t_fin, step_tau,init_state, n, tau, T_l, T_r,lam)
        T, Veriga=cal_T(p_matrix,t, T_r,T_l)
        ax2.set_title(r"Maxwell")
        ax2.plot(Veriga,T, label=r"$t_{max}=%i$" %t_fin,color=plt.cm.coolwarm(index/len(t_list)))
        ax2.legend()
        q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0(t_fin, init_state, n, tau, T_l, T_r,lam)
        T, Veriga=cal_T(p_matrix,t, T_r,T_l)
        ax1.set_title(r"Nose-Hoover")
        ax1.plot(Veriga,T, label=r"$t_{max}=%i$" %t_fin,color=plt.cm.coolwarm(index/len(t_list)))
        ax1.legend()
        index+=1
    pj, label_text=plot_Tj(T_l,T_r,n,lam)
    ax1.plot(Veriga,pj,"--", alpha=0.8, label=label_text,color="black")
    pj, label_text=plot_Tj(T_l,T_r,n,lam)
    ax2.plot(Veriga,pj,"--", alpha=0.8, label=label_text,color="black")
    ax1.grid()
    ax1.set_xlabel("N")
    ax1.set_ylabel(r"$\langle T \rangle$")
    ax2.set_xlabel("N")
    ax2.grid()
    ax2.set_ylabel(r"$\langle T \rangle$")
    plt.show()



step_tau=1
fit_J_VEC="Yes"
if fit_J_VEC=="Yes":
    lam_list=[10,15,20,30,40]
    jndex=0
    t_fin=50000
    fig4 = plt.figure()
    ax6 = fig4.add_subplot(2,1,1)
    ax8 = fig4.add_subplot(2,1,2)
    fig5 = plt.figure()
    ax7 = fig5.add_subplot(2,1,1)
    ax9 = fig5.add_subplot(2,1,2)
    mu_list=[]
    mu_list_Max=[]
    a=0
    lam=1
    tau=5
    fig4.suptitle(r"Prikaz povprečnih vrednosti $\langle J \rangle$ pri $\tau=%.2f$, $t_{fin}=%.2f$ in $\lambda=%.1f$" %(tau, t_fin,lam))
    fig5.suptitle(r"Prikaz odvisnosti toplotnega toka od velikosti verige pri $\tau=%.2f$, $\lambda=%.2f$" %(tau, lam))
    for n in tqdm(lam_list):
        lam=1
        index=0
        length=1
        init_state=initial_conditions(length, n,T_r,T_l)
        tau=1
        avg_T=[]
        T_line=[]
        avg_T_Max=[]
        T_line_Max=[]
        n_functions=np.linspace(1,5,4)
        for i in tqdm(n_functions):
            init_state=initial_conditions(length, n,T_r,T_l)
            q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0(t_fin, init_state, n, tau, T_l, T_r,lam)
            q_matrix_Max, p_matrix_Max, zeta_T_Max, zeta_R_Max,t_Max=solving_eq_same_0_Maxwell(t_fin, step_tau,init_state, n, tau, T_l, T_r,lam)
            q=q_matrix
            p=p_matrix
            q_Max=q_matrix_Max
            p_Max=p_matrix_Max
            T, Veriga=cal_J(q,p,t,a)
            T_Max, Veriga_Max=cal_J(q_Max,p_Max,t_Max,a)
            index+=1
            avg_T.append(sum(T/len(T)))
            avg_T_Max.append(sum(T_Max/len(T_Max)))
            T_line.append(T)
            T_line_Max.append(T_Max)
        T_line=np.array(T_line)
        T_avg=[]
        T_line_Max=np.array(T_line_Max)
        T_avg=[]
        T_avg_Max=[]
        for i in range(len(T_line.T)):
            T_avg.append(sum(T_line.T[i])/len(T_line.T[i]))
            T_avg_Max.append(sum(T_line_Max.T[i])/len(T_line_Max.T[i]))
        ax6.plot(Veriga,T_avg, label=r"$n=%i$" %n, color=plt.cm.coolwarm(jndex/(3*len(lam_list))))
        ax6.set_title(r"Nose-Hoover")
        ax6.set_xlabel("N")
        ax6.set_ylabel(r"$\langle J \rangle$")
        ax6.grid()
        ax6.legend()

        ax8.plot(Veriga_Max,T_avg_Max, label=r"$n=%i$" %n, color=plt.cm.coolwarm(jndex/(3*len(lam_list))))
        ax8.set_title(r"Maxwell")
        ax8.set_xlabel("N")
        ax8.set_ylabel(r"$\langle J \rangle$")
        ax8.grid()
        ax8.legend()
        (mu, sigma) = norm.fit(avg_T)
        mu_list.append(mu)

        (mu_Max, sigma_Max) = norm.fit(avg_T_Max)
        mu_list_Max.append(mu_Max)
        jndex+=3
    popt, pcov =scipy.optimize.curve_fit(lambda x, C,kappa: fit_j(x,kappa,C,T_r-T_l), lam_list, mu_list)
    ax7.set_title(r"Nose Hoover")
    ax7.plot(lam_list,-np.array(mu_list), label="Model")
    ax7.plot(lam_list,-np.array(fit_j(lam_list,popt[1],popt[0],T_r-T_l)),label=r"$Fit: \kappa=%.4f$" %(-popt[1]))
    ax7.set_xlabel(r"$n$")
    ax7.set_ylabel(r"$J$")
    ax7.grid()
    ax7.legend()
    popt, pcov =scipy.optimize.curve_fit(lambda x, C,kappa: fit_j(x,kappa,C,T_r-T_l), lam_list, mu_list_Max)
    ax9.set_title(r"Maxwell")
    ax9.plot(lam_list,-np.array(mu_list_Max), label="Model")
    ax9.plot(lam_list,-np.array(fit_j(lam_list,popt[1],popt[0],T_r-T_l)),label=r"$Fit: \kappa=%.4f$" %(-popt[1]))
    ax9.set_xlabel(r"$n$")
    ax9.set_ylabel(r"$J$")
    ax9.grid()
    ax9.legend()
    plt.show()


T_plot="Yes"
if T_plot=="Yes":
    t_list=[100,1000,10000,25000]
    index=0
    length=1
    tau=5
    n=10
    step_tau=0.5
    lam=1
    q_matrix0, p_matrix0, zeta_T0, zeta_R0,t0=solving_eq_same_0_Maxwell(t_fin, step_tau,init_state, n, tau, T_l, T_r,lam)
    T0, Veriga0=cal_T(p_matrix0,t0, T_r,T_l)
    init_state=initial_conditions(length, n,T_r,T_l)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,2,1)
    ax2 = fig1.add_subplot(1,2,2)
    fig1.suptitle(r"Prikaz $T$ v odvisnosti od končnega časa pri $\lambda=%.2f$ in $\tau=%.2f$" %(lam, tau))
    n_func=10
    for t_fin in tqdm(t_list):
        for i in range(n_func):
            q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0_Maxwell(t_fin, step_tau,init_state, n, tau, T_l, T_r,lam)
            T, Veriga=cal_T(p_matrix,t, T_r,T_l)
            ax2.set_title(r"Maxwell")
            ax2.plot(Veriga,T, label=r"$t_{max}=%i$" %t_fin,color=plt.cm.coolwarm(index/len(t_list)))
            ax2.legend()
            q_matrix, p_matrix, zeta_T, zeta_R,t=solving_eq_same_0(t_fin, init_state, n, tau, T_l, T_r,lam)
            T, Veriga=cal_T(p_matrix,t, T_r,T_l)
            ax1.set_title(r"Nose-Hoover")
            ax1.plot(Veriga,T, label=r"$t_{max}=%i$" %t_fin,color=plt.cm.coolwarm(index/len(t_list)))
            ax1.legend()
            index+=1
    T_line=np.array(T_line)
    T_avg=[]
    for i in range(len(T_line.T)):
        T_avg.append(sum(T_line.T[i]/len(T_line.T[i])))
    pj, label_text=plot_Tj(T_l,T_r,n,lam)
    ax1.plot(Veriga,pj,"--", alpha=0.8, label=label_text,color="black")
    pj, label_text=plot_Tj(T_l,T_r,n,lam)
    ax2.plot(Veriga,pj,"--", alpha=0.8, label=label_text,color="black")
    ax1.grid()
    ax1.set_xlabel("N")
    ax1.set_ylabel(r"$\langle T \rangle$")
    ax2.set_xlabel("N")
    ax2.grid()
    ax2.set_ylabel(r"$\langle T \rangle$")
    plt.show()