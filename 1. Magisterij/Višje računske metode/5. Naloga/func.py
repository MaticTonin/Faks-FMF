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


def solving_eq(t_fin,length, particles, t_n, T_l, T_r,lam):
    a=length/particles
    q0=np.arange(-length/2,length/2,a)
    p0=np.random.randn(particles)
    norm=np.sum(p0**2)
    p0/=np.sqrt(norm)
    zeta_T0=np.random.randn(1)
    zeta_R0=np.random.randn(1)
    tau=t_fin/t_n
    n=particles
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
    n=particles
    t=np.linspace(0,t_fin,t_n)
    sol=solve_ivp(derivates,(0,t_fin),init_state, method="DOP853",t_eval=np.linspace(0,t_fin,t_n),args=(lam, tau, T_l,T_r,particles,))
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
            pj[j]=T_l+(j-1)/(n-1)*(T_r-T_l)
            label_text=r"$T_L+\frac{j-1}{N-1}(T_r-T_l)$"
    if lam==0:
        pj=np.zeros(n)
        for j in range(n):
            pj[j]=(T_l+T_r)/2
            label_text=r"$\frac{T_l+T_r}{2}$"
    return pj, label_text


