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
import matplotlib.cm as cmž


def exp_T(z,p,q):
    q[0]=q[0]+z*p[0]
    q[1]=q[1]+z*p[1]
    return p,q
def exp_V(z,p,q, lam):
    p[0]=p[0]-(1+2*lam*q[1]**2)*z*q[0]
    p[1]=p[1]-(1+2*lam*q[0]**2)*z*q[1]
    return p,q

def S_2(p,q,tau,lam):
    p_t,q_t=exp_T(tau/2,p,q)
    p_t,q_t=exp_V(tau,p_t,q_t,lam)
    p_t,q_t=exp_T(tau/2,p_t,q_t)
    return p_t,q_t

def S_3(p,q,tau,lam):
    p_1=1/4*(1+1j/np.sqrt(3))
    p_5=np.conjugate(p_1)
    p_2=2*p_1
    p_4=np.conjugate(p_2)
    p_3=1/2
    p,q=exp_T(tau*p_5,p,q)
    p,q=exp_V(tau*p_4,p,q,lam)
    p,q=exp_T(tau*p_3,p,q)
    p,q=exp_V(tau*p_2,p,q,lam)
    p,q=exp_T(tau*p_1,p,q)
    return p,q


def S_3_conju(p,q,tau,lam):
    p_1=1/4*(1+1j/np.sqrt(3))
    p_5=np.conjugate(p_1)
    p_2=2*p_1
    p_4=np.conjugate(p_2)
    p_3=1/2
    p,q=exp_T(tau*np.conjugate(p_5),p,q)
    p,q=exp_V(tau*np.conjugate(p_4),p,q,lam)
    p,q=exp_T(tau*np.conjugate(p_3),p,q)
    p,q=exp_V(tau*np.conjugate(p_2),p,q,lam)
    p,q=exp_T(tau*np.conjugate(p_1),p,q)
    return p,q

def S_4(p,q,tau,lam):
    x_0=-2**(1/3)/(2-2**(1/3))
    x_1=1/(2-2**(1/3))
    p,q=S_2(p,q,tau*x_1,lam)
    p,q=S_2(p,q,tau*x_0,lam)
    p,q=S_2(p,q,tau*x_1,lam)
    return p,q

def S_5(p,q,tau,lam):
    p,q=S_3_conju(p,q,tau/2,lam)
    p,q=S_3(p,q,tau/2,lam)
    return p,q
    


def method(N,tau,lam,func,p0,q0):
    p_matrix,q_matrix=[p0.copy()], [q0.copy()]
    Energy=[]
    for i in range(N):
        Energy.append(H(q0.copy()[0],q0.copy()[1],p0.copy()[0],p0.copy()[1],lam))
        p,q=func(p0,q0,tau,lam)
        p0,q0=p.copy(),q.copy()
        p_matrix.append(p0.copy())
        q_matrix.append(q0.copy())
    p_matrix=np.array(p_matrix)
    q_matrix=np.array(q_matrix)
    Energy=np.array(Energy)
    return p_matrix,q_matrix,Energy


def time_dependance(N,tau,lam,func,p0,q0,epsilon):
    p_matrix,q_matrix=[p0.copy()], [q0.copy()]
    Energy_0=H(q0.copy()[0],q0.copy()[1],p0.copy()[0],p0.copy()[1],lam)
    for i in range(N):
        p,q=func(p0,q0,tau,lam)
        Energy=H(q0.copy()[0],q0.copy()[1],p0.copy()[0],p0.copy()[1],lam)
        if abs(Energy-Energy_0)/Energy_0>epsilon:
            break
        p0,q0=p.copy(),q.copy()
        p_matrix.append(p0.copy())
        q_matrix.append(q0.copy())
    return i*tau

def H(q0,q1,p0,p1,lam):
    return 1/2*p0**2+1/2*q0**2+1/2*p1**2+1/2*q1**2+lam*q0**2*q1**2

def ders(t,state,lamb):
    return np.array([state[2],state[3],-state[0]*(1+2*lamb*state[1]**2),-state[1]*(1+2*lamb*state[0]**2)])

from scipy.integrate import solve_ivp
def dop853(init_state, lamb, t_fin, N):
    sol=solve_ivp(ders,(0,t_fin),init_state, method="DOP853",t_eval=np.linspace(0,t_fin,N),args=(lamb,),rtol=10**(-12),atol=10**(-13))
    Energy=[]
    for i in range(len(sol.y[0])):
        Energy.append(H(sol.y[0,i], sol.y[1,i], sol.y[2,i], sol.y[3,i],lamb))
    Energy=np.array(Energy)
    return sol.y[0], sol.y[1], sol.y[2], sol.y[3],Energy #q0,q1,p0,p1

def dop853_no_energy(init_state, lamb, t_fin, N):
    sol=solve_ivp(ders,(0,t_fin),init_state, method="DOP853",t_eval=np.linspace(0,t_fin,N),args=(lamb,),rtol=10**(-12),atol=10**(-13))
    return sol.y[0], sol.y[1], sol.y[2], sol.y[3] #q0,q1,p0,p1
def integration_t(x,t):
    value=1/t[len(t)-1]*scipy.integrate.simpson(x,t)
    return value


def average_value(N,tau,lam,func,p0,q0,t):
    Energy=[]
    p1_=[p0[0]]
    p2_=[p0[1]]
    p1_avg,p2_avg=[p0[0]],[p0[1]]
    t=[0]
    for i in range(1,N):
        t.append(tau*i)
        Energy.append(H(q0.copy()[0],q0.copy()[1],p0.copy()[0],p0.copy()[1],lam))
        p,q=func(p0,q0,tau,lam)
        p0,q0=p.copy(),q.copy()
        p1_.append(p[0].copy())
        p2_.append(p[1].copy())
        p1_avg.append(integration_t(np.array(p1_)*np.array(p1_),t))
        p2_avg.append(integration_t(np.array(p2_)*np.array(p2_),t))
    return t, p1_avg,p2_avg

def average_value_RK(init_state,lam,t_fin,N,t):
    q0,q1,p0,p1,=dop853_no_energy(init_state,lam,t_fin,N)
    t=np.linspace(0,t_fin,N)
    p1_avg,p2_avg=[p0[0]],[p0[1]]
    for i in range(1,len(t)):
        p1_avg.append(integration_t(np.array(p0[:i])*np.array(p0[:i]),t[:i]))
        p2_avg.append(integration_t(np.array(p1[:i])*np.array(p1[:i]),t[:i]))
    return t, p1_avg,p2_avg