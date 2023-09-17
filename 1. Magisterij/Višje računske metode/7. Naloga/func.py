import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.random import rand
from numba import jit,njit
import scipy

@jit(nopython=True)
def potencial(x, lam):
    return 0.5*np.power(x,2)+lam*np.power(x,4)

@jit(nopython=True)
def acceptance_rate(config, M, beta, eps, lam, n):
    ar = [0.,0]
    ar_line=[]
    for i in range(n):
        j = np.random.randint(0, M)
        qj = config[j]
        qj_new  = qj + eps*np.random.normal()
        
        diffq = np.power(qj_new, 2) - np.power(qj, 2) + (qj - qj_new)*(config[(j+1)%M] + config[(j-1)%M])
        diffpot = potencial(qj_new, lam) - potencial(qj, lam)
        c = (-1.)*(M/beta)*(diffq) - (diffpot)*beta/M
        # minimum(1, exp(c))
        if c <= 0.:
            prob = np.exp(c)
        else:
            prob = 1.
        rand = np.random.rand()
        if rand < prob:
            ar[0] += 1
        else:
            ar[1] += 1
        if  i!=0:
            ar_line.append(ar[0]/(i+1))
        if i==0:
            ar_line.append(ar[0])
    return ar[0]/n, ar_line

@jit(nopython=True)
def estimate_eps(config, M, beta, eps, lam, n=100, delta=1e-2):
    ar,ar_line= acceptance_rate(config, M, beta, eps, lam, n)
    eps_copy=eps
    while np.absolute(ar - 0.33) > 0.01:
        if ar > 0.33:
            eps_copy += delta*np.random.rand()
        else:
            eps_copy -= delta*np.random.rand()
        #print(ar, eps_copy,np.absolute(ar - 0.33))
        ar,ar_line = acceptance_rate(config, M, beta, eps_copy, lam, n)
    #print(ar)
    return eps_copy
@jit(nopython=True)
def initial_Energy(start_state_2, beta,lam,M):
    e = 0.
    h_v = [0.5*M/beta, potencial(start_state_2[0], lam)] # [H, V]
    for i in range(M):
        q = np.power(start_state_2[(i+1)%M] - start_state_2[i], 2)
        pot = potencial(start_state_2[i], lam)
        e += 0.5*q
        e += pot
        h_v[0] -= 0.5*M*q/np.power(beta, 2)
        h_v[0] += pot/M 
    return e, h_v

@jit(nopython=True)
def Quantum_monte_carlo(start_state_1,beta,steps,lam, eps,M):
    M=len(start_state_1)
    Energy_list=[]
    H_list=[]
    V_list=[]
    epsilon=[]
    dE=0
    dH_V=[0,0]        
    eps_copy=estimate_eps(start_state_1, M, beta, eps, lam, steps, delta=1e-1)
    eps=eps_copy
    for i in range(steps):
        #eps_copy=estimate_eps(start_state_1, M, beta, eps, lam, steps, delta=1e-1)
        #eps=eps_copy
        j=np.random.randint(0,M)
        qj=start_state_1[j]
        qj_new=qj+eps*np.random.normal(0,1)

        dif_q=np.power(qj_new, 2) - np.power(qj, 2) + (qj - qj_new)*(start_state_1[(j+1)%M] + start_state_1[(j-1)%M])
        dif_pot=potencial(qj_new, lam) - potencial(qj, lam)
        diff_all=(-1)*M*dif_q/beta-dif_pot*beta/M
        if diff_all<=0.:
            probability=np.exp(diff_all)
        else:
            probability=1
        zeta=np.random.rand()
        if zeta<probability:
            q0_old=start_state_1[0]
            start_state_1[j]=qj_new
            dE+=dif_pot
            dE+=dif_q
            dH_V[0]+=(-1.)*M*dif_q/np.power(beta, 2) + dif_pot/M
            dH_V[1]+=potencial(start_state_1[0], lam) - potencial(q0_old, lam)
        #if i%1000==0:
            #eps_copy=estimate_eps(start_state_1, M, beta, eps, lam, steps, delta=1e-1)
            #eps=eps_copy
        if i%(steps/100)==0:
            e, h_v=initial_Energy(start_state_1, beta,lam,M)
            Energy_list.append(e)
            H_list.append(h_v[0])
            V_list.append(h_v[1])
            epsilon.append(eps)
    fin_state=start_state_1.copy()
    return fin_state, dE, dH_V, Energy_list, H_list, V_list ,epsilon

@jit(nopython=True)
def initial_Energy(start_state_2, beta,lam,M):
    e = 0.
    h_v = [0.5*M/beta, potencial(start_state_2[0], lam)] # [H, V]
    for i in range(M):
        q = np.power(start_state_2[(i+1)%M] - start_state_2[i], 2)
        pot = potencial(start_state_2[i], lam)
        e += 0.5*q
        e += pot
        h_v[0] -= 0.5*M*q/np.power(beta, 2)
        h_v[0] += pot/M 
    return e, h_v


@jit(nopython=True)
def Evaluation(start_state,beta,reps,lam, eps, steps,M):
    initial_E, initial_H_V=initial_Energy(start_state, beta,lam,M)
    init_st=[]
    E=[initial_E]
    H=[initial_H_V[0]]
    V=[initial_H_V[1]]
    delta=1e-2
    eps_copy=estimate_eps(start_state, M, beta, eps, lam, steps, delta)
    for i in range(reps):
        print(i)
        start_state_done, dEnergy, dH_V,Energy_list,H_list,V_list ,epsilon=Quantum_monte_carlo(start_state,beta,steps,lam, eps_copy,M)
        init_st.append(start_state_done)
        E.append(E[-1]+dEnergy)
        H.append(H[-1]+dH_V[0])
        V.append(V[-1]+dH_V[1])
        #H_V[0].append(H_V[-1][0]+dH_V[0])
        #H_V[1].append(H_V[-1][1]+dH_V[1])
    return init_st, E, H, V,Energy_list,H_list,V_list,epsilon



