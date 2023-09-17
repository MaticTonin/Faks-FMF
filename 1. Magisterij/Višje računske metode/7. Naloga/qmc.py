import numpy as np
from numba import jit

@jit(nopython=True)
def potential(x, lam):
    return 0.5*np.power(x, 2) + lam*np.power(x, 4)

@jit(nopython=True)
def acceptance_rate(config, shape, beta, eps, lam, n):
    m = shape[0]
    ar = [0., 0.]
    ar_line=[]
    for i in range(n):
        j = np.random.randint(0, m)
        qj = config[j]
        qj_new  = qj + eps*np.random.normal()
        
        a = np.power(qj_new, 2) - np.power(qj, 2) + (qj - qj_new)*(config[(j+1)%m] + config[(j-1)%m])
        b = potential(qj_new, lam) - potential(qj, lam)
        c = (-1.)*(m/beta)*(a) - (b)*beta/m
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
def estimate_eps(config, shape, beta, eps, lam, n, delta=1e-2):
    ar,ar_line = acceptance_rate(config, shape, beta, eps, lam, n)
    while np.absolute(ar - 0.33) > 0.01:
        if ar > 0.33:
            eps += delta*np.random.rand()
        else:
            eps -= delta*np.random.rand()
        ar,ar_line = acceptance_rate(config, shape, beta, eps, lam, n)
    #print(ar)
    return eps
@jit(nopython=True)

def initial_Energy(start_state_2, beta,lam,M):
    e = 0.
    h_v = [0.5*M/beta, potential(start_state_2[0], lam)] # [H, V]
    for i in range(M):
        q = np.power(start_state_2[(i+1)%M] - start_state_2[i], 2)
        pot = potential(start_state_2[i], lam)
        e += 0.5*q
        e += pot
        h_v[0] -= 0.5*M*q/np.power(beta, 2)
        h_v[0] += pot/M 
    return e, h_v

@jit(nopython=True)
def qmc_move(config, shape, beta, par=[1., 0.], rep=1):
    eps, lam = par
    m = shape[0]
    dE = 0.
    dextra = [0., 0.] # [H, V]
    Energy_list=[]
    H_list=[]
    V_list=[]
    epsilon=[]

    eps = estimate_eps(config, shape, beta, eps, lam, n=100)
    for i in range(rep):
        j = np.random.randint(0, m)
        qj = config[j]
        qj_new  = qj + eps*np.random.normal()
        
        a = np.power(qj_new, 2) - np.power(qj, 2) + (qj - qj_new)*(config[(j+1)%m] + config[(j-1)%m])
        b = potential(qj_new, lam) - potential(qj, lam)
        c = (-1.)*(m/beta)*(a) - (b)*beta/m
        # minimum(1, exp(c))
        if c <= 0.:
            prob = np.exp(c)
        else:
            prob = 1.
        rand = np.random.rand()
        if rand < prob:
            q0_old = config[0]
            config[j] = qj_new
            dE += a
            dE += b
            dextra[0] += (-1.)*m*a/np.power(beta, 2) + b/m
            dextra[1] += potential(config[0], lam) - potential(q0_old, lam)
        if i%(rep/100)==0:
            e, h_v=initial_Energy(config, beta,lam,m)
            Energy_list.append(e)
            H_list.append(h_v[0])
            V_list.append(h_v[1])
            epsilon.append(eps)
    return config, dE, dextra, Energy_list, H_list, V_list ,epsilon

@jit(nopython=True)
def qmc_eval(config, shape, beta, par):
    lam = par[1]
    m = shape[0]
    e = 0.
    extra = [0.5*m/beta, potential(config[0], lam)] # [H, V]
    for i in range(m):
        a = np.power(config[(i+1)%m] - config[i], 2)
        b = potential(config[i], lam)
        e += 0.5*a
        e += b
        extra[0] -= 0.5*m*a/np.power(beta, 2)
        extra[0] += b/m 
    return e, extra