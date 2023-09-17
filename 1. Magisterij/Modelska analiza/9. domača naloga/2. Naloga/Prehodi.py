import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from matplotlib import cm
from tqdm import tqdm
colors = ['r','b','g','m','c']  #pač barve

b = 1 
bs = 5*b
N0 = 250
dt  = np.logspace(-5,-1,100)

Sn = bs * N0

def P(s):
    return (Sn * dt)**s *np.exp(-Sn*dt)/np.math.factorial(s)

for i in range(1,4):
    plt.plot(dt, P(i), color = colors[i-1], label = 's = '+str(i))

plt.xlabel('dt')
plt.ylabel(r'$P_s$')
plt.xscale('log')

plt.title(r'$N_0$ = '+str(N0))
plt.hlines(0.01,10**(-5), 10**(-1), ls = '--' ,color='k')
plt.legend()
plt.show()


Ns = [25,250]
mu1 = 0
mu2 = 0
dt_list=[1,0.1,0.01,0.001]
index=1
N0=25

for dt in dt_list:
    colors = ['r','b','g','m','c']  #pač barve
    b = 1 
    bs = 5*b
    br = 4*b

    pop = np.array([])
    x = np.zeros(2*N0)
    y = np.zeros(2*N0)
    x[N0] = 1
    pop = np.array([x])
    M = np.zeros((2*N0, 2*N0))
    for i in range(2*N0):
        M[i][i] = 1 - i*(br + bs)*dt
        if i-1 >= 0:
            M[i-1][i] = i*bs*dt

        if i+1 < 2*N0:
            M[i+1][i] = i*br*dt
    plt.subplot(2, 2, index)
    plt.title(r"Prikaz matrike $M$ za $\beta=%.2f$ in $N_0=%i$ pri $dt=%.4f$" %(b,N0,dt))
    plt.imshow(M, cmap="coolwarm")
    plt.colorbar()
    index+=1
plt.show()
index=0
N0=250

dt_list=[1,0.1,0.01,0.001]
for dt in tqdm(dt_list):
    colors = ['r','b','g','m','c']  #pač barve
    b = 1 
    bs = 5*b
    br = 4*b

    pop = np.array([])
    x = np.zeros(2*N0)
    y = np.zeros(2*N0)
    x[N0] = 1
    pop = np.array([x])
    M = np.zeros((2*N0, 2*N0))
    for i in range(2*N0):
        M[i][i] = 1 - i*(br + bs)*dt
        if i-1 >= 0:
            M[i-1][i] = i*bs*dt

        if i+1 < 2*N0:
            M[i+1][i] = i*br*dt
    P = np.array([0])
    mu1 = np.array([0])
    mu2 = np.array([0])
    t = 0
    n = np.arange(0,2*N0)
    count = -1
    tmax = 7
    while t <= tmax:
        #if np.round(t, decimals=4) == 6.5  or np.round(t, decimals=4) == 0.25  or np.round(t, decimals=4) == 0.5 or np.round(t, decimals=4) == 1 or np.round(t, decimals=4) ==1.5 or np.round(t, decimals=4) == 0:
        #    plt.plot(n,x, label= 't = '+str(np.round(t, decimals=1)))
        count = count + 1
        for i in range(len(x)):
            for j in range(len(x)):
                y[i] = y[i] + M[i][j]*x[j]

        x = y
        pop = np.vstack([pop,x])
        P = np.append(P, x[0])
        sum1  = 0
        sum2  = 0
        for i in range(len(x)):
            sum1 = sum1 + i*x[i]
            sum2 = sum2 + i**2*x[i]
        mu1 = np.append(mu1, sum1)
        mu2 = np.append(mu2, sum2)

        y = np.zeros(2*N0)
        t = t + dt

    tlist = np.linspace(0,tmax, len(P))
    plt.plot(tlist,mu1, label = r'$N_0$ = '+str(N0)+r', $\beta = $'+str(b)+r', dt = '+str(dt), color=plt.cm.rainbow(index/len(dt_list)))
    index+=1
plt.title(r"Prikaz ustreznosti izbire $dt$ pri $N_0$ = "+str(N0)+r', $\beta = $'+str(b))
plt.xlim([-1,8])
plt.ylim([-1,26])
plt.xlabel('t')
plt.ylabel(r'$\mu_{1}$')
plt.grid()
plt.legend()
plt.show()

for N0 in Ns:
    colors = ['r','b','g','m','c']  #pač barve
    b = 1 
    bs = 5*b
    br = 4*b

    if N0 == 25:
        dt = 10**(-3)

    elif N0 == 250:
        dt = 10**(-4)

    pop = np.array([])
    x = np.zeros(2*N0)
    y = np.zeros(2*N0)
    x[N0] = 1
    pop = np.array([x])
    print(pop)
    M = np.zeros((2*N0, 2*N0))
    for i in range(2*N0):
        M[i][i] = 1 - i*(br + bs)*dt
        if i-1 >= 0:
            M[i-1][i] = i*bs*dt

        if i+1 < 2*N0:
            M[i+1][i] = i*br*dt
    plt.title(r"Prikaz matrike $M$ za $\beta=%.2f$ in $N_0=%i$ pri $dt=%.4f$" %(b,N0,dt))
    plt.imshow(M, cmap="coolwarm")
    plt.colorbar()
    plt.show()
    P = np.array([0])
    mu1 = np.array([0])
    mu2 = np.array([0])
    t = 0
    n = np.arange(0,2*N0)
    count = -1

    tmax = 7
    while t <= tmax:

        #if np.round(t, decimals=4) == 6.5  or np.round(t, decimals=4) == 0.25  or np.round(t, decimals=4) == 0.5 or np.round(t, decimals=4) == 1 or np.round(t, decimals=4) ==1.5 or np.round(t, decimals=4) == 0:
        #    plt.plot(n,x, label= 't = '+str(np.round(t, decimals=1)))

        count = count + 1
        for i in range(len(x)):
            for j in range(len(x)):
                y[i] = y[i] + M[i][j]*x[j]

        x = y
        pop = np.vstack([pop,x])
        P = np.append(P, x[0])
        sum1  = 0
        sum2  = 0
        for i in range(len(x)):
            sum1 = sum1 + i*x[i]
            sum2 = sum2 + i**2*x[i]


        mu1 = np.append(mu1, sum1)
        mu2 = np.append(mu2, sum2)
    





        y = np.zeros(2*N0)
        t = t + dt


    tlist = np.linspace(0,tmax, len(P))



    #
    #plt.xlabel('n')
    #plt.ylabel('P')
    #plt.vlines(x=N0, ymin =0, ymax =1, ls = '--', color = 'k')
    #plt.legend()
    #plt.title(r'$N_0$ = '+str(N0)+r', $\beta = $'+str(b)+r', dt = '+str(dt))
    #plt.show()

    plt.plot(tlist,P, label = r'$N_0$ = '+str(N0)+r', $\beta = $'+str(b)+r', dt = '+str(dt))

    plt.xlabel('t')
    plt.ylabel(r'$P_{izumrtje}$')
    plt.grid()
plt.legend()
plt.show()

plt.plot(tlist,np.sqrt(mu2-mu1**2), label = r'$N_0$ = '+str(N0)+r', $\beta = $'+str(b)+r', dt = '+str(dt))

plt.xlabel('t')
plt.ylabel(r'$\sigma$')
plt.grid()
plt.legend()
plt.show()

plt.plot(tlist,mu1, label = r'$N_0$ = '+str(N0)+r', $\beta = $'+str(b)+r', dt = '+str(dt))

plt.xlabel('t')
plt.ylabel(r'$\mu_{1}$')
plt.grid()
plt.legend()
plt.show()

#plt.savefig('N100'+str(ime))




