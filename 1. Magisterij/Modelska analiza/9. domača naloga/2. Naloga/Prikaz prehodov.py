import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

Ns = [25,250]
mu1 = 0
mu2 = 0 
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
    plt.title("Prikaz verjetnosti za izumrtje celotne populacije")
    plt.plot(tlist,P, label = r'$N_0$ = '+str(N0)+r', $\beta = $'+str(b)+r', dt = '+str(dt))

    plt.xlabel('t')
    plt.ylabel(r'$P_{izumrtje}$')
    plt.grid()
plt.legend()
plt.show()

plt.title("Prikaz povprečne vrednosti $\sigma$")
plt.plot(tlist,np.sqrt(mu2-mu1**2), label = r'$N_0$ = '+str(N0)+r', $\beta = $'+str(b)+r', dt = '+str(dt))

plt.xlabel('t')
plt.ylabel(r'$\sigma$')
plt.legend()
plt.show()

plt.plot(tlist,mu1, label = r'$N_0$ = '+str(N0)+r', $\beta = $'+str(b)+r', dt = '+str(dt))

plt.xlabel('t')
plt.ylabel(r'$\mu_{1}$')
plt.legend()
plt.show()

#plt.savefig('N100'+str(ime))




