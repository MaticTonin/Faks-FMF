import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numba import jit,njit
from tqdm import tqdm

colors = ['r','b','g','m','c']  #pač barve

dt = 0.01
dtlist = [0.2]


a= 0.01
b = 0.2
g = 0.01
m=0.0001
D0 = 99
B0 = 1
I0 = 0

D = D0
B = B0
I = I0

p = 4

Dlist = []
Blist = []
Ilist = []
tlist = []

tavr = np.zeros(len(dtlist))
@jit(nopython=True)
def model(a,D0,B0,dt,I0,g):
    Dlist = []
    Blist = []
    Ilist = []
    tlist = []
    Deadlist=[]
    D = D0
    B = B0
    I = I0
    Dead=0
    t = 0
    while B > 0:
        if t > 5000:
            tlist.append(t)
            Dlist.append(D)
            Blist.append(B)
            Ilist.append(I)
            Deadlist.append(Dead)
            break
        tlist.append(t)
        Dlist.append(D)
        Blist.append(B)
        Ilist.append(I)
        Deadlist.append(Dead)

        dD = -random.poisson(lam=a*D*B*dt) + random.poisson(lam=g*I*dt)
        dB = +random.poisson(lam=a*D*B*dt) - random.poisson(lam=b*B*dt)-random.poisson(lam=m*I*dt)  
        dI = random.poisson(lam=b*B*dt) - random.poisson(lam=g*I*dt)
        dDead = + random.poisson(lam=g*I*dt)  

        D = D + dD
        B = B + dB
        I = I + dI
        t = t + dt
        Dead= Dead +dDead

        if D < 0:
            D = 0
        if I < 0:
            I = 0
        tlist.append(t)
        Dlist.append(D)
        Blist.append(B)
        Ilist.append(I)
        Deadlist.append(Dead)
    return tlist, Dlist, Blist, Ilist,t,Deadlist
index=0
for j in range(len(dtlist)):
    Dlist = []
    Blist = []
    Ilist = []
    tlist = []
    dt = dtlist[j]
    for i in tqdm(range(p)):
        tlist, Dlist, Blist, Ilist,t, Deadlist=model(a,D0,B0,dt,I0,g)
        if (index==p-1 or index==p-2):
            t_line=np.linspace(0,t,len(tlist))
            plt.title("Prikaz izida populacije za $D_0$ = "+str(D0)+r'$, B_0 = $'+str(B0)+r'$, I_0 = $'+str(I0)+r', $\alpha$ = '+str(a)+r', $\beta = $'+str(b)+r', $\gamma = $'+str(g))
            plt.plot(tlist, Dlist, color = plt.cm.Reds(index/p), label="Dovzetni $%i$" %(index-1))
            plt.plot(tlist, Blist, color = plt.cm.Blues(index/p), label="Bolni $%i$"%(index-1))
            plt.plot(tlist, Ilist, color = plt.cm.Greens(index/p), label="Imuni $%i$"%(index-1))
            plt.plot(tlist, Deadlist,"--", color = plt.cm.Greys(index/p), label="Mrtvi $%i$"%(index-1))
            plt.axvline(x = t_line[len(t_line)-1], color = "black", label = 'Smrt populacije $%i$ t=$%.2f$' %(index-1,t_line[len(t_line)-1]))
            plt.ylabel('N')
            plt.xlabel('t')
        index+=1
        tavr[j] = tavr[j] + t
    tavr[j] = tavr[j]/p
        
  


#
plt.legend()
plt.xlabel(r't')
plt.ylabel(r'N')
plt.grid()
plt.show()