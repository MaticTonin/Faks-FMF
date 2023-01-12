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

D0 = 99
B0 = 1
I0 = 0

D = D0
B = B0
I = I0

p = 1000

Dlist = []
Blist = []
Ilist = []
tlist = []

tavr = np.zeros(len(dtlist))
@jit(nopython=True)
def model(a,b,D0,B0,dt,I0,g):
    Dlist = []
    Blist = []
    Ilist = []
    tlist = []
    D = D0
    B = B0
    I = I0
    t = 0
    while B > 0:
        if t > 60:
            tlist.append(t)
            Dlist.append(D)
            Blist.append(B)
            Ilist.append(I)
            break
        tlist.append(t)
        Dlist.append(D)
        Blist.append(B)
        Ilist.append(I)
        sprememba_I=random.poisson(lam=g*I*dt) 
        sprememba_D=random.poisson(lam=a*D*B*dt)
        sprememba_B=random.poisson(lam=b*B*dt)
        dD = -sprememba_D+ sprememba_I
        dB = +sprememba_D - sprememba_B 
        dI = sprememba_B - sprememba_I

        D = D + dD
        B = B + dB
        I = I + dI
        t = t + dt

        if D < 0:
            D = 0
        if I < 0:
            I = 0
        tlist.append(t)
        Dlist.append(D)
        Blist.append(B)
        Ilist.append(I)
    return tlist, Dlist, Blist, Ilist,t
"""
index=0
jndex=0
plt_izid="No"
plt_his="Yes"
if plt_his=="Yes":
    dtlist =  [2,1,0.1,0.01,0.001]
    p=10000

for j in range(len(dtlist)):
    Dlist = []
    Blist = []
    Ilist = []
    tlist = []
    dt = dtlist[j]
    t_hist=[]
    for i in tqdm(range(p)):
        tlist, Dlist, Blist, Ilist,t=model(a,D0,B0,dt,I0,g)
        if (index==p-1 or index==p-2) and plt_izid=="Yes":
            t_line=np.linspace(0,t,len(tlist))
            plt.title("Prikaz izida populacije za $D_0$ = "+str(D0)+r'$, B_0 = $'+str(B0)+r'$, I_0 = $'+str(I0)+r', $\alpha$ = '+str(a)+r', $\beta = $'+str(b)+r', $\gamma = $'+str(g))
            plt.plot(tlist, Dlist, color = plt.cm.Reds(index/p), label="Dovzetni $%i$" %(index-1))
            plt.plot(tlist, Blist, color = plt.cm.Blues(index/p), label="Bolni $%i$"%(index-1))
            plt.plot(tlist, Ilist, color = plt.cm.Greens(index/p), label="Imuni $%i$"%(index-1))
            plt.axvline(x = t_line[len(t_line)-1], color = "black", label = 'Smrt populacije $%i$ t=$%.2f$' %(index-1,t_line[len(t_line)-1]))
            plt.ylabel('N')
            plt.xlabel('t')
        index+=1
        t_hist.append(t)
    if plt_his=="Yes" and len(tlist)>1:
        plt.title("Prikaz histogramov za $D_0$ = "+str(D0)+r'$, B_0 = $'+str(B0)+r'$, I_0 = $'+str(I0)+r', $\alpha$ = '+str(a)+r', $\beta = $'+str(b)+r', $\gamma = $'+str(g))
        plt.hist(t_hist,bins = int(np.amax(t_hist)), label = r'$dt= $'+ str(dt), color = plt.cm.rainbow(jndex/len(dtlist)),edgecolor='black',alpha = 0.8)
        plt.ylabel(r'$t_{death}$')
        plt.xlabel('N')
    jndex+=1
        
  
plt.legend()
plt.grid()
plt.show()"""


index=0
jndex=0
plt_izid="No"
plt_his="Yes"
plt_graph="No"
p = 4
if plt_his=="Yes":
    dgammalist = [0.00001,0.00005,0.0001,0.001,0.005,0.01,0.02,0.03,0.04]
    p=50000
if plt_graph=="Yes":
    dgammalist = np.logspace(-2,1,70)
    p=1000
t_graph=[]
for j in tqdm(range(len(dgammalist))):
    p_none=0
    Dlist = []
    Blist = []
    Ilist = []
    tlist = []
    dt = 0.5
    t_hist=[]
    index=0
    for i in range(p):
        tlist, Dlist, Blist, Ilist,t=model(a,b,D0,B0,dt,I0,dgammalist[j])
        if (index==p-1 or index==p-2) and plt_izid=="Yes" and p<5:
            t_line=np.linspace(0,t,len(tlist))
            plt.title("Prikaz populacije za $D_0$ = "+str(D0)+r'$, B_0 = $'+str(B0)+r'$, I_0 = $'+str(I0)+r', $\alpha$ = '+str(a)+r', $\beta = $'+str(b)+r', $\gamma = $'+str(dgammalist[j]))
            plt.plot(tlist, Dlist, color = plt.cm.Reds(index/p), label="Dovzetni $%i$" %(index-1))
            plt.plot(tlist, Blist, color = plt.cm.Blues(index/p), label="Bolni $%i$"%(index-1))
            plt.plot(tlist, Ilist, color = plt.cm.Greens(index/p), label="Imuni $%i$"%(index-1))
            plt.axvline(x = t_line[len(t_line)-1], color = "black", label = 'Smrt populacije $%i$ t=$%.2f$' %(index-1,t_line[len(t_line)-1]))
            plt.ylabel('N')
            plt.ylim([-0.5,105])
            plt.xlabel('t')
            plt.legend()
        index+=1
        if t>30 and plt_graph=="Yes":
            t_hist.append(t)
            p_none+=1
        else:
            t_hist.append(t)
    if plt_his=="Yes" and len(dgammalist)>1 and p>5:
        plt.title("Prikaz histogramov za $D_0$ = "+str(D0)+r'$, B_0 = $'+str(B0)+r'$, I_0 = $'+str(I0)+r', $\alpha$ = '+str(a)+r', $\beta = $'+str(b))
        plt.hist(t_hist,bins = int(np.amax(t_hist)), label = r'$\gamma= $'+ str(dgammalist[j]), color = plt.cm.rainbow(jndex/len(dgammalist)),edgecolor='black',alpha = 0.8)
        plt.ylabel(r'$t_{death}$')
        plt.xlabel('N')
    if plt_graph=="Yes":
        t_graph.append(sum(t_hist)/p_none)
    jndex+=1
if plt_graph=="Yes" and len(dgammalist)>1:
    plt.title("Prikaz časa smrti od $\gamma$ $D_0$ = "+str(D0)+r'$, B_0 = $'+str(B0)+r'$, I_0 = $'+str(I0)+r', $\alpha$ = '+str(a)+r', $\beta = $'+str(b))
    plt.plot(dgammalist, t_graph)
    plt.xscale("log")
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$t_{death}$')
    plt.legend()
    plt.grid()
    plt.show()  
  
plt.legend()
plt.grid()
plt.show()


#plt.fill_between(dtlist, 0, liz/(liz+ziz+ni), color = 'b', label = 'Lisice')
#plt.fill_between(dtlist, liz/(liz+ziz+ni), liz/(liz+ziz+ni) + ziz/(liz+ziz+ni), color = 'r', label = 'Zajci')
#plt.fill_between(dtlist, liz/(liz+ziz+ni) + ziz/(liz+ziz+ni), 1,  color = 'k', label = 'time limit')
#
#plt.xscale('log')
#plt.xlabel(r'$dt$')
#plt.ylabel(r'$P$')
#plt.legend()
#plt.grid()
#plt.show()

#plt.hist(tlist,bins = int(np.amax(tlist)/10), label = r'$\beta = $'+ str(b), color = colors[0],alpha = 0.6)





#plt.xlabel(r'$t_{izumrtje}$')
#plt.ylabel('N')
#plt.grid()
#plt.show()
