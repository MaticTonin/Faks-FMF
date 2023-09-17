import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numba import jit,njit
from tqdm import tqdm
colors = ['r','b','g','m','c']  #pač barve

dt = 0.01
dtlist = np.logspace(-4.5,-2,30)
#dtlist =  [2,1,0.1,0.01,0.001]
a= 1
b = 1
a_list=[10,5,1,0.1]

L0 = 50
Z0 = 200

L = L0
Z = Z0

p = 5000

Zlist = []
Llist = []
tlist = []

tavr = np.zeros(len(dtlist))
ziz = np.zeros(len(dtlist))
liz = np.zeros(len(dtlist))
ni = np.zeros(len(dtlist))


@jit(nopython=True)
def model(a,b,dt,Z0,L0):
    Zlist = []
    Llist = []
    L = L0
    Z = Z0
    t = 0
    while L > 0 and Z > 0:
        if t > 500:
            break
        Zlist.append(Z)
        Llist.append(L)
        dZ = +random.poisson(lam=5*a*Z*dt) - random.poisson(lam=4*a*Z*dt) - random.poisson(lam=a/L0*Z*L*dt)
        dL = +random.poisson(lam=4*b*L*dt) - random.poisson(lam=5*b*L*dt) + random.poisson(lam=b/Z0*Z*L*dt)

        Z = Z + dZ
        L = L + dL
        t = t + dt
    return Z, L, t, Zlist, Llist
index=0
jndex=0
plot_sep="No"
plt_his="No"
plt_time="Yes"
tlist = []
t_graph=[]
for j in tqdm(range(len(dtlist))):
    dt = dtlist[j]
    if plt_time!="Yes":
        tlist=[]
    for i in range(p):
        Z,L,t,Zlist,Llist=model(a,b,dt,Z0,L0)
        if Z == 0:
            ziz[j] = ziz[j] +1
        elif L == 0:
            liz[j] =  liz[j] +1
        else:
            ni[j] = ni[j] + 1
        
        if p < 5 and plot_sep=="No":
            plt.title("Prikaz faznega diagrama zajcev in lisic za $Z_0$ = "+str(Z0)+r'$, L_0 = $'+str(L0)+r', $\alpha$ = '+str(a)+r', $\beta = $'+str(b))
            plt.plot(Zlist, Llist, color = plt.cm.rainbow(index/p), label="Ponovitev "+str(i+1))
            plt.xlabel('Z')
            plt.ylabel('L')
        if (index==p-1 or index==p-2) and plot_sep=="Yes":
            t_line=np.linspace(0,t,len(Llist))
            plt.title("Prikaz spremembe populacije za $Z_0$ = "+str(Z0)+r'$, L_0 = $'+str(L0)+r', $\alpha$ = '+str(a)+r', $\beta = $'+str(b))
            plt.plot(t_line, Llist, color = plt.cm.Reds(index/p), label="Lisice $%i$" %(index-1))
            plt.plot(t_line, Zlist, color = plt.cm.Blues(index/p), label="Zajci $%i$"%(index-1))
            plt.axvline(x = t_line[len(t_line)-1], color = "black", label = 'Smrt populacije $%i$ t=$%.2f$' %(index-1,t_line[len(t_line)-1]))
            plt.ylabel('L')
            plt.xlabel('t')
        tlist.append(t)
        tavr[j] = tavr[j] + t
        index+=1
    t_graph.append(sum(tlist)/len(tlist))
    jndex+=1
    if plt_his=="Yes" and len(tlist)>1:
        plt.title("Prikaz histogramov za $Z_0$ = "+str(Z0)+r'$, L_0 = $'+str(L0)+r', $\alpha$ = '+str(a)+r', $\beta = $'+str(b))
        plt.hist(tlist,bins = int(np.amax(tlist)), label = r'$dt= $'+ str(dt), color = plt.cm.rainbow(jndex/len(dtlist)),edgecolor='black',alpha = 0.8)
        plt.ylabel(r'$t_{death}$')
        plt.xlabel('N')
    tavr[j] = tavr[j]/p

if plt_time=="Yes":
    plt.title("Prikaz odvisnosti časa od koraka za $Z_0$ = "+str(Z0)+r'$, L_0 = $'+str(L0)+r', $\alpha$ = '+str(a)+r', $\beta = $'+str(b))
    plt.plot(dtlist, t_graph)
    plt.ylabel(r'$t_{death}$')
    plt.xlabel('dt')
    plt.xscale("log")
plt.grid()   
plt.legend()
plt.show()

index=0
jndex=0
plot_sep="No"
plt_his="Yes"
for a in tqdm(a_list):
    dt = 0.001
    tlist=[]
    for i in range(p):
        Z,L,t,Zlist,Llist=model(a,b,dt,Z0,L0)
        
        if p < 5 and plot_sep=="No":
            plt.title("Prikaz faznega diagrama zajcev in lisic za $Z_0$ = "+str(Z0)+r'$, L_0 = $'+str(L0)+r', $\alpha$ = '+str(a)+r', $\beta = $'+str(b))
            plt.plot(Zlist, Llist, color = plt.cm.rainbow(index/p), label="Ponovitev "+str(i+1))
            plt.xlabel('Z')
            plt.ylabel('L')
        if (index==p-1 or index==p-2) and plot_sep=="Yes":
            t_line=np.linspace(0,t,len(Llist))
            plt.title("Prikaz spremembe populacije za $Z_0$ = "+str(Z0)+r'$, L_0 = $'+str(L0)+r', $\alpha$ = '+str(a)+r', $\beta = $'+str(b))
            plt.plot(t_line, Llist, color = plt.cm.Reds(index/p), label="Lisice $%i$" %(index-1))
            plt.plot(t_line, Zlist, color = plt.cm.Blues(index/p), label="Zajci $%i$"%(index-1))
            plt.axvline(x = t_line[len(t_line)-1], color = "black", label = 'Smrt populacije $%i$ t=$%.2f$' %(index-1,t_line[len(t_line)-1]))
            plt.ylabel('L')
            plt.xlabel('t')
        tlist.append(t)
        index+=1
    jndex+=1
    if plt_his=="Yes" and len(tlist)>1:
        plt.title("Prikaz histogramov za $Z_0$ = "+str(Z0)+r'$, L_0 = $'+str(L0)+r', $dt$ = '+str(dt)+r', $\beta = $'+str(b))
        plt.hist(tlist,bins = int(np.amax(tlist)), label = r'$\alpha= %.2f$'%(a), color = plt.cm.rainbow(jndex/len(a_list)),edgecolor='black',alpha = 0.8)
        #plt.axvline(x = sum(tlist)/len(tlist),color="black")
        plt.xlabel(r'$t_{death}$')
        plt.ylabel('N')
plt.grid()   
plt.legend()
plt.show()   
  

#plt.plot(dtlist,tavr)
#
#plt.xscale('log')
#plt.xlabel(r'$dt$')
#plt.ylabel(r'$t_{izumrtje}$')
#plt.grid()
#plt.show()



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




if p < 5:
    plt.title(r'$Z_0$ = '+str(Z0)+r'$, L_0 = $'+str(L0)+r', $\alpha$ = '+str(a)+r', $\beta = $'+str(b)+', ponovitve = '+str(p))
    plt.xlabel('Z')
    plt.ylabel('L')
    plt.grid()
    plt.show()



plt.xlabel(r'$t_{izumrtje}$')
plt.ylabel('N')
plt.grid()
plt.show()
