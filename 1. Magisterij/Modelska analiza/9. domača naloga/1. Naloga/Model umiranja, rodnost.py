import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  
from numba import jit,njit
from numba.typed import List
import os
from numpy import random
from matplotlib import cm
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import norm

N0 = 25    #začetna velikost populacije
t = 0           #začetni čas

b = 1           #beta?
blist = [1]

dt = 0.1        #časovni interval
dtlist = np.linspace(0.001,1.5,100)

p = 3         #število runov

pl = 1   #kaj plotamo
#0 ... N(t)
#1 ... P(tiz)



colors = ['r','b','g','m','c']  #pač barve

N = N0 
@jit(nopython=True)
def model(N_0, dt,b):
    bs = 5*b  
    br = 4*b     
    N = N_0
    t = 0
    Nlist=[]
    tlist=[]
    while N>0:
        Nlist.append(N)
        tlist.append(t)
        dN = -random.poisson(lam=bs*N*dt)+ random.poisson(lam=br*N*dt)
        N = N + dN
        t = t + dt
    return Nlist, tlist
try_list=1000
t_max=0
t_max2=0
plot=1
t_death=[]
fig, ax=plt.subplots(1)
fig1, ax1=plt.subplots(1)
index=0
for i in tqdm(range(try_list)):
    Nlist, tlist=model(N,dt,b)
    color = cm.rainbow(np.linspace(0, 1, len(Nlist)))
    if tlist[len(tlist)-1]>t_max:
        t_max=tlist[len(tlist)-1]
    #plt.scatter(tlist,Nlist, cmap=plt.cm.rainbow, alpha=0.05)
    t_death.append(tlist[len(tlist)-1])
    ax1.plot(tlist,Nlist,".-", color="blue", alpha=0.005)
    if i%int(try_list/4)==0:
        if tlist[len(tlist)-1]>t_max2:
            t_max2=tlist[len(tlist)-1]
        ax.plot(tlist,Nlist,".-",label="Model $%i$" %(index+1), color=plt.cm.rainbow(index/3))
        index+=1

x = np.linspace(0, t_max, 100)
x2 = np.linspace(0, t_max2, 100)
ax.set_title("Prikaz populacijskega modela pri $N=%i$" %(N))
ax.plot(x2,N0*np.exp(-b*x2),color="black", label=r"$N_0e^{-\beta t}$")
ax.legend()
ax.set_xlabel("t[s]")
ax.set_ylabel("N")

ax1.set_title("Prikaz več generiranih populacijskih modelov")
ax1.plot(x,N0*np.exp(-b*x),color="black", label=r"$N_0e^{-\beta t}$")
ax1.set_ylim([0.5,N0+100])
ax1.set_yscale("log")
ax1.legend()
ax1.set_xlabel("t[s]")
ax1.set_ylabel("N")
plt.show()
plt.hist(t_death,bins = int(np.amax(t_death)*2), label = r'$\beta = $'+ str(b), color = "red",alpha = 0.6)
plt.show()
plt.show()
plt.hist(t_death,bins = int(np.amax(t_death)*2), label = r'$\beta = $'+ str(b), color = "red",alpha = 0.6)
plt.show()
from scipy.optimize import curve_fit
from math import factorial
def poisson(t_poiss, rate, scale): #scale is added here so the y-axis 
# of the fit fits the height of histogram
    return (scale*(rate**t_poiss/factorial(t_poiss))*np.exp(-rate))
b_list=[0.1,0.2,0.5,1,1.5]
index=0
colors=["blue", "red", "green"]
try_list=10000
for b_1 in b_list:
    t_death=[]
    for i in tqdm(range(try_list)):
        Nlist, tlist=model(N,dt,b_1)
        color = cm.rainbow(np.linspace(0, 1, len(Nlist)))
        t_death.append(tlist[len(tlist)-1])
    n, bins, patches =plt.hist(t_death,bins = int(np.amax(t_death)*2), label = r'$\beta = $'+ str(b_1), color = plt.cm.rainbow(index/len(b_list)),alpha = 0.8)
    index+=1
plt.title(r"Prikaz histograma smrti modela za različne $\beta$ pri $N=%i$"%(N))
plt.xlabel(r"$t_{death}$")
plt.xlim([-5.5,125])
plt.legend()
plt.show()


N0=250
N=N0
index=0
fig, ax=plt.subplots(1)
fig1, ax1=plt.subplots(1)
for i in tqdm(range(try_list)):
    Nlist, tlist=model(N,dt,b)
    color = cm.rainbow(np.linspace(0, 1, len(Nlist)))
    if tlist[len(tlist)-1]>t_max:
        t_max=tlist[len(tlist)-1]
    #plt.scatter(tlist,Nlist, cmap=plt.cm.rainbow, alpha=0.05)
    t_death.append(tlist[len(tlist)-1])
    ax1.plot(tlist,Nlist,".-", color="blue", alpha=0.005)
    if i%int(try_list/4)==0:
        if tlist[len(tlist)-1]>t_max2:
            t_max2=tlist[len(tlist)-1]
        ax.plot(tlist,Nlist,".-",label="Model $%i$" %(index+1), color=plt.cm.rainbow(index/3))
        index+=1

x = np.linspace(0, t_max, 100)
x2 = np.linspace(0, t_max2, 100)
ax.set_title("Prikaz populacijskega modela pri $N=%i$" %(N))
ax.plot(x2,N0*np.exp(-b*x2),color="black", label=r"$N_0e^{-\beta t}$")
ax.legend()
ax.set_xlabel("t[s]")
ax.set_ylabel("N")

ax1.set_title("Prikaz več generiranih populacijskih modelov")
ax1.plot(x,N0*np.exp(-b*x),color="black", label=r"$N_0e^{-\beta t}$")
ax1.set_ylim([0.5,N0+100])
ax1.set_yscale("log")
ax1.legend()
ax1.set_xlabel("t[s]")
ax1.set_ylabel("N")
plt.show()
plt.hist(t_death,bins = int(np.amax(t_death)*2), label = r'$\beta = $'+ str(b), color = "red",alpha = 0.6)
plt.show()
plt.show()
plt.hist(t_death,bins = int(np.amax(t_death)*2), label = r'$\beta = $'+ str(b), color = "red",alpha = 0.6)
plt.show()
from scipy.optimize import curve_fit
from math import factorial
def poisson(t_poiss, rate, scale): #scale is added here so the y-axis 
# of the fit fits the height of histogram
    return (scale*(rate**t_poiss/factorial(t_poiss))*np.exp(-rate))
b_list=[0.1,0.2,0.5,1,1.5]
index=0
colors=["blue", "red", "green"]
try_list=10000
for b_1 in b_list:
    t_death=[]
    for i in tqdm(range(try_list)):
        Nlist, tlist=model(N,dt,b_1)
        color = cm.rainbow(np.linspace(0, 1, len(Nlist)))
        t_death.append(tlist[len(tlist)-1])
    n, bins, patches =plt.hist(t_death,bins = int(np.amax(t_death)*2), label = r'$\beta = $'+ str(b_1), color = plt.cm.rainbow(index/len(b_list)),alpha = 0.8)
    index+=1
plt.title(r"Prikaz histograma smrti modela za različne $\beta$ pri $N=%i$"%(N))
plt.xlabel(r"$t_{death}$")
plt.xlim([-5.5,125])
plt.legend()
plt.show()
dtlist = np.linspace(0.1,10,100)
N=250
b_list=[0.2,0.3,0.5,1]
b_list=np.linspace(0.1,1,8)
index=0
colors=["blue", "red", "green"]
try_list=1000
for b_1 in tqdm(b_list):
    t_death_avg=[]
    for dt in tqdm(dtlist):
        t_death=[]
        for i in range(try_list):
            Nlist, tlist=model(N,dt,b_1)
            color = cm.rainbow(np.linspace(0, 1, len(Nlist)))
            if max(tlist)==0:
                t_death.append(dt)
            else:
                t_death.append(max(tlist))
        t_death_avg.append(sum(t_death)/len(t_death))

    sigma=sum(t_death_avg)/len(t_death_avg)-min(t_death_avg)
    maxi=sum(t_death_avg)/len(t_death_avg)-max(t_death_avg)
    if sigma<maxi:
        sigma=maxi
    sigma_list_plus=[]
    sigma_list_minus=[]
    for i in range(len(t_death_avg)):
        sigma_list_plus.append(t_death_avg[i]+sigma)
        sigma_list_minus.append(t_death_avg[i]-sigma)
    plt.plot(dtlist, t_death_avg, color = plt.cm.rainbow(index/len(b_list)), label = r'$\beta = %.2f$' %(b_1))
    #plt.fill_between(dtlist, sigma_list_minus ,sigma_list_plus, color = plt.cm.rainbow(index/len(b_list)), alpha = 0.5)
    index+=1
plt.title(r"Prikaz povprečnega časa izumrtja od časovnega koraka za $N=%i$"%(N))
plt.xlabel(r"$\Delta t$")
plt.ylabel(r"$\overline{t_{death}}$")
plt.legend()
plt.show()

N=25

index=0
colors=["blue", "red", "green"]
try_list=1000
for b_1 in tqdm(b_list):
    t_death_avg=[]
    for dt in tqdm(dtlist):
        t_death=[]
        for i in range(try_list):
            Nlist, tlist=model(N,dt,b_1)
            color = cm.rainbow(np.linspace(0, 1, len(Nlist)))
            if max(tlist)==0:
                t_death.append(dt)
            else:
                t_death.append(max(tlist))
        t_death_avg.append(sum(t_death)/len(t_death))
    sigma=sum(t_death_avg)/len(t_death_avg)-min(t_death_avg)
    maxi=sum(t_death_avg)/len(t_death_avg)-max(t_death_avg)
    if sigma<maxi:
        sigma=maxi
    sigma_list_plus=[]
    sigma_list_minus=[]
    for i in range(len(t_death_avg)):
        sigma_list_plus.append(t_death_avg[i]+sigma)
        sigma_list_minus.append(t_death_avg[i]-sigma)
    plt.plot(dtlist, t_death_avg, color = plt.cm.rainbow(index/len(b_list)), label = r'$\beta = %.2f$' %(b_1))
    #plt.fill_between(dtlist, sigma_list_minus ,sigma_list_plus, color = plt.cm.rainbow(index/len(b_list)), alpha = 0.5)
    index+=1
plt.title(r"Prikaz povprečnega časa izumrtja od časovnega koraka za $N=%i$"%(N))
plt.xlabel(r"$\Delta t$")
plt.ylabel(r"$\overline{t_{death}}$")
plt.legend()
plt.show()