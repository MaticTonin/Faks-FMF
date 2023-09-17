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
from tqdm import tqdm
import func

N=10000
T=200
t=np.linspace(0,T,N)
p02=[1,0]
q02=[0,0.5]
p03=[1,0]
q03=[0,0.5]
lam=1
p_2matrix=[[1,0]]
q_2matrix=[[0,0.5]]

p_3matrix=[[1,0]]
q_3matrix=[[0,0.5]]
tau=t[1]-t[0]
z0=[1,0,0,0.5]
init_state=np.array([0,0.5,1,0])
t_fin=T
N_store=100

fig=plt.figure()
fig.suptitle(r"Prikaz faznega diagrama za $p$ z $t=%i$ in $\tau=%.4f$" %(T,tau))

fig1=plt.figure()
fig1.suptitle(r"Prikaz spreminjanja vektorje $q,p$ z $t=%i$ in $\tau=%.4f$" %(T,tau))
lambda_list=[0,0.1,0.5,1]
i=1
for lam in tqdm(lambda_list):
    ax = fig.add_subplot(2,2,i)
    ax.set_title(r"$\lambda=%.2f$" %(lam))
    ax1 = fig1.add_subplot(2,2,i)
    ax1.set_title(r"$\lambda=%.2f$" %(lam))
    q0,q1,p0,p1, Energy=func.dop853(init_state,lam,t_fin,N)
    ax.plot(p0,p1,"-", label="RK Order 8")
    ax1.plot(t,p0,"-", label="$p_1$")
    ax1.plot(t,p1,"-", label="$p_2$")
    ax1.plot(t,q0,"-", label="$q_1$")
    ax1.plot(t,q1,"-", label="$q_2$")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_2,p0,q0)
    ax.plot(p_matrix.T[0],p_matrix.T[1],"--", label="$S_2$")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_3,p0,q0)
    ax.plot(p_matrix.T[0],p_matrix.T[1],"--", label="$S_3$")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_4,p0,q0)
    ax.plot(p_matrix.T[0],p_matrix.T[1],"-", label="$S_4$")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_5,p0,q0)
    ax.plot(p_matrix.T[0],p_matrix.T[1],"-", label="$S_5$")
    if i==1:
        ax.legend()
    ax1.legend()
    ax.set_xlabel("$p_1$")
    ax.set_ylabel("$p_2$")
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$q,p$")
    i+=1

plt.show()

divergence="NO"
if divergence=="Yes":
    lam=1
    init_state=np.array([0,0.5,1,0])
    tau_list=np.logspace(-4,0,100)
    t_fin=100
    i=1
    Energy_S2=[]
    Energy_S3=[]
    Energy_S4=[]
    Energy_S4=[]
    Energy_S5=[]
    Energy_RK=[]
    T=100
    from tqdm import tqdm
    for tau in tqdm(tau_list):
        N=int(T/tau)
        t=np.linspace(0,T,N)
        q0,q1,p0,p1, Energy=func.dop853(init_state,lam,t_fin,N)
        Energy_RK.append(abs(Energy[N-1]-Energy[0])/Energy[0])
        p0=[1,0]
        q0=[0,0.5]
        p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_2,p0,q0)
        Energy_S2.append(abs(Energy[N-1]-Energy[0])/Energy[0])
        p0=[1,0]
        q0=[0,0.5]
        p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_3,p0,q0)
        Energy_S3.append(abs(Energy[N-1]-Energy[0])/Energy[0])
        p0=[1,0]
        q0=[0,0.5]
        p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_4,p0,q0)
        Energy_S4.append(abs(Energy[N-1]-Energy[0])/Energy[0])
        p0=[1,0]
        q0=[0,0.5]
        p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_5,p0,q0)
        Energy_S5.append(abs(Energy[N-1]-Energy[0])/Energy[0])


    Energy_RK=np.array(Energy_RK)
    Energy_S2=np.array(Energy_S2)
    Energy_S3=np.array(Energy_S3)
    Energy_S4=np.array(Energy_S4)
    Energy_S5=np.array(Energy_S5)
    plt.title(r"Prikaz odvisnosti relativne divergence energij od izbire $\tau$, $t=%.2f$" %T)
    plt.plot(tau_list, Energy_RK, "-", alpha=0.7, label="RK Order 8")
    plt.plot(tau_list, Energy_S2, "-", alpha=0.7, label="$S_2$")
    plt.plot(tau_list, Energy_S3, "-", alpha=0.7, label="$S_3$")
    plt.plot(tau_list, Energy_S4, "-", alpha=0.7, label="$S_4$")
    plt.plot(tau_list, Energy_S5, "-", alpha=0.7, label="$S_5$")
    plt.xlabel(r"$\tau$",fontsize=20)
    plt.ylabel(r"$\frac{|E-E_{0}|}{E_{0}}$",fontsize=20)
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.show()

time_divergende="NO"
epsilon=0.01
if time_divergende=="Yes":
    lam=1
    init_state=np.array([0,0.5,1,0])
    tau_list=np.logspace(-2,0,500)
    t_fin=T
    i=1
    Energy_S2=[]
    Energy_S3=[]
    Energy_S4=[]
    Energy_S4=[]
    Energy_S5=[]
    Energy_RK=[]
    T=1000
    lambda_list=[0,0.1,0.5,1]
    from tqdm import tqdm
    fig=plt.figure()
    fig.suptitle(r"Prikaz čas divergence za $\epsilon=%.3f$ od izbire $\tau$, $t=%.2f$" %(epsilon,T))
    for lam in tqdm(lambda_list):
        init_state=np.array([0,0.5,1,0])
        Energy_S2=[]
        Energy_S3=[]
        Energy_S4=[]
        Energy_S4=[]
        Energy_S5=[]
        Energy_RK=[]
        for tau in (tau_list):
            N=int(T/tau)
            t=np.linspace(0,T,N)
            p0=[1,0]
            q0=[0,0.5]
            Energy= func.time_dependance(N,tau,lam,func.S_2,p0,q0,epsilon)
            Energy_S2.append(Energy)
            p0=[1,0]
            q0=[0,0.5]
            Energy= func.time_dependance(N,tau,lam,func.S_3,p0,q0,epsilon)
            Energy_S3.append(Energy)
            p0=[1,0]
            q0=[0,0.5]
            Energy= func.time_dependance(N,tau,lam,func.S_4,p0,q0, epsilon)
            Energy_S4.append(Energy)
            p0=[1,0]
            q0=[0,0.5]
            Energy= func.time_dependance(N,tau,lam,func.S_5,p0,q0,epsilon)
            Energy_S5.append(Energy)

        ax = fig.add_subplot(2,2,i)
        Energy_S2=np.array(Energy_S2)
        Energy_S3=np.array(Energy_S3)
        Energy_S4=np.array(Energy_S4)
        Energy_S5=np.array(Energy_S5)
        ax.set_title(r"$\lambda=%.2f$" %(lam))
        ax.plot(tau_list, Energy_S2, "-", alpha=0.7, label="$S_2$")
        ax.plot(tau_list, Energy_S3, "-", alpha=0.7, label="$S_3$")
        ax.plot(tau_list, Energy_S4, "-", alpha=0.7, label="$S_4$")
        ax.plot(tau_list, Energy_S5, "-", alpha=0.7, label="$S_5$")
        ax.set_xlabel(r"$\tau$",fontsize=12)
        ax.set_ylabel(r"$T$",fontsize=12)
        ax.legend()
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid()
        i+=1
    plt.show()


N=10000
T=20
t=np.linspace(0,T,N)
p02=[1,0]
q02=[0,0.5]
p03=[1,0]
q03=[0,0.5]
lam=1
p_2matrix=[[1,0]]
q_2matrix=[[0,0.5]]

p_3matrix=[[1,0]]
q_3matrix=[[0,0.5]]
tau=t[1]-t[0]
z0=[1,0,0,0.5]
init_state=np.array([0,0.5,1,0])
t_fin=T
N_store=100

plt_p2="Yes"
if plt_p2=="Yes":
    fig=plt.figure()
    fig.suptitle(r"Prikaz povprečja $\langle p_1^2 \rangle$  z $t=%i$ in $\tau=%.4f$" %(T,tau))
    fig2=plt.figure()
    fig2.suptitle(r"Prikaz povprečja $\langle p_2^2 \rangle$  z $t=%i$ in $\tau=%.4f$" %(T,tau))
    fig3, ax3=plt.subplots()
    fig3.suptitle(r"Prikaz povprečja $\langle p_j^2 \rangle$  z $t=%i$ in $\tau=%.4f$" %(T,tau))

    fig4, ax4=plt.subplots()
    fig4.suptitle(r"Prikaz vsote $\sum_j\langle p_j^2 \rangle$  z $t=%i$ in $\tau=%.4f$" %(T,tau))
    lambda_list=[0,0.5,1,5]
    i=1
    from tqdm import tqdm
    p1_,p2_=[],[]
    for lam in tqdm(lambda_list):
        ax = fig.add_subplot(2,2,i)
        ax.set_title(r"$\lambda=%.2f$" %(lam))
        ax1 = fig2.add_subplot(2,2,i)
        ax1.set_title(r"$\lambda=%.2f$" %(lam))
        #t, p1_avg,p2_avg=func.dop853(init_state,lam,t_fin,N)
        #ax.plot(q0,q1,"-", label="RK Order 8")
        p0=[1,0]
        q0=[0,0.5]
        t, p1_avg,p2_avg= func.average_value(N,tau,lam,func.S_2,p0,q0,t)
        ax.plot(t,p1_avg,"--", label="$S_2$")
        ax1.plot(t,p2_avg,"--", label="$S_2$")
        ax4.plot(t,np.array(p1_avg)+np.array(p2_avg),"--", label="$S_2$")
        p0=[1,0]
        q0=[0,0.5]
        t, p1_avg,p2_avg= func.average_value(N,tau,lam,func.S_3,p0,q0,t)
        ax.plot(t,p1_avg,"--", label="$S_3$")
        ax1.plot(t,p2_avg,"--", label="$S_3$")
        ax4.plot(t,np.array(p1_avg)+np.array(p2_avg),"--", label="$S_3$")
        p0=[1,0]
        q0=[0,0.5]
        t, p1_avg,p2_avg= func.average_value(N,tau,lam,func.S_4,p0,q0,t)
        ax.plot(t,p1_avg,"-", label="$S_4$")
        ax1.plot(t,p2_avg,"-", label="$S_4$")
        ax4.plot(t,np.array(p1_avg)+np.array(p2_avg),"--", label="$S_4$")
        p0=[1,0]
        q0=[0,0.5]
        t, p1_avg,p2_avg= func.average_value(N,tau,lam,func.S_5,p0,q0,t)
        ax.plot(t,p1_avg,"-", label="$S_5$")
        ax1.plot(t,p2_avg,"-", label="$S_5$")
        ax4.plot(t,np.array(p1_avg)+np.array(p2_avg),"--", label="$S_5$")
        ax3.plot(t,p1_avg,"-", label=r"$p_1$, $\lambda=%.2f$" %(lam), color=plt.cm.autumn((i-1)/len(lambda_list)))
        ax3.plot(t,p2_avg,"-", label=r"$p_2$, $\lambda=%.2f$" %(lam), color=plt.cm.winter((i-1)/len(lambda_list)))
        if i==1:
            ax.legend()
            ax1.legend()
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$\langle p_1^2\rangle$")
        ax1.set_xlabel("$t$")
        ax1.set_ylabel(r"$\langle p_2^2\rangle$")
        i+=1
    ax3.legend()
    ax3.set_xlabel("$t$")
    ax3.set_ylabel(r"$\langle p_j^2\rangle$")
    ax4.legend()
    ax4.set_xlabel("$t$")
    ax4.set_ylabel(r"$\sum_j\langle p_j^2\rangle$")
    plt.show()


plt_p2_pl1="No"
if plt_p2_pl1=="Yes":
    N=100000
    T=10000
    t=np.linspace(0,T,N)
    tau=t[1]-t[0]
    fig, ax=plt.subplots()
    fig.suptitle(r"Prikaz razmerje povprečja $\langle p_2^2 \rangle/\langle p_1^2 \rangle$  z $t=%i$ in $\tau=%.4f$" %(T,tau))
    init_state=np.array([0,0.5,1,0])
    t_fin=T
    lambda_list=[0,0.5,1,2,3,4,5,10]
    i=1
    from tqdm import tqdm
    p1_,p2_=[],[]
    for lam in tqdm(lambda_list):
        #t, p1_avg,p2_avg=func.dop853(init_state,lam,t_fin,N)
        #ax.plot(q0,q1,"-", label="RK Order 8")
        p0=[1,0]
        q0=[0,0.5]
        t, p1_avg,p2_avg= func.average_value_RK(init_state,lam,t_fin,N,t)
        ax.plot(t,np.array(p2_avg)/np.array(p1_avg),"-", label=r"$\lambda=%.2f$" %(lam), color=plt.cm.autumn((i-1)/len(lambda_list)))
        i+=1
    ax.legend()
    ax.set_ylim(-0.5,3)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\langle p_2^2 \rangle/\langle p_1^2 \rangle$")
    plt.show()
import random
plt_p2_pl1_lam="No"
if plt_p2_pl1_lam=="Yes":
    from scipy.stats import norm
    import matplotlib.mlab as mlab
    #sortiraj po N
    N1=1000
    N2=10000
    N=1000
    T=1000
    t=np.linspace(0,T,N)
    t1=np.linspace(0,T,N1)
    t2=np.linspace(0,T,N2)
    tau=t[1]-t[0]
    fig = plt.figure()
    fig.suptitle(r"Prikaz razmerje povprečja $\langle p_2^2 \rangle/\langle p_1^2 \rangle$  z $t=%i$" %(T))
    ax = fig.add_subplot(1,2,1)
    ax1 = fig.add_subplot(1,2,2)
    init_state=np.array([0,0.5,1,0])
    t_fin=T
    lambda_list=np.linspace(2,100,250)
    i=1
    from tqdm import tqdm
    p1_,p2_=[],[]
    p1_1,p2_1=[],[]
    p1_2,p2_2=[],[]
    lambda_list_whole=[]
    lambda_list_whole1=[]
    lambda_list_whole2=[]

    for lam in tqdm(lambda_list):
        #t, p1_avg,p2_avg=func.dop853(init_state,lam,t_fin,N)
        #ax.plot(q0,q1,"-", label="RK Order 8")
        p0=[1,0]
        q0=[0,0.5]
        t, p1_avg,p2_avg= func.average_value_RK(init_state,lam,t_fin,N,t)
        t1, p1_avg1,p2_avg1= func.average_value_RK(init_state,lam,t_fin,N1,t1)
        t2, p1_avg2,p2_avg2= func.average_value_RK(init_state,lam,t_fin,N2,t2)
        p1_.append(p1_avg[N-1])
        p2_.append(p2_avg[N-1])
        p1_1.append(p1_avg1[N1-1])
        p2_1.append(p2_avg1[N1-1])
        p1_2.append(p1_avg2[N2-1])
        p2_2.append(p2_avg2[N2-1])
        lambda_list_whole.append(lam)
        i+=1
    (mu, sigma) = norm.fit(np.array(p2_)/np.array(p1_))
    ax.scatter(lambda_list_whole,np.array(p2_)/np.array(p1_),color="blue",label="N=%i" %N)
    n, bins, patches = ax1.hist(np.array(p2_)/np.array(p1_),bins=30,alpha=0.5 ,color="blue", orientation="horizontal", density=True)
    y = scipy.stats.norm.pdf( bins, mu, sigma)
    ax1.plot(y, bins, 'r--', label=r"$\mu=%.2f, \sigma=%.3f$" %(mu,sigma),linewidth=2,color="blue")
    (mu, sigma) = norm.fit(np.array(p2_1)/np.array(p1_1))
    ax.scatter(lambda_list_whole,np.array(p2_1)/np.array(p1_1),color="green", label="N=%i" %N1)
    n, bins, patches = ax1.hist(np.array(p2_1)/np.array(p1_1),bins=30,alpha=0.5 ,color="green", orientation="horizontal", density=True)
    y = scipy.stats.norm.pdf( bins, mu, sigma)
    ax1.plot(y, bins, 'r--', label=r"$\mu=%.2f, \sigma=%.3f$" %(mu,sigma),linewidth=2,color="green")

    (mu, sigma) = norm.fit(np.array(p2_2)/np.array(p1_2))
    ax.scatter(lambda_list_whole,np.array(p2_2)/np.array(p1_2),color="red", label="N=%i" %N2)
    n, bins, patches = ax1.hist(np.array(p2_2)/np.array(p1_2),bins=30,alpha=0.5 ,color="red", orientation="horizontal", density=True)
    y = scipy.stats.norm.pdf( bins, mu, sigma)
    ax1.plot(y, bins, 'r--', label=r"$\mu=%.2f, \sigma=%.3f$" %(mu,sigma),linewidth=2,color="red")
    #ax.plot(t,np.array(p2_avg)/np.array(p1_avg),"-", label=r"$\lambda=%.2f$" %(lam), color=plt.cm.autumn((i-1)/len(lambda_list)))
    ax.axhline(1, 0, T,ls="--", color="black", alpha=0.7)
    ax1.axhline(1, 0, T,ls="--", color="black", alpha=0.7)
    ax.legend()
    ax1.legend()
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\langle p_2^2 \rangle/\langle p_1^2 \rangle$")
    ax1.set_ylabel(r"$\langle p_2^2 \rangle/\langle p_1^2 \rangle$")
    ax1.set_xlabel(r"$N$")
    ax.grid()
    ax1.grid()
    plt.show()



N=10000
T=200
t=np.linspace(0,T,N)
p02=[1,0]
q02=[0,0.5]
p03=[1,0]
q03=[0,0.5]
lam=1
p_2matrix=[[1,0]]
q_2matrix=[[0,0.5]]

p_3matrix=[[1,0]]
q_3matrix=[[0,0.5]]
tau=t[1]-t[0]
z0=[1,0,0,0.5]
init_state=np.array([0,0.5,1,0])
t_fin=T
N_store=100

fig=plt.figure()
fig.suptitle(r"Prikaz faznega diagrama za $q$ z $t=%i$ in $\tau=%.4f$" %(T,tau))
lambda_list=[0,0.1,0.5,1]
i=1
for lam in tqdm(lambda_list):
    ax = fig.add_subplot(2,2,i)
    ax.set_title(r"$\lambda=%.2f$" %(lam))
    q0,q1,p0,p1, Energy=func.dop853(init_state,lam,t_fin,N)
    ax.plot(q0,q1,"-", label="RK Order 8")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_2,p0,q0)
    ax.plot(q_matrix.T[0],q_matrix.T[1],"--", label="$S_2$")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_3,p0,q0)
    ax.plot(q_matrix.T[0],q_matrix.T[1],"--", label="$S_3$")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_4,p0,q0)
    ax.plot(q_matrix.T[0],q_matrix.T[1],"-", label="$S_4$")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_5,p0,q0)
    ax.plot(q_matrix.T[0],q_matrix.T[1],"-", label="$S_5$")
    if i==1:
        ax.legend()
    ax.set_xlabel("$q_1$")
    ax.set_ylabel("$q_2$")
    i+=1
plt.show()


z0=[1,0,0,0.5]
init_state=np.array([0,0.5,1,0])
t_fin=T


fig=plt.figure()
fig.suptitle(r"Prikaz energije z $t=%i$ in $\tau=%.4f$" %(T,tau))
lambda_list=[0,0.1,0.5,1]
i=1
for lam in lambda_list:
    ax = fig.add_subplot(2,2,i)
    q0,q1,p0,p1, Energy0=func.dop853(init_state,lam,t_fin,N)
    ax.set_title(r"$\lambda=%.2f$" %(lam))
    lambda_list=[0,0.1,0.5,1]
    ax.plot(t,Energy,"-",color="black",label="RK Order 8")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_2,p0,q0)
    ax.plot(t,Energy,"--", alpha=0.5,label="$S_2$")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_3,p0,q0)
    ax.plot(t,Energy,"--", alpha=0.5,label="$S_3$")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_4,p0,q0)
    ax.plot(t,Energy,"--",alpha=0.5,label="$S_4$" )
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_5,p0,q0)
    ax.plot(t,Energy,"--",alpha=0.5, label="$S_5$")
    ax.set_xlabel("t")
    ax.set_ylabel("E")
    if i==1:
        ax.legend()
    ax.set_yscale("log")
    i+=1
plt.show()
fig=plt.figure()
fig.suptitle(r"Prikaz relative razlike energije od RK8 z $t=%i$ in $\tau=%.4f$" %(T,tau))
lambda_list=[0,0.1,0.5,1]
i=1
for lam in lambda_list:
    ax = fig.add_subplot(2,2,i)
    q0,q1,p0,p1, Energy0=func.dop853(init_state,lam,t_fin,N)

    p0=[1,0]
    q0=[0,0.5]
    ax.set_title(r"$\lambda=%.2f$" %(lam))
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_2,p0,q0)
    ax.plot(t,abs(Energy-Energy0)/Energy0,"-",alpha=0.7, label="$S_2$")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_3,p0,q0)
    ax.plot(t,abs(Energy-Energy0)/Energy0,"-",alpha=0.7, label="$S_3$")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_4,p0,q0)
    ax.plot(t,abs(Energy-Energy0)/Energy0,"-",alpha=0.7,label="$S_4$" )
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_5,p0,q0)
    ax.plot(t,abs(Energy-Energy0)/Energy0,"-",alpha=0.7, label="$S_5$")
    if i==1:
        ax.legend()
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\frac{|E-E_{RK}|}{E_{RK}}$", fontsize=18)
    ax.set_yscale("log")
    i+=1
plt.show()


fig=plt.figure()
fig.suptitle(r"Prikaz relative razlike energije od začetne $E_0$ z $t=%i$ in $\tau=%.4f$" %(T,tau))
lambda_list=[0,0.1,0.5,1]
i=1
for lam in lambda_list:
    ax = fig.add_subplot(2,2,i)
    q0,q1,p0,p1, Energy=func.dop853(init_state,lam,t_fin,N)
    ax.plot(t,abs(Energy-Energy[0])/Energy[0],"-",alpha=0.7, label="RK 8 Order")
    p0=[1,0]
    q0=[0,0.5]
    ax.set_title(r"$\lambda=%.2f$" %(lam))
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_2,p0,q0)
    ax.plot(t,abs(Energy-Energy[0])/Energy[0],"-",alpha=0.7, label="$S_2$")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_3,p0,q0)
    ax.plot(t,abs(Energy-Energy[0])/Energy[0],"-",alpha=0.7, label="$S_3$")
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_4,p0,q0)
    ax.plot(t,abs(Energy-Energy[0])/Energy[0],"-",alpha=0.7,label="$S_4$" )
    p0=[1,0]
    q0=[0,0.5]
    p_matrix,q_matrix,Energy= func.method(N,tau,lam,func.S_5,p0,q0)
    ax.plot(t,abs(Energy-Energy[0])/Energy[0],"-",alpha=0.7, label="$S_5$")
    if i==1:
        ax.legend()
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\frac{|E-E_{0}|}{E_{0}}$",fontsize=18)
    ax.set_yscale("log")
    i+=1
plt.show()