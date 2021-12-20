import numpy as np
import matplotlib.pyplot as plt
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file1 = os.path.join(THIS_FOLDER, "podatki.txt")


lam2, a0, a1, a2_2, I1= np.loadtxt (my_file1, skiprows=0, unpack=True)
tau=np.linspace(0,1,20) #center tau=1, rob tau=0
sev=[]
h=6.67e-27
c=3e10
k=1.38e-16

lam= [x*10**(-8) for x in lam2]
col=['salmon','sandybrown','gold','greenyellow','limegreen','cornflowerblue', 'indigo']
bbox_props = dict(boxstyle="rarrow,pad=0.2", fc="w", ec="k", lw=0.09)
for i in range(len(lam)):
    sev=[(a0[i]+a1[i]*x+a2_2[i]*x**2)*I1[i] for x in tau]
    T=[h*c/(lam[i]*k)*(np.log(1+(2*h*c**2/((lam[i]**5)*x))))**(-1) for x in sev]
    plt.scatter(tau, T, c=col[i], marker='.',label=str(int(lam2[i]))+' nm')
    
    print('$\\lambda=$',lam[i],'A', '&', '\\\\')
    for i in range(len(tau)):
        print('$',np.round(tau[i],2),'$','&','$',"{:.2e}".format(T[i]),'}$', '\\\\')


plt.xlabel('$\\tau_{\\lambda}$')
plt.ylabel('$T_{\\lambda}[K]$')
plt.title('$T_{\\lambda}(\\tau_{\\lambda})$')
plt.legend()
plt.show()

#data load and manipulation
lambd, a_0, a_1, a_2, I = np.loadtxt(my_file1, delimiter=" ", unpack="True")
N=500
tau=np.linspace(-1,0,N)
S=[]

for i in range(len(a_0)):
    S.append((a_0[i]+a_1[i]*tau+a_2[i]*tau**2)*I[i])
    #plt.plot(tau, S[i], '-', label=r"$\lambda=$"+str(lambd[i]))
    plt.plot(tau, S[i], '-', label=f'$\lambda$={lambd[i]/10} nm')
plt.xlabel(r"$\tau$")
plt.ylabel(r"$S_\lambda$")
plt.title(r"Odvisnost funckije izvora $S_\lambda$ od $\tau$")
plt.legend()
plt.show()

for i in range(len(a_0)):
    S.append((a_0[i]+a_1[i]*tau+a_2[i]*tau**2)*I[i])
    S_index=S[i]
    S_max=S_index[0]
    #plt.plot(tau, S[i], '-', label=r"$\lambda=$"+str(lambd[i]))
    plt.plot(lambd[i]/10, S_max, 'x', label=f'$\lambda$={lambd[i]/10} nm')
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$S_\lambda$")
plt.title(r"Odvisnost funckije izvora $S_\lambda$ na robu od $\lambda$")
plt.legend()
plt.show()

for i in range(len(a_0)):
    S.append((a_0[i]+a_1[i]*tau+a_2[i]*tau**2)*I[i])
    S_index=S[i]
    S_max=S_index[N-1]
    #plt.plot(tau, S[i], '-', label=r"$\lambda=$"+str(lambd[i]))
    plt.plot(lambd[i]/10, S_max, 'x', label=f'$\lambda$={lambd[i]/10} nm')
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$S_\lambda$")
plt.title(r"Odvisnost funckije izvora $S_\lambda$ v središču od $\lambda$")
plt.legend()
plt.show()

A=1.19268*10**(-22)
B=1.4404*10**(-5)
h = 6.62607015e-27
c = 2.99792458e10
k = 1.380649e-16
T=[]



for i in range(len(a_0)):
    val=lambd[i]*10**(-8)
    c1=h*c/(val*k)
    T.append(c1/np.log(1+2*h*c**2/(S[i]*val**(5))))
    T_index=T[i]
    T_max=T_index[N-1]
    plt.plot(lambd[i]/10, T_max, 'x', label=f'$\lambda$={lambd[i]/10} nm')
    #plt.plot(tau, T[i], '-', label=r"$\lambda=$"+str(lambd[i]))

plt.xlabel(r"$\lambda$")
plt.ylabel(r"T")
plt.legend()
plt.title(r"Temperatura v središču zvezde za določene $\lambda$")
plt.show()

for i in range(len(a_0)):
    val=lambd[i]*10**(-8)
    c1=h*c/(val*k)
    T.append(c1/np.log(1+2*h*c**2/(S[i]*val**(5))))
    T_index=T[i]
    T_max=T_index[0]
    plt.plot(lambd[i]/10, T_max, 'x', label=f'$\lambda$={lambd[i]/10} nm')
    #plt.plot(tau, T[i], '-', label=r"$\lambda=$"+str(lambd[i]))

plt.xlabel(r"$\lambda$")
plt.ylabel(r"T")
plt.legend()
plt.title(r"Temperatura na robu zvezde za določene $\lambda$")
plt.show()

for i in range(len(a_0)):
    val=lambd[i]*10**(-8)
    c1=h*c/(val*k)
    T.append(c1/np.log(1+2*h*c**2/(S[i]*val**(5))))
    plt.plot(tau, T[i], '-', label=f'$\lambda$={lambd[i]/10} nm')
    #plt.plot(tau, T[i], '-', label=r"$\lambda=$"+str(lambd[i]))

plt.xlabel(r"$\tau$")
plt.ylabel("T[K]")
plt.legend()
plt.title(r"Temperatura zvezde v odvisnosti od $\lambda$")
plt.show()

