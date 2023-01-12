import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D  

N = 10**4
a=1
Nivoji=19
cleni=17
kT=0.1

Energy="True"
def izdelava_verizce(Nivoji,N,a,kT,Energy,cleni):
    verizica=np.zeros(cleni)
    verizica[0]=0
    verizica[cleni-1]=0
    E = 0
    for i in range(1,16):
        verizica[i]=random.randint(0,int(Nivoji))*(-1)   
    for i in range(N):
        place=random.randint(1,15)
        delta_i=random.randrange(0,3,2)-1
        dE=(delta_i)**2-(delta_i)*(verizica[place+1]-2*verizica[place]+verizica[place-1]-a)
        if Energy == "True": #ali plota energijo verige
            E = 0
            for j in verizica:           
                E = E + a*j
            for k in range(0, len(verizica)-1):
                E = E + 1/2 *(verizica[k] - verizica[k+1])**2
        if verizica[place]==-Nivoji and delta_i==-1:
            continue
        if verizica[place]==0 and delta_i==1:
            continue
        if dE<=0:
            verizica[place]= verizica[place]+delta_i
        if dE>0:
            zeta=random.random()
            if zeta<=np.exp(-dE/kT):
                verizica[place]=verizica[place]+delta_i
    return verizica, E


N_list=np.linspace(1000,10**4,30)
Mulekle=[]
for i in range(cleni):
    Mulekle.append(i)
#N_list=np.logspace(0.1,4,500)
kT_list=[0.1,0.5,1,5,10]
kT_list=np.logspace(-1,1,10)
kT_list=np.logspace(-1,2,10)
kT_list=np.linspace(0.01,100,10)
Energy_matrix_kT=[]
index=0
verizica_matrix_kT=[]
fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)

fig3, ax3 = plt.subplots(1)
fig4, ax4 = plt.subplots(1)

fig5, ax5 = plt.subplots(1)
fig6, ax6 = plt.subplots(1)


fig7, ax7 = plt.subplots(1)
fig8, ax8 = plt.subplots(1)

for kT in tqdm(kT_list):
    Energy_N=[]
    verizica=[]
    jndex=0
    for N in N_list:
        verizica,energija=izdelava_verizce(Nivoji,int(N),a,kT,Energy,cleni)
        Energy_N.append(energija)
        if jndex==len(N_list)-1 and index%3==0:
            ax1.plot(Mulekle,verizica,".-",label=r"$k_bT=%.2f$" %(kT), color=plt.cm.rainbow(index/len(kT_list)))
        elif index%3==0: 
            ax1.plot(Mulekle,verizica,".-",alpha=0.04, color=plt.cm.rainbow(index/len(kT_list)))
        jndex+=1
    verizica_matrix_kT.append(np.array(verizica))
    Energy_matrix_kT.append(np.array(Energy_N))
    
    ax2.plot(N_list,Energy_N, label=r"$k_bT=%.2f$" %(kT), color=plt.cm.rainbow(index/len(kT_list)))
    index+=1

Energy_matrix_a=[]
index=0
verizica_matrix_a=[]
kT=1
a=1
for a in tqdm(kT_list):
    Energy_N=[]
    verizica=[]
    jndex=0
    for N in N_list:
        verizica,energija=izdelava_verizce(int(Nivoji),int(N),a,kT,Energy,cleni)
        Energy_N.append(energija)
        if jndex==len(N_list)-1 and index%3==0:
            ax3.plot(Mulekle,verizica,".-",label=r"$\alpha=%.2f$" %(a), color=plt.cm.rainbow(index/len(kT_list)))
        elif index%3==0: 
            ax3.plot(Mulekle,verizica,".-",alpha=0.04, color=plt.cm.rainbow(index/len(kT_list)))
        jndex+=1
    verizica_matrix_a.append(np.array(verizica))
    Energy_matrix_a.append(np.array(Energy_N))
    
    ax4.plot(N_list,Energy_N, label=r"$\alpha=%.2f$" %(a), color=plt.cm.rainbow(index/len(kT_list)))
    index+=1

Energy_matrix_a_kt=[]
index=0
verizica_matrix_a_kt=[]
kT=1
a=1
N=10**4
for a in tqdm(kT_list):
    Energy_N=[]
    verizica=[]
    jndex=0
    for kT in kT_list:
        verizica,energija=izdelava_verizce(Nivoji,int(N),a,kT,Energy,cleni)
        Energy_N.append(energija)
        if jndex==len(kT_list)-1 and index%3==0:
            ax5.plot(Mulekle,verizica,".-",label=r"$\alpha=%.2f$" %(kT), color=plt.cm.rainbow(index/len(kT_list)))
        elif index%3==0: 
            ax5.plot(Mulekle,verizica,".-",alpha=0.04, color=plt.cm.rainbow(index/len(kT_list)))
        jndex+=1
    verizica_matrix_a_kt.append(np.array(verizica))
    Energy_matrix_a_kt.append(np.array(Energy_N))
    
    ax6.plot(kT_list,Energy_N, label=r"$\alpha=%.2f$" %(a), color=plt.cm.rainbow(index/len(kT_list)))
    index+=1



Energy_matrix_Nivoji=[]
index=0
verizica_matrix_Nivoji=[]
Nivoji_list=np.linspace(20,50,30)
kT=0.1
a=5
for Nivoji in tqdm(Nivoji_list):
    Energy_N=[]
    verizica=[]
    jndex=0
    for N in N_list:
        verizica,energija=izdelava_verizce(Nivoji,int(N),a,kT,Energy,cleni)
        Energy_N.append(energija)
        if jndex==len(N_list)-1 and index%10==0:
            ax7.plot(Mulekle,verizica,".-",label=r"Nivoji=%i" %(Nivoji), color=plt.cm.rainbow(index/len(Nivoji_list)))
        elif index%10==0: 
            ax7.plot(Mulekle,verizica,".-",alpha=0.05, color=plt.cm.rainbow(index/len(Nivoji_list)))
        jndex+=1
    verizica_matrix_Nivoji.append(np.array(verizica))
    Energy_matrix_Nivoji.append(np.array(Energy_N))
    if index%5==0:
        ax8.plot(N_list,Energy_N, label=r"$Nivoji=%i$" %(Nivoji), color=plt.cm.rainbow(index/len(Nivoji_list)))
    index+=1

ax7.set_title(r"Prikaz verižice za različne Nivoje za $k_BT=%.2f$ in $\alpha=%.2f$" %(kT,a))
ax7.legend()
ax7.set_xlabel("Mulekule")
ax7.set_ylabel(r"h")

ax8.set_title(r"Prikaz minimalne energije verižice v odvisnosti do Nivojev za $k_BT=%.2f$ in $\alpha=%.2f$" %(kT,a))
ax8.set_xlabel("$Nivoji$")
ax8.legend()
ax8.grid()
ax8.set_ylabel("E")

ax5.set_title(r"Prikaz verižice za različne $\alpha$ in $k_b$")
ax5.legend()
ax5.set_xlabel("Mulekule")
ax5.set_ylabel(r"h")

ax6.set_title(r"Prikaz minimalne energije verižice v odvisnosti do $k_T$")
ax6.set_xlabel("$k_BT$")
ax6.legend()
ax6.grid()
ax6.set_ylabel("E")

ax3.set_title(r"Prikaz verižice za različne $\alpha$ pri $k_bT=%.2f$" %(kT))
ax3.legend()
ax3.set_xlabel("Mulekule")
ax3.set_ylabel(r"h")

ax4.set_title(r"Prikaz energije verižice v odvisnosti do iteracij")
ax6.set_xlabel("N")
ax6.legend()
ax6.grid()
ax6.set_ylabel("E")

ax1.set_title(r"Prikaz verižice za različne $k_bT$ pri $\alpha=%.2f$" %(a))
ax1.legend()
ax1.set_xlabel("Mulekule")
ax1.set_ylabel(r"h")

ax2.set_title(r"Prikaz energije verižice v odvisnosti do iteracij")
ax2.set_xlabel("N")
ax2.legend()
ax2.grid()
ax2.set_ylabel("E")
ax2.set_xscale("log")
plt.show()

X, Y = np.meshgrid(Mulekle,kT_list)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz spreminjanja verižice od $k_bT$")
surf = ax.plot_surface(X, Y, np.array(verizica_matrix_kT), cmap="coolwarm", linewidth=0, antialiased=False)
ax.set_ylabel("$k_bT$")
ax.set_xlabel(r"Mulekule")
ax.set_zlabel("h")
plt.show()

X, Y = np.meshgrid(N_list,kT_list)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz spreminjanja energije od $k_bT$")
surf = ax.plot_surface(X, Y, np.array(Energy_matrix_kT), cmap="coolwarm", linewidth=0, antialiased=False)
ax.set_ylabel("$k_bT$")
ax.set_xlabel(r"N")
ax.set_zlabel("Energija")
plt.show()

X, Y = np.meshgrid(Mulekle,kT_list)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz spreminjanja verižice od $\alpha$")
surf = ax.plot_surface(X, Y, np.array(verizica_matrix_a), cmap="coolwarm", linewidth=0, antialiased=False)
ax.set_ylabel(r"$\alpha$")
ax.set_xlabel(r"Mulekule")
ax.set_zlabel("h")
plt.show()

X, Y = np.meshgrid(N_list,kT_list)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz spreminjanja energije od $\alpha$")
surf = ax.plot_surface(X, Y, np.array(Energy_matrix_a), cmap="coolwarm", linewidth=0, antialiased=False)
ax.set_ylabel(r"$\alpha$")
ax.set_xlabel(r"N")
ax.set_zlabel("Energija")
plt.show()


X, Y = np.meshgrid(Mulekle,kT_list)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz spreminjanja verižice od $\alpha$ in $k_BT$")
surf = ax.plot_surface(X, Y, np.array(verizica_matrix_a_kt), cmap="coolwarm", linewidth=0, antialiased=False)
ax.set_ylabel("$k_BT$")
ax.set_xlabel(r"Mulekule")
ax.set_zlabel("h")
plt.show()

X, Y = np.meshgrid(kT_list,kT_list)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz spreminjanja energije od $\alpha$ in $k_BT$")
surf = ax.plot_surface(X, Y, np.array(Energy_matrix_a_kt), cmap="coolwarm", linewidth=0, antialiased=False)
ax.set_ylabel(r"$\alpha$")
ax.set_xlabel(r"$k_BT$")
ax.set_zlabel("Energija")
plt.show()

X, Y = np.meshgrid(Mulekle,Nivoji_list)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz spreminjanja verižice od nivojev")
surf = ax.plot_surface(X, Y, np.array(verizica_matrix_Nivoji), cmap="coolwarm", linewidth=0, antialiased=False)
ax.set_ylabel("$Nivoji$")
ax.set_xlabel(r"Mulekule")
ax.set_zlabel("h")
plt.show()

X, Y = np.meshgrid(N_list,Nivoji_list)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz spreminjanja energije od nivojev")
surf = ax.plot_surface(X, Y, np.array(Energy_matrix_Nivoji), cmap="coolwarm", linewidth=0, antialiased=False)
ax.set_ylabel(r"$\alpha$")
ax.set_xlabel(r"$k_BT$")
ax.set_zlabel("Energija")
plt.show()