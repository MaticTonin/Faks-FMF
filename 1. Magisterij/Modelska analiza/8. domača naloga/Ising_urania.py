import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D  
from numba import jit,njit
from numba.typed import List
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
N=50
J=1
H=0
iteration=10**7
k=1
mreza=np.zeros((N,N))
generated_mreza=np.zeros((N,N))
h=1

def energy(N,J, mreza,H):
    E=0
    for i in range(N):
        for j in range(N):
            E+=H*mreza[i,j]
            """
            if i-1 < 0:
                E = E - J*mreza[i][j]*mreza[N-1][j]
            else:
                E = E - J*mreza[i][j]*mreza[i-1][j]

            if i+1 >= N:
                E = E - J*mreza[i][j]*mreza[0][j]
            else:
                E = E - J*mreza[i][j]*mreza[i+1][j]

            if j-1 < 0:
                E = E - J*mreza[i][j]*mreza[i][N-1]
            else:
                E = E - J*mreza[i][j]*mreza[i][j-1]

            if j+1 >= N:
                E = E - J*mreza[i][j]*mreza[i][0]
            else:
                E = E - J*mreza[i][j]*mreza[i][j+1]"""
            E-=J*mreza[i,j]*(mreza[i-1,j]+mreza[(i+1)%N,j]+mreza[i,j-1]+mreza[i,(j+1)%N])
    return E
def energy_c(N,J,mreza,H):
    N_grid = mreza.shape[0]
    S=mreza
    E = np.zeros((N_grid,N_grid))
    for px in range(N_grid):
        for py in range(N_grid):

            E[px,py] = -J * S[px,py] * (S[px-1,py] + S[(px+1)%N_grid,py] + S[px,py-1] + S[px,(py+1)%N_grid]) - 2*H*S[px,py]

    return E

def spins(N,J, mreza,H):
    M=0
    for i in range(N):
        for j in range(N):
            M+=mreza[i,j]
    return M
def ising_model(N, iteration, k, J,H):
    mreza=np.zeros((N,N))
    generated_mreza=np.zeros((N,N))
    changes=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            number=random.randint(0,1)*2-1
            mreza[i,j]=number
            generated_mreza[i,j]=number
    for i in range(iteration):
        x=random.randint(0,N)
        y=random.randint(0,N)
        sprememba=1 if np.radom.rand()>0.5 else -1
        dE=0
        dE = 2*J*sprememba * (mreza[x-1,y] + mreza[(x+1)%N,y] + mreza[x,y-1] + mreza[x,(y+1)%N]) +2*sprememba
        if dE<0 or ( np.random.rand() <= np.exp(-dE/k) ):
            mreza[x,y] = sprememba
                
    return generated_mreza, mreza, dE, changes


mreza=np.zeros((N,N))
generated_mreza=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        number=random.randint(0,1)*2-1
        mreza[i,j]=number
        generated_mreza[i,j]=number
plt.imshow(mreza, cmap="Greys")
plt.show()
changes=np.zeros((N,N))
E1_list=[]
iteracije_list=[]
E=0
@jit(nopython=True)
def izdelava(N,k,H):
    mreza=np.zeros((N,N))
    generated_mreza=np.zeros((N,N))
    changes=np.zeros((N,N))
    E1_list=[]
    iteracije_list=[]
    for i in range(N):
        for j in range(N):
            number=random.randint(0,1)*2-1
            mreza[i,j]=number
            generated_mreza[i,j]=number
    E=0
    for i in range(iteration):
        #if i%(iteration/100)==0:
            #print(i/(iteration/100))
        x=random.randint(0,N-1)
        y=random.randint(0,N-1)
        mreza[x][y]=-mreza[x][y]
        changes[x][y]+=1
        dE=0
        if x-1 >=0:
            dE+=-2*J*mreza[x][y]*mreza[x-1][y]
        if x+1 < N:
            dE+=-2*J*mreza[x][y]*mreza[x+1][y]
        if y-1 >=0:
            dE+=-2*J*mreza[x][y]*mreza[x][y-1]
        if y+1 < N:
            dE+=-2*J*mreza[x][y]*mreza[x][y+1]
        if dE > 0:
            zeta = random.random()
            if zeta > np.exp(-dE/k):
                mreza[x][y] = - mreza[x][y]

        dE+=dE-2*H*mreza[x][y]
        E+=dE
        E1_list.append(E/N)
        iteracije_list.append(i)
    return E1_list,iteracije_list,mreza
  
#for i in tqdm(range(iteration)):
#    x=random.randint(0,N-1)
#    y=random.randint(0,N-1)
#    mreza[x][y]=-mreza[x][y]
#    changes[x,y]+=1
#    dE=0
#    if x-1 >=0:
#        dE+=-2*J*mreza[x][y]*mreza[x-1][y]
#    if x+1 < N:
#        dE+=-2*J*mreza[x][y]*mreza[x+1][y]
#    if y-1 >=0:
#        dE+=-2*J*mreza[x][y]*mreza[x][y-1]
#    if y+1 < N:
#        dE+=-2*J*mreza[x][y]*mreza[x][y+1]
#    if dE > 0:
#        zeta = random.random()
#        if zeta > np.exp(-dE/k):
#            mreza[x][y] = - mreza[x][y]
#
#    dE+=dE-2*H*mreza[x][y]
#    E+=dE
#    E1_list.append(E/N)
#    iteracije_list.append(i)
#
#E1_list,iteracije_list,mreza=izdelava(N,k,H)
#plt.title("Prikaz odvisnosti povprečne energije od iteracij")
#plt.plot(iteracije_list,E1_list, color="red")
#plt.grid()
#plt.xlabel("N")
#plt.ylabel(r"$\langle E \rangle$")
#plt.show()



#kB_T_list=np.linspace(0.1,5,70)
#Energy_kt_list=[]
#for k in tqdm(kB_T_list):
#    E1_list,iteracije_list,mreza=izdelava(N,k,H)
#    E=energy(N,J,mreza,H)
#    Energy_kt_list.append(E/N**2)
#plt.title("Prikaz odvisnosti povprečne energije od $k_BT$")
#plt.axvline(x=2.259185, color='b', label='Fazni prehod')
#plt.scatter(kB_T_list,Energy_kt_list, color="red")
#plt.legend()
#plt.grid()
#plt.xlabel("$k_BT$")
#plt.ylabel(r"$\langle E \rangle$")
#plt.show()

kB_T_list=np.linspace(0.1,3,100)
H_list=[0]
Energy_kt_list=[]
index=0
fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
fig3, ax3 = plt.subplots(1)
fig4, ax4 = plt.subplots(1)
for H in tqdm(H_list):
    Energy_kt_list=[]
    c_kt_list=[]
    M_list=[]
    chi_list=[]
    for k in tqdm(kB_T_list):
        E1_list,iteracije_list,mreza=izdelava(N,k,H)
        E=energy(N,J,mreza,H)
        Energy_kt_list.append(E/N**2)
        E_matrix=energy_c(N,J,mreza,H)
        c=(np.average(E_matrix**2) - np.average(E_matrix)**2)/(N**2*k**2)*k
        c_kt_list.append(c)
        M=spins(N,J,mreza,H)
        M_list.append(abs(M))
        chi=(np.average(mreza**2) - np.average(mreza)**2)/(2*N**2*k**2)*k
        chi_list.append(chi)
    ax1.scatter(kB_T_list,Energy_kt_list, label="H=$%.2f$" %(H), color=plt.cm.rainbow(index/(len(H_list))))
    ax2.scatter(kB_T_list,c_kt_list, label="H=$%.2f$" %(H),color=plt.cm.rainbow(index/(len(H_list))))
    ax3.scatter(kB_T_list,M_list, label="H=$%.2f$" %(H), color=plt.cm.rainbow(index/(len(H_list))))
    ax4.scatter(kB_T_list,chi_list, label="H=$%.2f$" %(H),color=plt.cm.rainbow(index/(len(H_list))))
    index+=1
ax1.set_title("Prikaz odvisnosti povprečne energije od $k_BT$")
ax1.axvline(x=2.259185, color='b', label='Fazni prehod')
ax1.legend()
ax1.grid()
ax1.set_xlabel("$k_BT$")
ax1.set_ylabel(r"$\langle E \rangle$")
fig1.savefig(THIS_FOLDER+ '/E.png', dpi=fig1.dpi)

ax2.set_title("Prikaz odvisnosti specifične toplote od $k_BT$")
ax2.axvline(x=2.259185, color='b', label='Fazni prehod')
ax2.legend()
ax2.grid()
ax2.set_xlabel("$k_BT$")
ax2.set_ylabel(r"$c$")
fig2.savefig(THIS_FOLDER+ '/C.png', dpi=fig2.dpi)

ax3.set_title("Prikaz odvisnosti magnetizacije od $k_BT$")
ax3.axvline(x=2.259185, color='b', label='Fazni prehod')
ax3.legend()
ax3.grid()
ax3.set_xlabel("$k_BT$")
ax3.set_ylabel(r"$\langle M \rangle$")
fig3.savefig(THIS_FOLDER+ '/M.png', dpi=fig3.dpi)

ax4.set_title("Prikaz odvisnosti susceptibilnosti od $k_BT$")
ax4.axvline(x=2.259185, color='b', label='Fazni prehod')
ax4.legend()
ax4.grid()
ax4.set_xlabel("$k_BT$")
ax4.set_ylabel(r"$\chi$")
fig4.savefig(THIS_FOLDER+ '/Chi.png', dpi=fig4.dpi)
plt.show()


