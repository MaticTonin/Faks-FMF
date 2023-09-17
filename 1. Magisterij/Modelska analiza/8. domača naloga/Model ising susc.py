import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D  

N=100
J=1
H=0
iteration=10**6
k=0.01
mreza=np.zeros((N,N))
generated_mreza=np.zeros((N,N))
h=1
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
        x=random.randint(0,N-1)
        y=random.randint(0,N-1)
        mreza[x][y]=-mreza[x][y]
        changes[x,y]+=1
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
    return generated_mreza, mreza, dE, changes

def ising_model_energy_total(N, iteration, k, J,H, index,Elist):
    mreza=np.zeros((N,N))
    generated_mreza=np.zeros((N,N))
    changes=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            number=random.randint(0,1)*2-1
            mreza[i,j]=number
            generated_mreza[i,j]=number
    E=0
    for i in range(N):
        for j in range(N):
            E+=h*mreza[i,j]
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
                E = E - J*mreza[i][j]*mreza[i][j+1]

    dE_list=[]
    iteration_list=[]
    for i in range(iteration):
        x=random.randint(0,N-1)
        y=random.randint(0,N-1)
        mreza[x][y]=-mreza[x][y]
        changes[x,y]+=1
        dE=0
        if x-1 >=0:
            dE+=-2*J*mreza[x][y]*mreza[x-1][y]
        if x+1 < N:
            dE+=-2*J*mreza[x][y]*mreza[x+1][y]
        if y-1 >=0:
            dE+=-2*J*mreza[x][y]*mreza[x][y-1]
        if y+1 < N:
            dE+=-2*J*mreza[x][y]*mreza[x][y+1]
        E+=dE
        if dE > 0:
            zeta = random.random()
            if zeta > np.exp(-dE/k):
                mreza[x][y] = - mreza[x][y]
                E-=dE
        Elist[i]=E
        dE+=dE-2*H*mreza[x][y]
        dE_list.append(E)
        iteration_list.append(i)
    st = 6000
    n = 0
    esum = 0
    eprod = 0
    while st < iteration:
        esum = esum + Elist[st]
        eprod = eprod + (Elist[st])**2

        n = n + 1 
        st = st + 5000

    eavr = esum/n
    eavr2 = eprod/n 
    
    plt.plot(iteration_list,dE_list, label=r"$k_BT=$%.2f,H=$%.2f$" %(k,H), color=plt.cm.rainbow(index/(len(kt_list)+len(kt_list))))
    return generated_mreza, mreza, dE, changes, eavr, eavr2

def ising_model_energy_n(N, iteration, k, J,H, index):
    mreza=np.zeros((N,N))
    generated_mreza=np.zeros((N,N))
    changes=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            number=random.randint(0,1)*2-1
            mreza[i,j]=number
            generated_mreza[i,j]=number
    E=0
    for i in range(N):
        for j in range(N):
            E+=h*mreza[i,j]
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
                E = E - J*mreza[i][j]*mreza[i][j+1]

    dE_list=[]
    iteration_list=[]
    for i in range(iteration):
        x=random.randint(0,N-1)
        y=random.randint(0,N-1)
        mreza[x][y]=-mreza[x][y]
        changes[x,y]+=1
        dE=0
        if x-1 >=0:
            dE+=-2*J*mreza[x][y]*mreza[x-1][y]
        if x+1 < N:
            dE+=-2*J*mreza[x][y]*mreza[x+1][y]
        if y-1 >=0:
            dE+=-2*J*mreza[x][y]*mreza[x][y-1]
        if y+1 < N:
            dE+=-2*J*mreza[x][y]*mreza[x][y+1]
        E+=dE
        if dE > 0:
            zeta = random.random()
            if zeta > np.exp(-dE/k):
                mreza[x][y] = - mreza[x][y]
                E-=dE
        dE+=dE-2*H*mreza[x][y]
        dE_list.append(E)
        iteration_list.append(i)
    plt.plot(iteration_list,dE_list, label=r"$N=%i$" %(N), color=plt.cm.rainbow(index/len(N_list)))
    return generated_mreza, mreza, dE, changes

kt_list=np.linspace(0.1,5,6)
kt_list=[0.5,1,2.27,5]
dE_list=[]
index=0
plt.title("Prikaz konvergence energije za velikost matrike $N=%i$" %(N))
esum_list=[]
eprod_list=[]
no=0
plot=0
for k in tqdm(kt_list):
    esum_list=[]
    eprod_list=[]
    Elist = np.zeros(iteration)
    for H in tqdm(kt_list):
        generated_mreza, mreza, dE, changes, eavr, eavr2=ising_model_energy_total(N, iteration, k, J,H,index,Elist)
        esum_list.append(eavr)
        eprod_list.append(eavr2)
        index+=1
    """
    f.suptitle(r"Prikaz končnega in začetnega stanja $k_BT=%.2f, H=%.2f$" %(k,H))
    f.add_subplot(1,2, 1)
    plt.title(r"Začetno stanje")
    im=plt.imshow(generated_mreza, cmap="coolwarm")
    f.add_subplot(1,2, 2)
    plt.title(r"Končno stanje")
    im=plt.imshow(mreza, cmap="coolwarm")
    plt.show()
    plt.title("Prikaz ustvarjanja domen $k_BT=%.2f, H=%.2f$" %(k,H))
    im1=plt.imshow(generated_mreza,alpha=0.7, cmap="jet", label="A")
    im2=plt.imshow(mreza, alpha=0.5,cmap="Greys")
    plt.legend()
    plt.show()
    plt.title("Prikaz števila sprememb spinov $k_BT=%.2f, H=%.2f$" %(k,H))
    im=plt.imshow(changes, cmap="jet")
    plt.colorbar(im, label="$N$")
    plt.show()"""
    c = np.zeros((esum_list))
    print(eprod_list, esum_list,kt_list)
    for i in range(len(esum_list)):
        c[i] =  (eprod_list[i] - esum_list[i]**2)/(N*kt_list[i]**2)

    if plot == 0:
        plt.plot(kt_list, c, label= 'H='+str(H), color = c)
    if plot == 1:
        plt.plot(kt_list, esum_list/(N**2), label= 'H='+str(H), color = plt.cm.rainbow(index/len(kt_list)))
    no = no + 1
plt.legend()
plt.xscale("log")
plt.xlabel("$k_BT")
plt.ylabel("E")
plt.grid()
plt.show()

N_list=[10,50,75,100]
k=0.5
index=0
plt.title("Prikaz konvergence energije za $k_BT=%.2f$" %(k))
for N in tqdm(N_list):
    generated_mreza, mreza, dE, changes=ising_model_energy_n(N, iteration, k, J,H,index)
    index+=1
    """
    f.suptitle(r"Prikaz končnega in začetnega stanja $k_BT=%.2f, H=%.2f$" %(k,H))
    f.add_subplot(1,2, 1)
    plt.title(r"Začetno stanje")
    im=plt.imshow(generated_mreza, cmap="coolwarm")
    f.add_subplot(1,2, 2)
    plt.title(r"Končno stanje")
    im=plt.imshow(mreza, cmap="coolwarm")
    plt.show()
    plt.title("Prikaz ustvarjanja domen $k_BT=%.2f, H=%.2f$" %(k,H))
    im1=plt.imshow(generated_mreza,alpha=0.7, cmap="jet", label="A")
    im2=plt.imshow(mreza, alpha=0.5,cmap="Greys")
    plt.legend()
    plt.show()
    plt.title("Prikaz števila sprememb spinov $k_BT=%.2f, H=%.2f$" %(k,H))
    im=plt.imshow(changes, cmap="jet")
    plt.colorbar(im, label="$N$")
    plt.show()"""
plt.legend()
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("E")
plt.grid()
plt.show()