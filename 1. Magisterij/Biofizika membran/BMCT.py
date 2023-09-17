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
R_max=1
N=100
R=np.linspace(0,R_max,N)
gamma_list=np.linspace(0.1,100,N)
i=0
R_extr=[]
W_extr=[]
normalize = mcolors.Normalize(vmin=gamma_list.min(), vmax=gamma_list.max())
colormap = cm.coolwarm
W_matrix=[]
gamma_extr=[]
for gamma in gamma_list:
    W=np.pi/R-2*np.pi*gamma*(1-R)
    W_matrix.append(W)
    plt.plot(R,W, color=colormap(normalize(gamma)))
    Meja=1/np.sqrt(2*gamma)
    if Meja<R_max:
        gamma_extr.append(gamma_list[i])
        R_extr.append(1/np.sqrt(2*gamma))
        W_extr.append(np.pi*np.sqrt(2*gamma)-2*np.pi*gamma*(1-1/np.sqrt(2*gamma)))
    i+=1
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(gamma_list)
cbar=plt.colorbar(scalarmappaple)
cbar.set_label(r"$\gamma$", rotation=90)
plt.title(r"Prikaz energij za različne $\gamma$")
plt.plot(R_extr,W_extr,"--",color="black", label=r"$R=\frac{1}{\sqrt{2\gamma}}$")
plt.xlabel("R")
plt.ylabel("W")
plt.legend()
plt.show()
W_matrix=np.array(W_matrix, dtype=float)
X,Y=np.meshgrid(R,gamma_list)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title(r"Prikaz energij za različne $\gamma$")
im=ax.plot_surface(X, Y, W_matrix, rstride=1, cstride=1,
                cmap='seismic', edgecolor='none')
ax.plot3D(R_extr, gamma_extr,np.array(W_extr)+0.1,"black")
#im=ax.plot_surface(X, Y, abs(Wawe0), rstride=1, cstride=1, cmap='seismic', edgecolor='none')
cbar = ax.figure.colorbar(im, label=r"$W$")
ax.set_xlabel("R")
ax.set_ylabel(r"$\gamma$")
ax.legend()
plt.show()


a=np.linspace(0.1,1,N)
gamma_0_plus=1/2*1/(1+np.sqrt(1-a))**2
gamma_0_minus=1/2*1/(1-np.sqrt(1-a))**2
gamma_real_plus=1/2*1/(1+2*np.sqrt((a-1)/(np.pi-4)))**2
gamma_real_minus=1/2*1/(1-2*np.sqrt((a-1)/(np.pi-4)))**2
min=50
max=0
index=0
import cmath
for gamma_list in [gamma_0_plus, gamma_0_minus, gamma_real_plus, gamma_real_minus]:
    print(gamma_list)
    for gamma in gamma_list:
        if gamma<min:
            min=gamma
        elif gamma>max and gamma!="inf":
            max=gamma
real_minus=0
ind0_minus=0
gamma_real_minus_pred=[]
gamma_real_minus_po=[]
test=0
for i in range(N):
    if gamma_real_minus[i]<max and test==1:
        gamma_real_minus_po.append(gamma_real_minus[i])
    if gamma_real_minus[i]==max:
        gamma_real_minus_po.append(gamma_real_minus[i])
        test=1
        index=i
print(max)
print(index)
plt.title("Prikaz $\gamma$ v odvisnosti od paramtera a")
plt.fill_between(a, gamma_0_minus,max,color="Green", alpha=0.5, label=r"$\alpha>0$")
plt.fill_between(a, gamma_0_minus,max,color="Green", alpha=0.5)
plt.fill_between(a[index:N],gamma_real_minus_po,max,color=plt.cm.Reds(50/100), alpha=0.5, label=r"$Imag(\alpha)$")
plt.fill_between(a[index:N],gamma_real_minus_po,max,color=plt.cm.Reds(50/100), alpha=0.5)
plt.fill_between(a,gamma_0_plus,gamma_0_minus,color="Green", alpha=0.5)
plt.fill_between(a,gamma_0_plus,min,color=plt.cm.Reds(50/100), alpha=0.5)
plt.plot(a,gamma_0_plus, label=r"$\gamma_{0,+}$", color=plt.cm.PiYG(0/100))
plt.plot(a,gamma_0_minus,label=r"$\gamma_{0,-}$",color=plt.cm.PiYG(20/100))
plt.plot(a,gamma_real_plus,label=r"$\gamma_{real,+}$",color=plt.cm.PiYG(70/100))
plt.plot(a,gamma_real_minus,label=r"$\gamma_{real,-}$",color=plt.cm.PiYG(100/100))
plt.xlabel("a")
plt.ylabel(r"$\gamma$")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()
W_0_minus=np.pi*np.sqrt(2*gamma_0_minus)*(2-np.sqrt(2*gamma_0_minus))
W_0_plus=np.pi*np.sqrt(2*gamma_0_plus)*(2-np.sqrt(2*gamma_0_plus))
W_real_plus=np.pi*np.sqrt(2*gamma_real_plus)*(2-np.sqrt(2*gamma_real_plus))
W_real_minus=np.pi*np.sqrt(2*gamma_real_minus)*(2-np.sqrt(2*gamma_real_minus))
W_real_minus_po=np.pi*np.sqrt(2*gamma_real_minus_po)*(2-np.sqrt(2*gamma_real_minus_po))
print(W_real_minus_po)
min=0
max=0
for gamma_list in [W_0_plus, W_0_minus, W_real_plus,W_real_minus]:
    for gamma in gamma_list:
        if gamma<min:
            min=gamma
        elif gamma>max and gamma!="inf":
            max=gamma
W_real_minus_po=[]
index=0
for i in range(len(W_real_minus)):
    print(a[i])
    if abs(a[i]-0.844)<0.1 and index==0:
        #W_real_minus_po.append(W_real_minus[i])
        index=i
    if index!=0:
        W_real_minus_po.append(W_real_minus[i])


import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "W.txt")
a1, W = np.loadtxt(my_file, unpack="True")
plt.title("Prikaz energije v odvisnosti od paramtera a")
plt.fill_between(a, W_0_minus,min,color="Green", alpha=0.5)
plt.fill_between(a[index:N],W_real_minus_po,min,color=plt.cm.Reds(50/100), alpha=0.5)
plt.fill_between(a,W_0_plus,W_0_minus,color="Green", alpha=0.5)
plt.fill_between(a,W_0_minus,min,color="Green", alpha=0.5,label=r"$\alpha>0$")
plt.fill_between(a[index:N],W_real_minus_po,min,color=plt.cm.Reds(50/100), alpha=0.5, label=r"$Imag(\alpha)$")
plt.fill_between(a[index:N],W_real_minus_po,min,color=plt.cm.Reds(50/100), alpha=0.5)
plt.fill_between(a,W_0_plus,max,color=plt.cm.Reds(50/100), alpha=0.5)
plt.plot(a1,W, "--",label=r"$W_{free}$ ", color="black")
plt.plot(a,W_0_plus, label=r"$W_{0,+}$", color=plt.cm.PiYG(0/100))
plt.plot(a,W_0_minus,label=r"$W_{0,-}$",color=plt.cm.PiYG(20/100))
plt.plot(a,W_real_plus,label=r"$W_{real,+}$",color=plt.cm.PiYG(70/100))
plt.plot(a,W_real_minus,label=r"$W_{real,-}$",color=plt.cm.PiYG(100/100))
plt.xlabel("a")
plt.ylabel(r"$W$")
plt.yscale('symlog')
plt.legend()
plt.ylim(-10**(2),10**1.5)
plt.grid()
plt.show()
M=10000
a=np.linspace(0.1,1,M)
alpha=np.zeros((M,M),dtype=complex)
gamma_list=np.linspace(max,min,M)
print(min, max)

from tqdm import tqdm
fig, ax=plt.subplots()
for i in tqdm(range(len(gamma_list))):
    for j in range(len(a)):
        R_c=1/np.sqrt(2*gamma_list[i])
        calculation_alpha_plus=(1/2*(np.pi*(1-R_c)+cmath.sqrt(np.pi**2*(1-R_c)**2+4*np.pi*R_c*(2-R_c)-4*np.pi*a[j])))
        calculation_alpha_minus=(1/2*(np.pi*(1-R_c)-cmath.sqrt(np.pi**2*(1-R_c)**2+4*np.pi*R_c*(2-R_c)-4*np.pi*a[j])))
        calculation_alpha_minus=calculation_alpha_plus
        calculation_beta_plus=np.pi*(1-R_c)-calculation_alpha_plus
        calculation_beta_minus=np.pi*(1-R_c)-calculation_alpha_minus
        #print(calculation)
        if calculation_alpha_plus.real>0 and calculation_beta_plus.real>0 and calculation_alpha_plus.imag==0:
            alpha[i,j]=3
        if calculation_alpha_minus.real>0 and calculation_beta_minus.real>0 and calculation_alpha_minus.imag==0:
            alpha[i,j]=3
        elif calculation_alpha_plus.real<0 and calculation_alpha_plus.imag==0:
            alpha[i,j]=3
        elif calculation_alpha_minus.real<0 and calculation_alpha_minus.imag==0:
            alpha[i,j]=1
        elif calculation_alpha_plus.imag!=0:
            alpha[i,j]=0
n_clusters=4
cmap = plt.get_cmap('PiYG', n_clusters)
im=ax.imshow(alpha.real, cmap=cmap, extent=[0.1, 4, 0.1, 100])
cbar=plt.colorbar(im, alpha=0.5)
cbar.ax.get_yaxis().set_ticks([])
tick_locs = (np.arange(n_clusters) + 0.5)*(n_clusters-1)/n_clusters
cbar.set_ticks(tick_locs)
ax.set_yscale("log")
# set tick labels (as before)
ax.set_title(r"Prikaz matrike $\alpha$ v odvisnosti od $\gamma$ in $a$")
cbar.set_ticklabels([r'$Imag(\alpha)$',r'$\alpha_-<0$',r'$\alpha_+<0$',r'$\alpha>0$'])
ax.set_xticks([0.1, 1,2, 3,4])
ax.set_xticklabels([str(round(a[0],2)),str(round(a[int(M/4)],2)),str(round(a[int(2*M/4)],2)),str(round(a[int(3*M/4)],2)),str(round(a[M-1],2))])
#ax.set_yticks([0, M/4,2*M/4, 3*M/4,M-1], [str(round(gamma_list[0],2)),str(round(gamma_list[int(M/4)],2)),str(round(gamma_list[int(2*M/4)],2)),str(round(gamma_list[int(3*M/4)],2)),str(round(gamma_list[M-1],2))])
ax.set_ylabel(r"$\gamma$")
ax.set_xlabel(r"a")


plt.show()

from tqdm import tqdm
fig, ax=plt.subplots()
for i in tqdm(range(len(gamma_list))):
    for j in range(len(a)):
        R_c=1/np.sqrt(2*gamma_list[i])
        calculation_alpha_plus=(1/2*(np.pi*(1-R_c)+cmath.sqrt(np.pi**2*(1-R_c)**2+4*np.pi*R_c*(2-R_c)-4*np.pi*a[j])))
        calculation_alpha_minus=(1/2*(np.pi*(1-R_c)-cmath.sqrt(np.pi**2*(1-R_c)**2+4*np.pi*R_c*(2-R_c)-4*np.pi*a[j])))
        calculation_beta_plus=np.pi*(1-R_c)-calculation_alpha_plus
        calculation_beta_minus=np.pi*(1-R_c)-calculation_alpha_minus
        #print(calculation)
        if calculation_alpha_plus.real>0 and calculation_beta_plus.real>0 and calculation_alpha_plus.imag==0:
            alpha[i,j]=3
        if calculation_alpha_minus.real>0 and calculation_beta_minus.real>0 and calculation_alpha_minus.imag==0:
            alpha[i,j]=3
        elif calculation_alpha_plus.real<0 and calculation_alpha_plus.imag==0:
            alpha[i,j]=3
        elif calculation_alpha_minus.real<0 and calculation_alpha_minus.imag==0:
            alpha[i,j]=1
        elif calculation_alpha_plus.imag!=0:
            alpha[i,j]=0
n_clusters=4
cmap = plt.get_cmap('PiYG', n_clusters)
im=ax.imshow(alpha.real, cmap=cmap)
cbar=plt.colorbar(im, alpha=0.5)
cbar.ax.get_yaxis().set_ticks([])
tick_locs = (np.arange(n_clusters) + 0.5)*(n_clusters-1)/n_clusters
cbar.set_ticks(tick_locs)
# set tick labels (as before)
ax.set_title(r"Prikaz matrike $\alpha$ v odvisnosti od $\gamma$ in $a$")
cbar.set_ticklabels([r'$Imag(\alpha)$',r'$\alpha_-<0$',r'$\alpha_+<0$',r'$\alpha>0$'])
ax.set_xticks([0, M/4,2*M/4, 3*M/4,M-1])
ax.set_xticklabels([str(round(a[0],2)),str(round(a[int(M/4)],2)),str(round(a[int(2*M/4)],2)),str(round(a[int(3*M/4)],2)),str(round(a[M-1],2))])
ax.set_yticks([0, M/4,2*M/4, 3*M/4,M-1], [str(round(gamma_list[0],2)),str(round(gamma_list[int(M/4)],2)),str(round(gamma_list[int(2*M/4)],2)),str(round(gamma_list[int(3*M/4)],2)),str(round(gamma_list[M-1],2))])
ax.set_ylabel(r"$\gamma$")
ax.set_xlabel(r"a")

ax21 = ax.twinx()
a=np.linspace(1,M,N)
ax21.plot(a,gamma_0_plus-0.15, label=r"$\gamma_{0,+}$", color="orange")
ax21.plot(a,gamma_0_minus-0.15,label=r"$\gamma_{0,-}$",color="orange")
ax21.plot(a,gamma_real_plus-0.15,label=r"$\gamma_{real,+}$",color="yellow")
ax21.plot(a,gamma_real_minus-0.15,label=r"$\gamma_{real,+}$",color="yellow")
ax21.set_yticks([],[])
ax21.grid()
ax21.legend()
#ax21.set_yscale("log")
#ax21.set_ylabel(r"$\log(\gamma)$")
#ax21.set_xlim(ax.get_xlim())
a = np.diff(ax21.get_ylim())[0]/np.diff(ax.get_xlim())*alpha.shape[1]/alpha.shape[0]
ax21.set_aspect(1./a, "datalim")

plt.show()