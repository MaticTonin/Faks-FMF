import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pandas as pd
import random
import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import matplotlib as mpl

# function that returns dz/dt

n = 100000 #stevilo korakov integracije
l = 30 #čas do kamor simuliramo


def model(z,t,p):
    Z = z[0]
    L = z[1]
    dZdt = p*Z*(1-L)
    dLdt = L*(Z-1)/p
    dzdt = [dZdt,dLdt,p]
    return dzdt
"""
Z0 = 2
L0 = 1.1
p0 = 2
z0 = [Z0,L0,p0]
t = np.linspace(0,l,n)

p = 1
n_lines=20
IC_max=6
IC_min=1
IC = np.linspace(IC_min, IC_max, n_lines)
B_tixcs=[]
cm = plt.cm.winter
fig,(ax, ax2) = plt.subplots(2)
c = np.arange(1, n_lines + 1)
i=IC_min
while i<IC_max:
    B_tixcs.append(i)
    i+=0.5
B_tixcs=np.array(B_tixcs)
norm = mpl.colors.Normalize(vmin=IC.min(), vmax=IC.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
cmap.set_array([])

nums=np.random.random((10,len(IC)))


colors = plt.cm.rainbow(np.linspace(0, 1, nums.shape[0]))
colors = plt.cm.jet(np.linspace(0,1,n))
fig.suptitle(r"Prikaz obnašanja gozda pri parametru $p=%.2f$ in spreminjanju začetne populacije" %(p))
index=0
for i in IC:
    z0 = [i,1.1,p0]
    z = odeint(model,z0,t, args=(p,))
    ax.plot(z[:,0],z[:,1], label='Z0 = %.3f' %(i),c=cmap.to_rgba(index))
    ax.set_title("Spreminjanje popualcije zajcev")
    z0 = [i,i,p0]
    z = odeint(model,z0,t, args=(p,))
    ax2.set_title("Spreminjanje populacije obeh")
    ax2.plot(z[:,0],z[:,1], label='Z0 = %.3f' %(i),c=cmap.to_rgba(index))
   
    k = 0
    s = 0
    č = 0
    for j in (z[:,0]):
        if s > 10000:
            j = round(j,3)

            if j == z[0,0]:
                k=k+1
                print(k)
                if k == 2:
                    č = t[s] - t[0]
                    break
        s = s + 1
    print(z0[0], č)
    index+=0.30
    #if č != 0:
        #plt.scatter(z0[0],č, marker='x' ,color = 'k')
    
IC = np.linspace(0.1, 0.9, 9)
cbar=fig.colorbar(cmap, cax = fig.add_axes([0.92, 0.1, 0.03, 0.8]) ,ticks=B_tixcs)
plt.show()



"""

Z0 = 2
L0 = 1.1
p0 = 2
z0 = [Z0,L0,p0]
t = np.linspace(0,l,n)

p = 1
n_lines=20
IC_max=8
IC_min=1
IC = np.linspace(IC_min, IC_max, n_lines)
B_tixcs=[]
cm = plt.cm.winter
fig,(ax, ax2) = plt.subplots(2)
c = np.arange(1, n_lines + 1)
i=IC_min
while i<IC_max:
    B_tixcs.append(i)
    i+=0.5
B_tixcs=np.array(B_tixcs)
norm = mpl.colors.Normalize(vmin=IC.min(), vmax=IC.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.cool)
cmap.set_array([])

nums=np.random.random((10,len(IC)))


colors = plt.cm.rainbow(np.linspace(0, 1, nums.shape[0]))
n = 20
colors = plt.cm.jet(np.linspace(0,1,n))
fig.suptitle(r"Prikaz obnašanja gozda pri spreminjanju parametra p")
index=0
for i in IC:
    z0 = [2,2,i]
    z = odeint(model,z0,t, args=(i,))
    ax.plot(z[:,0],z[:,1], label='Z0 = %.3f' %(i),c=cmap.to_rgba(index))
    ax.set_title("Populacija 2 zajcev in 2 lisici")
    ax.set_xlabel("Zajci")
    ax.set_ylabel("Lisice")
    z0 = [10,2,i]
    z = odeint(model,z0,t, args=(i,))
    ax2.set_title("Populacija 10 zajcev in 2 lisice")
    ax2.plot(z[:,0],z[:,1], label='Z0 = %.3f' %(i),c=cmap.to_rgba(index))
    ax2.set_xlabel("Zajci")
    ax2.set_ylabel("Lisice")
   
    index+=0.40
    #if č != 0:
        #plt.scatter(z0[0],č, marker='x' ,color = 'k'

IC = np.linspace(0.001, 1, 1000)
cbar=fig.colorbar(cmap, cax = fig.add_axes([0.92, 0.1, 0.03, 0.8]) ,ticks=B_tixcs)
plt.show()
nums=np.random.random((10,len(IC)))

colors = plt.cm.rainbow(np.linspace(0, 1, nums.shape[0]))

for i in IC:
    z0 = [i,1.1,p0]
    z = odeint(model,z0,t, args=(p,))
    #plt.plot(z[:,0],z[:,1], label='Z0 = %.3f' %(i), )
   
    k = 0
    s = 0
    č = 0
    for j in (z[:,0]):
        if s > 10000:
            j = round(j,3)

            if j == z[0,0]:
                k=k+1
                print(k)
                if k == 2:
                    č = t[s] - t[0]
                    break
        s = s + 1
    print(z0[0], č)
    if č != 0:
        plt.scatter(z0[0],č, marker='.' ,color = 'r')

IC = np.linspace(1, 10, 100)
cbar=fig.colorbar(cmap, cax = fig.add_axes([0.92, 0.1, 0.03, 0.8]) ,ticks=B_tixcs)

nums=np.random.random((10,len(IC)))

colors = plt.cm.rainbow(np.linspace(0, 1, nums.shape[0]))

for i in IC:
    z0 = [i,1.1,p0]
    z = odeint(model,z0,t, args=(p,))
    #plt.plot(z[:,0],z[:,1], label='Z0 = %.3f' %(i), )
   
    k = 0
    s = 0
    č = 0
    for j in (z[:,0]):
        if s > 10000:
            j = round(j,3)

            if j == z[0,0]:
                k=k+1
                print(k)
                if k == 2:
                    č = t[s] - t[0]
                    break
        s = s + 1
    print(z0[0], č)
    if č != 0:
        plt.scatter(z0[0],č, marker='.' ,color = 'r')

#plt.text(2.5, 9, "p = {}, L0 = {}".format(p, L0))
plt.title(r"Prikaz obhodnega časa v odvisnosti od števila zajcev")
plt.xlabel('Število zajcev')
plt.ylabel('Obhodni čas')
plt.legend(loc='best')
plt.show()