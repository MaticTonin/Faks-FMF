import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import random
import pandas as pd
from tqdm import tqdm


# function that returns dz/dt

n = 100#stevilo korakov integracije
l = 35 #čas do kamor simuliramo


def model(z,t,p):
    Z = z[0]
    L = z[1]
    dZdt = p*Z*(1-L)
    dLdt = L*(Z-1)/p
    dzdt = [dZdt,dLdt]
    return dzdt

Z0 = 1
L0 = 1
t = np.linspace(0,l,n)
p=1
index=0
Z0=np.linspace(0.00001, 1,100)
L0=np.linspace(0.00001, 1,100)
C_l=[]
T=[]
for i in tqdm(L0):
    C_l_1=[]
    for j in Z0:
        z = odeint(model,[j,i],t, args=(p,))
        T.append(t)
        #ax2.plot(t,z[:,0] - np.log(z[:,0]) + (z[:,1] - np.log(z[:,1]))*p**2,'b',label='Lisice = %.3f' %(i), color=cm_b[index])
        C_l_1.append((z[:,0] - np.log(z[:,0]) + (z[:,1] - np.log(z[:,1]))*p**2)[0])
        index+=1
    C_l.append(C_l_1)

plt.show()
C_l=np.array(C_l)
T=np.array(T)
X, Y = np.meshgrid(L0, Z0)
print(X)
print(z[:,0])
print(C_l)
fig = plt.figure()
ax = plt.axes(projection='3d')
fig.suptitle(r"Prikaz stacionarnih točk na grafu za različne $l_0$ in $z_0$")
#ax.axes.set_zlim3d(bottom=0, top=20)
ax.set_xlabel("Lisice")
ax.set_ylabel('Zajci')
ax.set_zlabel('Konstanta C')
z=np.linspace(0,max(C_l[0]),100)
x=np.full((100),1)
y=np.full((100),1)
ax.plot3D(x, y, z, color='gray')
ax.scatter(1, 1, 0, color='black')
ax.text(1.2, 1.2, 0, "Point (1,1)", color='black')
scamap=plt.cm.ScalarMappable(cmap="coolwarm")
fcolor=scamap.to_rgba(C_l)
surf = ax.plot_surface(X,Y,C_l,facecolors=fcolor, cmap=cm.coolwarm)
fig.colorbar(scamap, shrink=0.5, aspect=5)
plt.show()
z0 = [Z0,L0]
t = np.linspace(0,l,n)
p=np.linspace(0.1, 8,15)
cm_b=cm.winter(np.linspace(0,1,15))
index=0
for i in p:
    z = odeint(model,z0,t, args=(i,))
    plt.plot(t,z[:,0] - np.log(z[:,0]) + (z[:,1] - np.log(z[:,1]))*i**2,'b',label='p = %.3f' %(i), color=cm_b[index])
    index+=1

plt.title('Energija populacije pri začetnih vrednostih zajcev=1 in lisic=1')
plt.ylabel("Energija, C")
plt.xlabel('t')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

Z0 = 1
L0 = 1

z0 = [Z0,L0]
p=1
t = np.linspace(0,l,n)
Z0=np.linspace(0, 8,2)
#L0=np.linspace(0, 8,9)
cm_b=cm.winter(np.linspace(0,1,8))
index=0
fig, (ax1,ax2) = plt.subplots(2)
fig.suptitle(r"Prikaz obnašanja gozda ")
for i in Z0:
    z = odeint(model,[i,L0],t, args=(p,))
    #ax1.plot(t,z[:,0] - np.log(z[:,0]) + (z[:,1] - np.log(z[:,1]))*p**2,'b',label='Zajci = %.3f' %(i), color=cm_b[index])
    index+=1

ax1.set_ylabel("Energija, C")
ax1.set_xlabel('t')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
Z0=np.linspace(0, 8,2)
L0=np.linspace(0.1, 10,10000)
C_l=[]
for i in L0:
    for j in Z0:
        z = odeint(model,[j,i],t, args=(p,))
    #ax2.plot(t,z[:,0] - np.log(z[:,0]) + (z[:,1] - np.log(z[:,1]))*p**2,'b',label='Lisice = %.3f' %(i), color=cm_b[index])
        C_l.append(z[:,0] - np.log(z[:,0]) + (z[:,1] - np.log(z[:,1]))*p**2)
        index+=1

ax2.set_ylabel("Energija, C")
ax2.set_xlabel('t')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
C_l=np.array(C_l)
X, Y = np.meshgrid(z[:,0], z[:,1])
print(X)
print(z[:,0])
print(C_l)
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.axes.set_zlim3d(bottom=0, top=20)
ax.set_xlabel("Zajci")
ax.set_ylabel('Lisice')
ax.set_zlabel('Konstanta C')
surf = ax.plot_surface(X,Y,C_l, cmap=cm.coolwarm)
fig.colorbar(surf, shrink=0.5, aspect=5)


#for i in range(1,10):
#    p = i/4
#    z = odeint(model,z0,t)
## plot results
#    #plt.plot(t,z[:,0],'b',alpha=i/4,label='Zajci,  p = %.3f' %(p))
#    plt.plot(t,z[:,1],'r',alpha=i/9,label='Lisice, p = %.3f' %(p))
#plt.ylabel('brezdimenzijsko število živali')
#plt.xlabel('čas')
#plt.legend(loc='best')
#
plt.show()
