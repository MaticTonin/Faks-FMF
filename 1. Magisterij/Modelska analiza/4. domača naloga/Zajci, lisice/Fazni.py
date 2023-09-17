import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pandas as pd
import random
import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator

# function that returns dz/dt

n = 40000 #stevilo korakov integracije
l = 10 #čas do kamor simuliramo


def model(z,t,p):
    Z = z[0]
    L = z[1]
    dZdt = p*Z*(1-L)
    dLdt = L*(Z-1)/p
    dzdt = [dZdt,dLdt,p]
    return dzdt

Z0 = 2
L0 = 1.1
p0 = 2
z0 = [Z0,L0,p0]
t = np.linspace(0,l,n)

p = 1

IC = np.linspace(1.5, 6, 10)


nums=np.random.random((10,len(IC)))

colors = cm.rainbow(np.linspace(0, 1, nums.shape[0]))

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
        plt.scatter(z0[0],č, marker='x' ,color = 'k')
    
IC = np.linspace(0.1, 0.9, 9)

plt.show()
nums=np.random.random((10,len(IC)))

colors = cm.rainbow(np.linspace(0, 1, nums.shape[0]))

for i in IC:
    z0 = [i,1.1,p0]
    z = odeint(model,z0,t, args=(p,))
    plt.plot(z[:,0],z[:,1], label='Z0 = %.3f' %(i), )
   
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






plt.text(2.5, 9, "p = {}, L0 = {}".format(p, L0))
plt.xlabel('Z_0')
plt.grid()
plt.ylabel('obhodni čas')
plt.legend(loc='best')




plt.show()