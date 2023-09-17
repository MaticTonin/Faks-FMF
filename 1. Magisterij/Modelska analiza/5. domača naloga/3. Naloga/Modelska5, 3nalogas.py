from re import A
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from itertools import cycle

# function that returns dz/dt
n = 250000 #stevilo korakov integracije
l = 100 #čas do kamor simuliramo
p = 1
q = 1
la=p/q
def model(w,ti,la):
    x = w[0]
    y = w[1]
    z = w[2]
    A=  w[3]
    B=  w[4]
    C=  w[5]

    dxdt = 2*y*z**2-2*A*la*x**2
    dydt = la*A*x**2-y*z**2
    dzdt = -2*y*z**2
    dAdt = -la*A*x**2
    dBdt = 2*la*A*x**2
    dCdt = y*z**2

    dwdt = [dxdt,dydt, dzdt, dAdt, dBdt, dCdt]
    return dwdt

# initial condition
x0 = 1 #HI-
y0 = 0  #I2
z0 = 1  #SO
A0 = 1
B0 = 0
C0 = 0

w0 = [x0, y0, z0, A0, B0, C0]

# time points
ti = np.linspace(0,l,n)
IC = np.linspace(0,10,100)
for i in [1,10,100]:
    lines = ["-","--",":"]
    linecycler = cycle(lines)
    w = odeint(model,w0,ti,args=(i,))
    x,y,z,A,B,C=w[:,0],w[:,1], w[:,2], w[:,3], w[:,4], w[:,5]
    plt.title(r"Prikaz reakcije; $\lambda=%.2f$" %(i))
    plt.plot(ti,x,label='x = %.1f' %(x0), color=plt.cm.gist_rainbow(0/len(w0)))
    plt.plot(ti,y,label="y = %.1f" %(y0), color=plt.cm.gist_rainbow(1/len(w0)))
    plt.plot(ti,z,label='z = %.1f' %(z0), color=plt.cm.gist_rainbow(2/len(w0)))
    plt.plot(ti,A,label='A = %.0f' %(A0), color=plt.cm.gist_rainbow(3/len(w0)))
    plt.plot(ti,B,label='B = %.0f' %(B0), color=plt.cm.gist_rainbow(4/len(w0)))
    plt.plot(ti,C,label='C = %.0f' %(C0), color=plt.cm.gist_rainbow(5/len(w0)))
    plt.ylabel('Koncentracija')
    plt.xlabel('čas')
    plt.xscale("log")
    plt.legend()
    plt.grid()
    plt.show()
index=0
number=[1,2,5,10,25,50,100]
for i in [1,2,5,10,25,50,100]:
    lines = ["-","--",":"]
    linecycler = cycle(lines)
    w = odeint(model,w0,ti,args=(i,))
    x,y,z,A,B,C=w[:,0],w[:,1], w[:,2], w[:,3], w[:,4], w[:,5]
    plt.title(r"Prikaz reakcije za različne $\lambda=[1,2,5,10,25,50,100]$")
    if index==0 or index==len(number)-1: 
        plt.plot(ti,x,label='x', color=plt.cm.gist_rainbow(index/len(number)))
        plt.plot(ti,y,label="y" , color=plt.cm.cool(index/len(number)))
        plt.plot(ti,z,label='z', color=plt.cm.summer(index/len(number)))
    else:
        plt.plot(ti,x, color=plt.cm.gist_rainbow(index/len(number)))
        plt.plot(ti,y, color=plt.cm.cool(index/len(number)))
        plt.plot(ti,z, color=plt.cm.summer(index/len(number)))
    plt.ylabel('Koncentracija')
    plt.xlabel('čas')
    plt.xscale("log")
    plt.legend()
    plt.grid()
    index+=1
plt.show()
index=0
for i in [1,5,10,25,50,100]:
    lines = ["-","--",":"]
    linecycler = cycle(lines)
    w = odeint(model,w0,ti,args=(i,))
    x,y,z,A,B,C=w[:,0],w[:,1], w[:,2], w[:,3], w[:,4], w[:,5]
    plt.title(r"Prikaz reakcije za različne $\lambda=[1,2,5,10,25,50,100]$")
    plt.plot(ti,x,label='x = %.1f' %(x0), color=plt.cm.gist_rainbow(index/len(number)))
    plt.ylabel('Koncentracija')
    plt.xlabel('čas')
    plt.xscale("log")
    plt.legend()
    plt.grid()
    index+=1
index=0
for i in [1,5,10,25,50,100]:
    lines = ["-","--",":"]
    linecycler = cycle(lines)
    w = odeint(model,w0,ti,args=(i,))
    x,y,z,A,B,C=w[:,0],w[:,1], w[:,2], w[:,3], w[:,4], w[:,5]
    plt.title(r"Prikaz reakcije za različne $\lambda=[1,2,5,10,25,50,100]$")
    plt.plot(ti,y,label="y = %.1f" %(y0), color=plt.cm.cool(index/len(number)))
    plt.ylabel('Koncentracija')
    plt.xlabel('čas')
    plt.xscale("log")
    plt.legend()
    plt.grid()
    index+=1
index=0
for i in [1,5,10,25,50,100]:
    lines = ["-","--",":"]
    linecycler = cycle(lines)
    w = odeint(model,w0,ti,args=(i,))
    x,y,z,A,B,C=w[:,0],w[:,1], w[:,2], w[:,3], w[:,4], w[:,5]
    plt.title(r"Prikaz reakcije za različne $\lambda=[1,2,5,10,25,50,100]$")
    plt.plot(ti,z,label='z = %.1f' %(z0), color=plt.cm.summer(index/len(number)))
    plt.ylabel('Koncentracija')
    plt.xlabel('čas')
    plt.xscale("log")
    plt.legend()
    plt.grid()
    index+=1
plt.ylabel('Koncentracija')
plt.xlabel('čas')
plt.xscale("log")
plt.legend()
plt.grid()
plt.show()
# solve ODE
