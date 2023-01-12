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


index=0
number=[1,5,10,20,35,50]
for i in [1,5,10,20,35,50]:
    lines = ["-","--",":"]
    linecycler = cycle(lines)
    w0 = [x0, y0, i, A0, B0, C0]
    w = odeint(model,w0,ti,args=(la,))
    x,y,z,A,B,C=w[:,0],w[:,1], w[:,2], w[:,3], w[:,4], w[:,5]
    plt.title(r"Prikaz reakcije; $z_0=[1,2,5,10,25,50,100]$ pri $\lambda=1$")
    plt.plot(ti,x,label=r"x, $z_0=%.2f$" %(i), color=plt.cm.gist_rainbow(index/len(number)))
    plt.legend()
    index+=1
index=0
for i in [1,5,10,20,35,50]:
    lines = ["-","--",":"]
    linecycler = cycle(lines)
    w0 = [x0, y0, i, A0, B0, C0]
    w = odeint(model,w0,ti,args=(la,))
    x,y,z,A,B,C=w[:,0],w[:,1], w[:,2], w[:,3], w[:,4], w[:,5]
    plt.title(r"Prikaz reakcije; $z_0=[1,2,5,10,25,50,100]$ pri $\lambda=1$")
    plt.plot(ti,y,label=r"y, $z_0=%.2f$" %(i) , color=plt.cm.cool(index/len(number)))
    plt.legend()
    index+=1
plt.ylabel('Koncentracija')
plt.xlabel('čas')
plt.xscale("log")
plt.legend()
plt.grid()
plt.show()
index=0
number=[1,5,10,20,35,50]
for i in [1,5,10,20,35,50]:
    lines = ["-","--",":"]
    linecycler = cycle(lines)
    w0 = [x0, y0, i, i, B0, C0]
    w = odeint(model,w0,ti,args=(la,))
    x,y,z,A,B,C=w[:,0],w[:,1], w[:,2], w[:,3], w[:,4], w[:,5]
    plt.title(r"Prikaz reakcije; $z_0=[1,2,5,10,25,50,100]$ pri $\lambda=1$ in $z_0/A_0=konst$")
    plt.plot(ti,x,label=r"x, $z_0=%.2f$" %(i), color=plt.cm.gist_rainbow(index/len(number)))
    plt.legend()
    index+=1
index=0
for i in [1,5,10,20,35,50]:
    lines = ["-","--",":"]
    linecycler = cycle(lines)
    w0 = [x0, y0, i, i, B0, C0]
    w = odeint(model,w0,ti,args=(la,))
    x,y,z,A,B,C=w[:,0],w[:,1], w[:,2], w[:,3], w[:,4], w[:,5]
    plt.title(r"Prikaz reakcije; $z_0=[1,2,5,10,25,50,100]$ pri $\lambda=1$ in $z_0/A_0=konst$")
    plt.plot(ti,y,label=r"y, $z_0=%.2f$" %(i) , color=plt.cm.cool(index/len(number)))
    plt.legend()
    index+=1
plt.ylabel('Koncentracija')
plt.xlabel('čas')
plt.xscale("log")
plt.legend()
plt.grid()
plt.show()

index=0
number=[1,5,10,20,35,50]
for i in [1,5,10,20,35,50]:
    lines = ["-","--",":"]
    linecycler = cycle(lines)
    w0 = [x0, y0, z0, A0, B0, C0]
    w = odeint(model,w0,ti,args=(la,))
    x,y,z,A,B,C=w[:,0],w[:,1], w[:,2], w[:,3], w[:,4], w[:,5]
    plt.title(r"Prikaz reakcije za različne $z_0=[1,2,5,10,25,50,100]$ pri $\lambda=1$")
    if index==0 or index==len(number)-1: 
        plt.plot(ti,x,label='x', color=plt.cm.gist_rainbow(index/len(number)))
        plt.plot(ti,y,label="y" , color=plt.cm.cool(index/len(number)))
        #plt.plot(ti,z/i,label='z', color=plt.cm.summer(index/len(number)))
    else:
        plt.plot(ti,x, color=plt.cm.gist_rainbow(index/len(number)))
        plt.plot(ti,y, color=plt.cm.cool(index/len(number)))
        #plt.plot(ti,z/i, color=plt.cm.summer(index/len(number)))
    plt.ylabel('Koncentracija')
    plt.xlabel('čas')
    plt.xscale("log")
    plt.legend()
    plt.grid()
    index+=1
plt.show()
# solve ODE
