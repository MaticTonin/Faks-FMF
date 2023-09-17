from re import A
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from itertools import cycle

# function that returns dz/dt
n = 250000 #stevilo korakov integracije
l = 300 #čas do kamor simuliramo
p = 1
q = 100

def model(w,ti,q):
    x = w[0]
    y = w[1]
    z = w[2]
    
    dxdt = 2*q*y*z**2-2*p*x**2
    dydt = p*x**2-q*y*z**2
    dzdt = -2*q*y*z**2

    dwdt = [dxdt,dydt, dzdt]
    return dwdt

# initial condition
x0 = 1 #HI-
y0 = 0  #I2
z0 = 1  #SO

w0 = [x0, y0, z0]

# time points
ti = np.linspace(0,l,n)
IC = np.linspace(0,10,100)
lines = ["-","--",":"]
linecycler = cycle(lines)

# solve ODE
for i in IC:
    w0 = [x0, y0, i]
    w = odeint(model,w0,ti,args=(q,))
#    tip = next(linecycler)
## plot results
#    plt.plot(ti,w[:,0],'r', ls = tip, label='I-, q = %.0f' %(i))
#    plt.plot(ti,w[:,1],'b', ls = tip, label='I_2, q = %.0f' %(i))
#    plt.plot(ti,w[:,2],'g', ls = tip, label='S_2O_3, q = %.0f' %(i))

    š = 0
    for j in w[:,1]:
        if j >= 0.1*0.5*x0:
            t1 = ti[š]
            break
        š = š+1

    š = 0
    for j in w[:,1]:
        if j >= 0.75*0.5*x0:
            t2 = ti[š] - t1
            break
        š = š+1

    plt.scatter(i,t1, marker='.',color ='r',linewidths=1)
    plt.scatter(i,t2, marker='.',color ='b',linewidths=1)

plt.scatter(10,t1, marker='.',color ='r',linewidths=1, label= 't1, q = 100')
plt.scatter(10,t2, marker='.',color ='b',linewidths=1, label = 't2, q = 100')



q = 10
# solve ODE
for i in IC:
    w0 = [x0, y0, i]
    w = odeint(model,w0,ti,args=(q,))
#    tip = next(linecycler)
## plot results
#    plt.plot(ti,w[:,0],'r', ls = tip, label='I-, q = %.0f' %(i))
#    plt.plot(ti,w[:,1],'b', ls = tip, label='I_2, q = %.0f' %(i))
#    plt.plot(ti,w[:,2],'g', ls = tip, label='S_2O_3, q = %.0f' %(i))

    š = 0
    for j in w[:,1]:
        if j >= 0.1*0.5*x0:
            t1 = ti[š]
            break
        š = š+1

    š = 0
    for j in w[:,1]:
        if j >= 0.75*0.5*x0:
            t2 = ti[š] - t1
            break
        š = š+1

    plt.scatter(i,t1, marker='_', color ='r',linewidths=1)
    plt.scatter(i,t2, marker='_',color ='b',linewidths=1)

plt.scatter(10,t1, marker='_',color ='r',linewidths=1, label= 't1, q = 10')
plt.scatter(10,t2, marker='_',color ='b',linewidths=1, label = 't2, q = 10')




q = 1
# solve ODE
for i in IC:
    w0 = [x0, y0, i]
    w = odeint(model,w0,ti,args=(q,))
#    tip = next(linecycler)
## plot results
#    plt.plot(ti,w[:,0],'r', ls = tip, label='I-, q = %.0f' %(i))
#    plt.plot(ti,w[:,1],'b', ls = tip, label='I_2, q = %.0f' %(i))
#    plt.plot(ti,w[:,2],'g', ls = tip, label='S_2O_3, q = %.0f' %(i))

    š = 0
    for j in w[:,1]:
        if j >= 0.1*0.5*x0:
            t1 = ti[š]
            break
        š = š+1

    š = 0
    for j in w[:,1]:
        if j >= 0.75*0.5*x0:
            t2 = ti[š] - t1
            break
        š = š+1

    plt.scatter(i,t1, marker='x', color ='r',linewidths=1)
    plt.scatter(i,t2, marker='x',color ='b',linewidths=1)

plt.scatter(10,t1, marker='x',color ='r',linewidths=1, label= 't1, q = 1')
plt.scatter(10,t2, marker='x',color ='b',linewidths=1, label = 't2, q = 1')



plt.ylabel('t')
plt.figtext(0.5, 0.9, "I-(0) = {}, I_2(0) = {}".format(x0, y0))
plt.xlabel('S2O3(t=0)')
plt.legend(loc='best')

plt.show()


#plt.plot(ti,w[:,0],'r',label='H2, H2_0 = %.0f' %(u0))
#plt.plot(ti,w[:,1]+w[:,0],'b',label='Br2, Br2_0 = %.0f' %(v0))
#plt.plot(ti,w[:,2]+w[:,1]+w[:,0],'g',label='HBr, HBr_0 = %.0f' %(x0))
#plt.plot(ti,w[:,3]+w[:,2]+w[:,1]+w[:,0],'k',label='H, H_0 = %.0f' %(y0))
#plt.plot(ti,w[:,4]+w[:,3]+w[:,2]+w[:,1]+w[:,0],'orange',label='Br, Br_0 = %.0f' %(z0))
#
#plt.fill_between(ti,w[:,4]+w[:,3]+w[:,2]+w[:,1]+w[:,0],color ='orange')
#plt.fill_between(ti,w[:,3]+w[:,2]+w[:,1]+w[:,0],color ='k')
#plt.fill_between(ti,w[:,2]+w[:,1]+w[:,0],color ='g')
#plt.fill_between(ti,w[:,1]+w[:,0],color ='b')
#plt.fill_between(ti,w[:,0],color ='r')
#
#
#plt.ylabel('koncentracija snovi')
#plt.text(1.5, 1.5, "p = {}, q = {}, r = {}, s = {}, t = {}".format(p, q, r, s, t))
#plt.xlabel('čas')
#plt.legend(loc='best')
#
#plt.show()
