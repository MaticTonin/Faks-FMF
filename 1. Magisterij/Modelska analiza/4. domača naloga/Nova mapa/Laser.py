import numpy as np
import scipy.integrate as intg
import matplotlib.pyplot as plt
from scipy.integrate import odeint

n = 4000
l = 300 
def diff_eq(y, p):
    F, A = y

    return np.array([F/p * (A-1), r-p*A*(F+1)])

p = 0.5
r = 4
e =0.001

A0 = r/p 
F0 = 0

z0 = [A0,F0]
t = np.linspace(0,l,n)
N = 10
x = np.linspace(r/p-1-4, r/p-1+4, N)
y = np.linspace(0.25, 2.25, N)

x, y = np.meshgrid(x, y)

u = np.empty((N, N))
v = np.empty((N, N))
plt.subplot(1, 2, 1)
for i in range(N):
    for j in range(N):
        u[i, j], v[i, j] = diff_eq(np.array([x[i, j], y[i, j]]), p)

qq=plt.quiver(x, y, u, v,np.sqrt(u**2+v**2),cmap=plt.cm.coolwarm)
plt.colorbar(qq, cmap=plt.cm.jet)
plt.scatter(r/p-1,1,label="Stacionarna točka")

plt.xlabel(r"$z$")
plt.ylabel(r"$l$")

plt.title("Prikaz stac. točke (r/p-1,1) pri $p=%.2f$ in $r=%.2f$" %(p,r))



x = np.linspace(-4, 4, N)
y = np.linspace(r/p-4, r/p+4, N)

x, y = np.meshgrid(x, y)

u = np.empty((N, N))
v = np.empty((N, N))

for i in range(N):
    for j in range(N):
        u[i, j], v[i, j] = diff_eq(np.array([x[i, j], y[i, j]]), p)
plt.subplot(1, 2, 2)
qq=plt.quiver(x, y, u, v,np.sqrt(u**2+v**2),cmap=plt.cm.coolwarm)
plt.colorbar(qq, cmap=plt.cm.jet)
plt.scatter(0,r/p,label="Stacionarna točka")
plt.scatter(r/p-1,1,label="(r/p-1,1)")

plt.xlabel(r"$A$")
plt.ylabel(r"$F$")

plt.title("Prikaz stac. točke (0,r/p) pri $p=%.2f$ in $r=%.2f$" %(p,r))

plt.show()
plt.close()


import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import random
import pandas as pd
from matplotlib.widgets import Slider, Button


# function that returns dz/dt

n = 10000 #stevilo korakov integracije
l = 10 #čas do kamor simuliramo


def model(z,t,p,r):
    A = z[0]
    F = z[1]
    dAdt = r - p* A *(F + 1)
    dFdt = F/p*(A - 1)
    dzdt = [dAdt,dFdt]
    return dzdt


p = 1
r = 2
e =0.01

A0 = 1
F0 = 1
z0 = [A0,F0]
t = np.linspace(0,l,n)

R=np.linspace(3,3,6)


fig, (ax1,ax2)=plt.subplots(2)
fig.suptitle(r"Prikaz modela laserja za $F_0=%.2f$ in $p=%.2f$"%(F0,p))
for i in R:
    #z0 = [A0,F0]
    z = odeint(model,z0,t, args=(p,i,))
    ax1.plot(t,z[:,0],label='r = %.3f' %(i))
    #z0 = [i,F0]
    #z = odeint(model,z0,t, args=(p,r))
    #ax1.set_title(r"$A_0=%.2f$"%(A0))
    #ax2.plot(t,z[:,0],label='A_0 = %.3f' %(i))
    #ax2.set_title(r"$r=%.2f$"%(r))
ax1.legend()
ax1.grid()
ax2.grid()
ax2.legend()
ax2.set_xlabel(r"t")
ax1.set_ylabel(r"A")
ax2.set_ylabel(r"A")
plt.show()

R=np.linspace(0.5,2.5,6)

fig, (ax1,ax2)=plt.subplots(2)
fig.suptitle(r"Prikaz modela laserja za spreminjanje $A_0$, $F_0=%.2f$, $p=%.2f$ in $r=%.2f$"%(F0,p,r))
for i in R:
    z0 = [i,F0]
    z = odeint(model,z0,t, args=(p,r))
    ax1.plot(t,z[:,0],label=r'$A_0$ = %.3f' %(i))
    ax1.set_title("Fotoni")
    ax2.plot(t,z[:,1],label=r'$A_0$ = %.3f' %(i))
    ax2.set_title("Fotoni")
ax1.legend()
ax1.grid()
ax2.grid()
ax2.legend()
ax2.set_xlabel(r"\tau")
ax1.set_ylabel(r"A")
ax2.set_ylabel(r"A")
plt.show()
import numpy as np
import scipy.integrate as intg
import matplotlib.pyplot as plt

def diff_eq(y, p, r):
    a, f = y

    return np.array([r - p*a * (f+1),  f / p * (a - 1)])

small_F = lambda t, p, r, f0: f0 * np.exp((r/p - 1) * t / p)



y0 = z0
N = n
t0 = l




n = 10000 #stevilo korakov integracije
l = 10 #čas do kamor simuliramo


def model(z,t,p,r):
    A = z[0]
    F = z[1]
    dAdt = r - p* A *(F + 1)
    dFdt = F/p*(A - 1)
    dzdt = [dAdt,dFdt]
    return dzdt


p = 1
r = 1
e =0.01

A0 = 1
F0 = 1
z0 = [A0,F0]
t = np.linspace(0,l,n)

R=np.linspace(0.1,3,6)


fig, (ax1,ax2)=plt.subplots(2)
fig.suptitle(r"Prikaz modela laserja za $F_0=%.2f$ in $p=%.2f$"%(F0,p))
for i in R:
    y0 = [A0,F0]
    sol = intg.solve_ivp(lambda t, y: diff_eq(y, p, i), (0, t0), y0, t_eval=np.linspace(0, t0, N), method="DOP853", max_step=0.01)
    ax1.plot(sol.t,sol.y[0],label='r = %.3f' %(i))
    y0 = [i,F0]
    sol = intg.solve_ivp(lambda t, y: diff_eq(y, p, r), (0, t0), y0, t_eval=np.linspace(0, t0, N), method="DOP853", max_step=0.01)
    ax2.plot(sol.t,sol.y[0],label=r"A_0 = %.3f" %(i))
    ax1.set_title(r"$A_0=%.2f$"%(A0))
    ax2.set_title(r"$r=%.2f$"%(r))
ax1.legend()
ax1.grid()
ax2.grid()
ax2.legend()
ax2.set_xlabel(r"t")
ax1.set_ylabel(r"A")
ax2.set_ylabel(r"A")
plt.show()

fig, (ax1,ax2)=plt.subplots(2)
fig.suptitle(r"Prikaz modela laserja za $F_0=%.2f$ in $p=%.2f$"%(F0,p))
fig1, (ax11,ax22)=plt.subplots(2)
fig1.suptitle(r"Prikaz modela laserja za $F_0=%.2f$ in $p=%.2f$"%(F0,p))
for i in R:
    y0 = [A0,F0]
    sol = intg.solve_ivp(lambda t, y: diff_eq(y, p, i), (0, t0), y0, t_eval=np.linspace(0, t0, N), method="DOP853", max_step=0.01)
    u[i, j], v[i, j] = diff_eq(np.array([x[i, j], y[i, j]]), p, r)
    ax1.plot(sol.y[0],sol.y[1],label='r = %.3f' %(i))
    y0 = [i,F0]
    sol = intg.solve_ivp(lambda t, y: diff_eq(y, p, r), (0, t0), y0, t_eval=np.linspace(0, t0, N), method="DOP853", max_step=0.01)
    u[i, j], v[i, j] = diff_eq(np.array([x[i, j], y[i, j]]), p, r)
    ax2.plot(sol.y[0],sol.y[1],label=r"A_0 = %.3f" %(i))
    ax1.set_title(r"$A_0=%.2f$"%(A0))
    ax2.set_title(r"$r=%.2f$"%(r))
ax1.legend()
ax1.grid()
ax2.grid()
ax2.legend()
ax2.set_xlabel(r"A")
ax1.set_ylabel(r"F")
ax2.set_ylabel(r"F")
plt.show()

