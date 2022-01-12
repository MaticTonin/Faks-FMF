
from numpy import sin, cos, pi, array
import numpy as np
import scipy.integrate as integrate
import matplotlib
import scipy
import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import time
import os
from diffeq_2 import *
#!/usr/bin/env python
import numpy
def pefrl( f, x0, v0, t ):
    """Position Extended Forest-Ruth Like 4th order symplectic method by Omelyan et al.

    USAGE:
        (x,v) = varlet(f, x0, v0, t)

    INPUT:
        f     - function of x and t equal to d^2x/dt^2.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s) of x.  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        v0    - the initial condition(s) of v=dx/dt.  Specifies the value of v when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values for x corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
        v     - NumPy array containing solution values for v=dx/dt corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.

    NOTES:
        This function uses the Omelyan et al (symplectic) method
        method to solve the initial value problem

            dx^2
            -- = f(x),     x(t(1)) = x0  v(t(1)) = v0
            dt^2

        at the t values stored in the t array (so the interval of solution is
        [t[0], t[N-1]].

    """

    xsi=0.1786178958448091
    lam=-0.21234183100626054
    chi=-0.6626458266981849e-1
    n = len( t )
    x = numpy.array( [ x0 ] * n )
    v = numpy.array( [ v0 ] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        y=numpy.copy(x[i])
        w=numpy.copy(v[i])
        y += xsi*h*w
        w += (1-2*lam)*(h/2)*f(y)
        y += chi*h*w
        w += lam*h*f(y)
        y += (1-2*(chi+xsi))*h*w
        w += lam*h*f(y)
        y += chi*h*w
        w += (1-2*lam)*(h/2)*f(y)
        y += xsi*h*w
        x[i+1]=numpy.copy(y)
        v[i+1]=numpy.copy(w)

    return numpy.array([x,v])

omega2=1

#-----------------------------------------------------------------------------
# linear force (e.g. spring pendulum)
def forcel(theta):
    return -omega2*np.sin(theta) # sin(y)

#-----------------------------------------------------------------------------
# linear force (e.g. spring pendulum)
def energy(y,v):
    return (omega2*y*y+v*v)/2.

#-----------------------------------------------------------------------------
# a simple pendulum y''= F(y) , state = (y,v)
def pendulum(state,t): 
    dydt=np.zeros_like(state)
    dydt[0]=state[1] # x' = v 
    dydt[1]=forcel(state[0])  # v' = F(x)
    return dydt

#-----------------------------------------------------------------------------
# a simple pendulum

def Error(Y,value):
    Error=[]
    for i in range(len(Y)):
        Error.append(abs(Y[i]-value[i]))
    return Error
theta0=1.
vtheta0=0.
dt=0.0001
t= np.arange(0.0, 100, dt)
iconds=np.array([theta0,vtheta0])
main_f=integrate.odeint(pendulum,iconds,t)
#dt=0.1

dt =  0.5
t05= np.arange(0.0, 100, dt)
#initial conditions
Verle05_theta=pefrl(forcel,theta0,vtheta0,t05)
main_f05=integrate.odeint(pendulum,iconds,t05)
Error05_Velrel_theta=Verle05_theta[0,:]-main_f05[:,0]

dt =  0.2
t02 = np.arange(0.0, 100, dt)
#initial conditions
Verle02_theta=pefrl(forcel,theta0,vtheta0,t02)
main_f02=integrate.odeint(pendulum,iconds,t02)
Error02_Velrel_theta=Verle02_theta[0,:]-main_f02[:,0]


dt =  0.1
t01 = np.arange(0.0, 100, dt)
#initial conditions
Verle01_theta=pefrl(forcel,theta0,vtheta0,t01)
main_f01=integrate.odeint(pendulum,iconds,t01)
Error01_Velrel_theta=Verle01_theta[0,:]-main_f01[:,0]

dt =  0.05
t005 = np.arange(0.0, 100, dt)
#initial conditions
Verle005_theta=pefrl(forcel,theta0,vtheta0,t005)
main_f005=integrate.odeint(pendulum,iconds,t005)
Error005_Velrel_theta=Verle005_theta[0,:]-main_f005[:,0]

dt =  0.01
t001 = np.arange(0.0, 100, dt)
#initial conditions
Verle001_theta=pefrl(forcel,theta0,vtheta0,t001)
main_f001=integrate.odeint(pendulum,iconds,t001)
Error001_Velrel_theta=Verle001_theta[0,:]-main_f001[:,0]

dt =  0.00005
t0005 = np.arange(0.0, 100, dt)
#initial conditions
Verle0005_theta=pefrl(forcel,theta0,vtheta0,t0005)
main_f0005=integrate.odeint(pendulum,iconds,t0005)
Error0005_Velrel_theta=Verle0005_theta[0,:]-main_f0005[:,0]

dt =  0.0001
t0001 = np.arange(0.0, 100, dt)
#initial conditions
Verle0001_theta=pefrl(forcel,theta0,vtheta0,t0001)
main_f0001=integrate.odeint(pendulum,iconds,t0001)
Error0001_Velrel_theta=Verle0001_theta[0,:]-main_f0001[:,0]

plt.plot(t05,Verle05_theta[0, :],'-', label="dt=0.5")
plt.plot(t02,Verle02_theta[0, :],'-', label="dt=0.2")
plt.plot(t01,Verle01_theta[0, :],'-', label="dt=0.1")
plt.plot(t005,Verle005_theta[0, :],'-', label="dt=0.05")
plt.plot(t001,Verle001_theta[0, :],'-', label="dt=0.01")
plt.plot(t0005,Verle0005_theta[0, :],'-', label="dt=0.005")
plt.plot(t0001,Verle0001_theta[0, :],'-', label="dt=0.001")
plt.plot(t,main_f[:,0],'-', label="Main f")
plt.title('Prikaz približkov Pefrl za $\u03B8 (t)$')
plt.xlabel("t")
plt.ylabel("$\u03B8 (t)$ ")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


plt.plot(t05,Verle05_theta[1, :],'-', label="dt=0.5")
plt.plot(t02,Verle02_theta[1, :],'-', label="dt=0.2")
plt.plot(t01,Verle01_theta[1, :],'-', label="dt=0.1")
plt.plot(t005,Verle005_theta[1, :],'-', label="dt=0.05")
plt.plot(t001,Verle001_theta[1, :],'-', label="dt=0.01")
plt.plot(t0005,Verle0005_theta[1, :],'-', label="dt=0.005")
plt.plot(t0001,Verle0001_theta[1, :],'-', label="dt=0.001")
plt.plot(t,main_f[:,1],'-', label="Main f")
plt.title('Prikaz približkov Pefrl za $\dot{\u03B8} (t)$')
plt.xlabel("t")
plt.ylabel("$\u03B8 (t)$ ")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


#plt.plot(t05,Error05_Velrel_theta,'-', label="dt=0.5")
plt.plot(t02,Error02_Velrel_theta,'-', label="dt=0.2")
plt.plot(t01,Error01_Velrel_theta,'-', label="dt=0.1")
plt.plot(t005,Error005_Velrel_theta,'-', label="dt=0.05")
plt.plot(t001,Error001_Velrel_theta,'-', label="dt=0.01")
plt.plot(t0005,Error0005_Velrel_theta,'-', label="dt=0.005")
plt.plot(t0001,Error0001_Velrel_theta,'-', label="dt=0.001")
plt.title('Prikaz napake Pefrl za $\u03B8 (t)$')
plt.xlabel("t")
plt.ylabel("$\u03B8 (t)$ ")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()