import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os

S=0.5
rho=1.3
c=1
m=80
g=9.8
t=251/100
tmax2=2*t
tmax3=3*t
h=(t-0)/50
h2=(tmax2-0)/50
h3=(tmax3-0)/50
x0=1500
v0=0
t0=0
B=1/2*c*rho*S/m

def dvdt(t,v):
    return(g-B*v**2)

def RK3(t0,v0,t,h):
    n=(int)((t-t0)/h)
    v=v0
    h1=1500
    for i in range(1,n+1):
         k1 = h * dvdt(t0,v);
         k2 = h * dvdt(t0 + 0.5*h, v+ 0.5*k1);
         k3 = h * dvdt(t0 + 0.5*h, v + 0.5*k2);
         k4 = h * dvdt(t0 + h, v + k3);
         v = v + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6;
         h1=h1-v*h
    return h1
print(RK3(t0,v0,t,h))

def RK3v(t0,v0,t,h):
    n=(int)((t-t0)/h)
    v=v0
    h1=1500
    for i in range(1,n+1):
         k1 = h * dvdt(t0,v);
         k2 = h * dvdt(t0 + 0.5*h, v+ 0.5*k1);
         k3 = h * dvdt(t0 + 0.5*h, v + 0.5*k2);
         k4 = h * dvdt(t0 + h, v + k3);
         v = v + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6;
         h1=h1-v*h
    return v
print(RK3v(t0,v0,tmax2,h2))
def RK3a(t0,v0,t,h):
    n=(int)((t-t0)/h)
    v=v0
    h1=1500
    for i in range(1,n+1):
         k1 = h * dvdt(t0,v);
         k2 = h * dvdt(t0 + 0.5*h, v+ 0.5*k1);
         k3 = h * dvdt(t0 + 0.5*h, v + 0.5*k2);
         k4 = h * dvdt(t0 + h, v + k3);
         v = v + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6;
         a=g-B*v**2
    return a
print(RK3a(t0,v0,tmax3,h3))
