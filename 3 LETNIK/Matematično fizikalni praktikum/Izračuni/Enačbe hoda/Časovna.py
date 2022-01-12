import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import time
N=1000 #KORAKI
lenght_T= 100 # Dolčina časa
h=lenght_T/N
k=0.1
y_zunaj=-5
y_0=21

start1 = time.time()
def Derivate(y_0):
    value=-k*(y_0-y_zunaj)
    return value

"""Eulerjeva metoda ___
kjer aproksimiramo odvod kot  y_{n+1}=y(x)+h dy/dx_x """

def Derivate(y_0):
    value=-k*(y_0-y_zunaj)
    return value

def Eulerjeva(y_0, h):
    y_1=y_0+h*Derivate(y_0)
    return y_1

t=np.linspace(0,lenght_T, N)
def izdelava_Euler(N,h,y_0):
    Euler=[]
    for i in range(N):
        if i==0:
            Euler.append(y_0)
            if i>=1:
                Euler.append(Eulerjeva(Euler[i-1],h))
    return Euler

"""Simetrizirana Eulerjeva metoda"""

def Simetric_Euler(y_minus,y_middle, h):
    y_plus=y_minus+2*h*Derivate(y_middle)
    return y_plus

def izdelava_Simetric(y_0, h, N):
    Simetric=[]
    for i in range(N):
        if i==0:
            Simetric.append(y_0)
        if i==1:
            Simetric.append(Eulerjeva(Simetric[i-1], h))
        if i>=2:
            Simetric.append(Simetric_Euler(Simetric[i-2],Simetric[i-1], h))
    return Simetric

"""Heunova metoda"""
def Heun_metod(y_0, h):
    y_delta=y_0+h*Derivate(y_0)
    y= y_0+ h/2 *(Derivate(y_0)+Derivate(y_delta))
    return y
def izdelava_Heun(y_0,h, N):
    Heun=[]
    for i in range(N):
        if i==0:
            Heun.append(y_0)
        if i>=1:
            Heun.append(Heun_metod(Heun[i-1],h))
    return Heun
"""Midpoint metoda"""
def Midpoint_method(y_0, h):
    k_1=Derivate(y_0)
    k_2=Derivate(y_0+1/2*h*k_1)
    y_1=y_0+h*k_2
    return y_1
def izdelava_Midpoint(y_0, h, N):
    Midpoint=[]
    for i in range(N):
        if i==0:
            Midpoint.append(y_0)
        if i>=1:
            Midpoint.append(Midpoint_method(Midpoint[i-1],h))
    return Midpoint

"""Runge-Kutta metoda s 4 koraki"""
def RK4_method(y_0, h):
    k_1=Derivate(y_0)
    k_2=Derivate(y_0+h/2*k_1)
    k_3=Derivate(y_0+h/2*k_2)
    k_4=Derivate(y_0+h*k_3)
    y_1=y_0+h/6*(k_1+2*k_2+2*k_3+k_4)
    return y_1
def izdelava_RK4(y_0, h, N):
    RK4=[]
    for i in range(N):
        if i==0:
            RK4.append(y_0)
        if i>=1:
            RK4.append(RK4_method(RK4[i-1],h))
    return RK4
end1 = time.time()
"""Dejanska funkcija """
def main_function(t):
    y=y_zunaj+np.exp(-k*t)*(y_0-y_zunaj)
    return y
y_main=main_function(t)
Euler=izdelava_Euler(N,h,y_0)
Simetric=izdelava_Simetric(y_0,h,N)
Heun=izdelava_Heun(y_0,h,N)
Midpoint=izdelava_Midpoint(y_0,h,N)
RK4=izdelava_RK4(y_0, h, N)
plt.plot(t,Euler,'-',color="blue", label="Eulerjeva")
plt.plot(t,Simetric,'-',color="green",label="Eulerjeva simetrična")
plt.plot(t,y_main,'-',color="red", label="Dejanska funkcija")
plt.plot(t,Heun,'-',color="purple", label="Heunova")
plt.plot(t,Midpoint,'-',color="orange", label="Midpoint")
plt.plot(t,RK4,'-',color="pink", label="Runge Kutta 4")
plt.title('Prikaz približkov')
plt.xlabel("t")
plt.ylabel("T(t)")
plt.legend()
plt.show()


def Error(Y,value):
    Error=[]
    for i in range(len(Y)):
        Error.append(abs(Y[i]-value[i]))
    return Error

plt.plot(t,Error(Euler,y_main),'.',color="blue", label="Eulerjeva")
plt.plot(t,Error(Simetric,y_main),'.',color="green",label="Eulerjeva simetrična")
plt.plot(t,Error(Heun,y_main),'.',color="purple", label="Heunova")
plt.plot(t,Error(Midpoint,y_main),'.',color="orange", label="Midpoint")
plt.plot(t,Error(RK4,y_main),'-',color="pink", label="Runge Kutta 4")
plt.title('Prikaz približkov')
plt.xlabel("t")
plt.ylabel("Error(t)")
plt.legend()
plt.show()
