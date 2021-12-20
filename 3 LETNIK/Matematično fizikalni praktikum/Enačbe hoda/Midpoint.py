import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import time

k=0.1
y_zunaj=-5
y_0=21

def Derivate(y_0):
    value=-k*(y_0-y_zunaj)
    return value
start1 = time.time()
def Midpoint_method(y_0, h):
    k_1=Derivate(y_0)
    k_2=Derivate(y_0+1/2*h*k_1)
    y_1=y_0+h*k_2
    return y_1
def izdelava_Midpoint(N, h, y_0):
    Midpoint=[]
    for i in range(N):
        if i==0:
            Midpoint.append(y_0)
        if i>=1:
            Midpoint.append(Midpoint_method(Midpoint[i-1],h))
    return Midpoint
"""Dejanska funkcija """
def main_function(t):
    y=y_zunaj+np.exp(-k*t)*(y_0-y_zunaj)
    return y

N=1000 #KORAKI
lenght_T= 70 # Dolčina časa
h=lenght_T/N
t=np.linspace(0,lenght_T, N)
y_main=main_function(t)

N=10 #KORAKI
h=lenght_T/N
t10=np.linspace(0,lenght_T, N)
Euler10=izdelava_Midpoint(N,h,y_0)
y_main10=main_function(t10)

N=20 #KORAKI
h=lenght_T/N
t20=np.linspace(0,lenght_T, N)
Euler20=izdelava_Midpoint(N,h,y_0)

N=50 #KORAKI
h=lenght_T/N
t50=np.linspace(0,lenght_T, N)
Euler50=izdelava_Midpoint(N,h,y_0)

N=100 #KORAKI
h=lenght_T/N
t100=np.linspace(0,lenght_T, N)
Euler100=izdelava_Midpoint(N,h,y_0)

N=300 #KORAKI
h=lenght_T/N
t300=np.linspace(0,lenght_T, N)
Euler300=izdelava_Midpoint(N,h,y_0)

N=1000 #KORAKI
h=lenght_T/N
t1000=np.linspace(0,lenght_T, N)
Euler1000=izdelava_Midpoint(N,h,y_0)

N=5000 #KORAKI
h=lenght_T/N
t5000=np.linspace(0,lenght_T, N)
Euler5000=izdelava_Midpoint(N,h,y_0)

N=5000 #KORAKI
h=lenght_T/N
t5000=np.linspace(0,lenght_T, N)
Euler5000=izdelava_Midpoint(N,h,y_0)

N=10000 #KORAKI
h=lenght_T/N
t10000=np.linspace(0,lenght_T, N)
Euler10000=izdelava_Midpoint(N,h,y_0)

N=50000 #KORAKI
h=lenght_T/N
t50000=np.linspace(0,lenght_T, N)
Euler50000=izdelava_Midpoint(N,h,y_0)

N=100000 #KORAKI
h=lenght_T/N
t100000=np.linspace(0,lenght_T, N)
Euler100000=izdelava_Midpoint(N,h,y_0)

N=500000 #KORAKI
h=lenght_T/N
t500000=np.linspace(0,lenght_T, N)
Euler500000=izdelava_Midpoint(N,h,y_0)

N=1000000 #KORAKI
h=lenght_T/N
t1000000=np.linspace(0,lenght_T, N)
Euler1000000=izdelava_Midpoint(N,h,y_0)

plt.plot(t10,Euler10,'-', label="N=10")
plt.plot(t20,Euler20,'-', label="N=20")
plt.plot(t50,Euler50,'-', label="N=50")
plt.plot(t100,Euler100,'-', label="N=100")
plt.plot(t300,Euler300,'-', label="N=300")
plt.plot(t1000,Euler1000,'-', label="N=1000")
plt.plot(t,y_main,'-', label="Dejanska funkcija")
plt.title('Prikaz približkov Midpoint funkcije')
plt.xlabel("t")
plt.ylabel("T(t)")
plt.legend()
plt.show()

plt.plot(t100,Euler100,'-', label="N=100")
plt.plot(t300,Euler300,'-', label="N=300")
plt.plot(t1000,Euler1000,'-', label="N=1000")
plt.plot(t,y_main,'-', label="Dejanska funkcija")
plt.title('Prikaz približkov Midpoint funkcije')
plt.xlabel("t")
plt.ylabel("T(t)")
plt.legend()
plt.show()


plt.plot(t10,Euler10,'-', label="N=10")
plt.plot(t20,Euler20,'-', label="N=20")
plt.plot(t50,Euler50,'-', label="N=50")
plt.plot(t100,Euler100,'-', label="N=100")
plt.plot(t300,Euler300,'-', label="N=300")
plt.plot(t1000,Euler1000,'-', label="N=1000")
plt.plot(t,y_main,'-', label="Dejanska funkcija")
plt.title('Prikaz približkov Midpoint funkcije, približana')
plt.xlabel("t")
plt.ylabel("T(t)")
plt.axis([5,20, -5,10])
plt.legend()
plt.show()


def Error(Y,value):
    Error=[]
    for i in range(len(Y)):
        Error.append(abs(Y[i]-value[i]))
    return Error
y_main10=main_function(t10)
y_main20=main_function(t20)
y_main50=main_function(t50)
y_main100=main_function(t100)
y_main300=main_function(t300)
y_main1000=main_function(t1000)
y_main5000=main_function(t5000)
y_main10000=main_function(t10000)
y_main50000=main_function(t50000)
y_main100000=main_function(t100000)
y_main500000=main_function(t500000)
y_main1000000=main_function(t1000000)

plt.plot(t10,Error(Euler10,y_main10),'.', label="N=10")
plt.plot(t20,Error(Euler20,y_main20),'.', label="N=20")
plt.plot(t50,Error(Euler50,y_main50),'.', label="N=50")
plt.plot(t100,Error(Euler100,y_main100),'.', label="N=100")
plt.plot(t300,Error(Euler300,y_main300),'.', label="N=300")
plt.plot(t1000,Error(Euler1000,y_main1000),'.', label="N=1000")
plt.title('Prikaz približkov za Midpoint metodo')
plt.xlabel("t")
plt.ylabel("Error(t)")
plt.legend()
plt.show()

plt.plot(t1000,Error(Euler1000,y_main1000),'.', label="N=1000")
plt.plot(t5000,Error(Euler5000,y_main5000),'.', label="N=5000")
plt.plot(t10000,Error(Euler10000,y_main10000),'.', label="N=10000")
plt.plot(t50000,Error(Euler50000,y_main50000),'.', label="N=50000")
plt.plot(t100000,Error(Euler100000,y_main100000),'.', label="N=100000")
plt.title('Prikaz približkov za Midpoint metodo, več korakov')
plt.xlabel("t")
plt.ylabel("Error(t)")
plt.legend()
plt.show()

plt.plot(t50000,Error(Euler50000,y_main50000),'.', label="N=50000")
plt.plot(t100000,Error(Euler100000,y_main100000),'.', label="N=100000")
plt.plot(t500000,Error(Euler500000,y_main500000),'.', label="N=500000")
plt.plot(t1000000,Error(Euler1000000,y_main1000000),'.', label="N=1000000")
plt.title('Prikaz približkov za Midpoint metodo, več korakov')
plt.xlabel("t")
plt.ylabel("Error(t)")
plt.legend()
plt.show()
