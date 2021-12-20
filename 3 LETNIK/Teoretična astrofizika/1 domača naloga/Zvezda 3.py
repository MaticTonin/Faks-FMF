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
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Zvezda_n=3.txt")
M, R, T, RHO, P, L, X, Y, HE, C12, N14, O16= np.loadtxt(my_file, dtype=float, delimiter = "\t", unpack ="True")  
N=3
n=N
def f(icons, xi):
    return [icons[1], -icons[0]**n-2/xi*icons[1]]

icons=[1,0]
xi=np.linspace(0.01, 10, 10000)
main_f=integrate.odeint(f, icons, xi)

#Prva ničla 
theta=np.transpose(main_f)[0]
odvod=np.transpose(main_f)[1]
index=0
for i in range(len(theta)):
    if theta[i]>0:
        index=i
    else:
        break
#Določanje xi
xi0=(xi[index]+xi[index+1])/2
odvod0=odvod[index]
Masa=-xi0**2*odvod0
n=N
#Gostota
rho=[]
for i in range(len(theta)):
    rho.append(theta[i]**n)
plt.plot(xi/6.895,rho,'-',color="blue", label=r"$\rho(\xi)$")
plt.plot(R/max(R),RHO/max(RHO),'-',color="red", label=r"$\rho(\xi) podatki$")
plt.title(r'Prikaz približkov za $\rho$, $n=$' +str(n))
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\rho$")
plt.axis([0,1,0,1])
plt.legend()
plt.show()

#Masa
m=[]
for i in range(len(theta)):
    m.append(-xi[i]**2*odvod[i])

plt.plot(xi/6.895,m/max(m),'-',color="blue", label=r"$m(\xi)$")
plt.plot(R/max(R),M/max(M),'-',color="red", label=r"$M(\xi) podatki$")
plt.title(r'Prikaz približkov za $m$, $n=$' +str(n))
plt.xlabel(r"$\xi$")
plt.ylabel(r"$m$")
plt.axis([0,1,0,1])
plt.legend()
plt.show()  
#Tlak
p=[]
for i in range(len(theta)):
    p.append(theta[i]**(1+n))

plt.plot(xi/6.895,p,'-',color="blue", label=r"$p(\xi)$")
plt.plot(R/max(R),P/max(P),'-',color="red", label=r"$P(\xi) podatki$")
plt.title(r'Prikaz približkov za $p$, $n=$' +str(n))
plt.xlabel(r"$\xi$")
plt.ylabel(r"$p$")
plt.axis([0,1,0,1])
plt.legend()
plt.show() 

#Temperatura
t=[]
for i in range(len(theta)):
    t.append(theta[i])

plt.plot(xi/6.895,t,'-',color="blue", label=r"$T(\xi)$")
plt.plot(R/max(R),T/max(T),'-',color="red", label=r"$T(\xi) podatki$")
plt.title(r'Prikaz približkov za $T$, $n=$' +str(n))
plt.xlabel(r"$\xi$")
plt.ylabel(r"$T$")
plt.axis([0,1,0,1])
plt.legend()
plt.show() 

plt.plot(xi/6.895,m/max(m),'-',color="blue", label=r"$m(\xi)$")
plt.plot(xi/6.895,rho,'-',color="yellow", label=r"$\rho(\xi)$")
plt.plot(xi/6.895,p,'-',color="green", label=r"$p(\xi)$")
plt.plot(xi/6.895,t,'-',color="red", label=r"$T(\xi)$")
plt.title(r'Prikaz približkov za vse količine, $n=$' +str(n))
plt.xlabel(r"$\xi$")
plt.ylabel(r"$X$")
plt.axis([0,1,0,1])
plt.legend()
plt.show() 