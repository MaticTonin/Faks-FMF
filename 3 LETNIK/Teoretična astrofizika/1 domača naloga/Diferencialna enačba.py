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
N=3
n=N
def f(icons, xi):
    return [icons[1], -icons[0]**n-2/xi*icons[1]]
icons=[1,0]
xi=np.linspace(0.01, 10, 1000)
main_f=integrate.odeint(f, icons, xi)

#Prva ničla 
theta=np.transpose(main_f)[0]
odvod=np.transpose(main_f)[1]
index=0
#\u03BE=xi
#\u03B8= theta
plt.plot(xi,main_f[:,0],'-',color="red", label=r"$\theta(\xi)$")
plt.plot(xi,main_f[:,1],'-',color="blue", label=r"$\frac{d\theta}{d\xi}$")
plt.title('Prikaz približkov za $\u03B8$ in $\dot{\u03B8}$, $n=$'+str(n))
plt.xlabel("$\u03BE$")
plt.ylabel("$\u03B8$ and $\dot{\u03B8}$")
plt.legend()
plt.show()
#Primerjave N
n=1
icons=[1,0]
xi=np.linspace(0.01, 10, 1000)

def f05(icons, xi):
    return [icons[1], -icons[0]**(10)-2/xi*icons[1]]
main_f05=integrate.odeint(f05, icons, xi)

def f1(icons, xi):
    return [icons[1], -icons[0]**1-2/xi*icons[1]]
main_f1=integrate.odeint(f1, icons, xi)

def f2(icons, xi):
    return [icons[1], -icons[0]**2-2/xi*icons[1]]
main_f2=integrate.odeint(f2, icons, xi)

def f3(icons, xi):
    return [icons[1], -icons[0]**3-2/xi*icons[1]]
main_f3=integrate.odeint(f3, icons, xi)

def f4(icons, xi):
    return [icons[1], -icons[0]**4-2/xi*icons[1]]
main_f4=integrate.odeint(f4, icons, xi)

def f5(icons, xi):
    return [icons[1], -icons[0]**5-2/xi*icons[1]]
main_f5=integrate.odeint(f5, icons, xi)

plt.plot(xi,main_f1[:,0],'-',color="red", label=r"$n=1$")
plt.plot(xi,main_f2[:,0],'-',color="blue", label=r"$n=2$")
plt.plot(xi,main_f3[:,0],'-',color="green", label=r"$n=3$")
plt.plot(xi,main_f4[:,0],'-',color="black", label=r"$n=$4")
plt.plot(xi,main_f5[:,0],'-',color="yellow", label=r"$n=5$")
plt.plot(xi,main_f05[:,0],'-',color="pink", label=r"$n=10$")
plt.title('Prikaz približkov za $\u03B8$ za nekaj n')
plt.xlabel("$\u03BE$")
plt.ylabel("$\u03B8$")
plt.legend()
plt.show()

plt.plot(xi,main_f1[:,1],'-',color="red", label=r"$n=1$")
plt.plot(xi,main_f2[:,1],'-',color="blue", label=r"$n=2$")
plt.plot(xi,main_f3[:,1],'-',color="green", label=r"$n=3$")
plt.plot(xi,main_f4[:,1],'-',color="black", label=r"$n=$4")
plt.plot(xi,main_f5[:,1],'-',color="yellow", label=r"$n=5$")
plt.plot(xi,main_f05[:,1],'-',color="pink", label=r"$n=10$")
plt.title('Prikaz približkov za $\dot{\u03B8}$ za nekaj n')
plt.xlabel("$\u03BE$")
plt.ylabel("$\dot{\u03B8}$")
plt.legend()
plt.show()

n=N
#Gostota
rho=[]
for i in range(len(theta)):
    rho.append(theta[i]**n)
plt.plot(xi,rho,'-',color="blue", label=r"$\rho(\xi)$")
plt.title(r'Prikaz približkov za $\rho$, $n=$' +str(n))
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\rho$")
plt.legend()
plt.show()

#Masa
m=[]
for i in range(len(theta)):
   m.append(-xi[i]**2*odvod[i])

plt.plot(xi,m,'-',color="blue", label=r"$m(\xi)$")
plt.title(r'Prikaz približkov za $m$, $n=$' +str(n))
plt.xlabel(r"$\xi$")
plt.ylabel(r"$m$")
plt.legend()
plt.show()  

#Tlak
p=[]
for i in range(len(theta)):
    p.append(theta[i]**(1+n))

plt.plot(xi,p,'-',color="blue", label=r"$p(\xi)$")
plt.title(r'Prikaz približkov za $p$, $n=$' +str(n))
plt.xlabel(r"$\xi$")
plt.ylabel(r"$p$")
plt.legend()
plt.show() 

#Temperatura
T=[]
for i in range(len(theta)):
    T.append(theta[i])
plt.plot(xi,T,'-',color="blue", label=r"$T(\xi)$")
plt.title(r'Prikaz približkov za $T$, $n=$' +str(n))
plt.xlabel(r"$\xi$")
plt.ylabel(r"$T$")
plt.legend()
plt.show() 

plt.plot(xi/6.895,m/max(m),'-',color="blue", label=r"$m(\xi)$")
plt.plot(xi/6.895,rho,'-',color="yellow", label=r"$\rho(\xi)$")
plt.plot(xi/6.895,p,'-',color="green", label=r"$p(\xi)$")
plt.plot(xi/6.895,T,'-',color="red", label=r"$T(\xi)$")
plt.title(r'Prikaz približkov za vse količine, $n=$' +str(n))
plt.xlabel(r"$\xi$")
plt.ylabel(r"$T$")
plt.legend()
plt.show() 
for i in range(len(theta)):
    if theta[i]>0:
        index=i
    else:
        break

#Določanje xi
xi0=(xi[index]+xi[index+1])/2
odvod0=odvod[index]
M=-xi0**2*odvod0

print(xi0)
print(odvod0)
print(M)