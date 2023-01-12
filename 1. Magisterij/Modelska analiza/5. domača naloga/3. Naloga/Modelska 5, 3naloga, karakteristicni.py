from re import A, M
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from itertools import cycle
from tqdm import tqdm
# function that returns dz/dt
n = 250000 #stevilo korakov integracije
l = 30 #čas do kamor simuliramo
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

"""
ti = np.linspace(0,l,n)
IC = np.linspace(0,10,100)
karakteristicni=[]
lam=np.linspace(10,1000,500)
for i in tqdm(lam):
    lines = ["-","--",":"]
    linecycler = cycle(lines)
    w = odeint(model,w0,ti,args=(i,))
    x,y,z,A,B,C=w[:,0],w[:,1], w[:,2], w[:,3], w[:,4], w[:,5]
    indexation=0
    for j in range(len(x)):
        if x[j]<y[j] and indexation==0:
            karakteristicni.append(ti[j])
            indexation+=1
plt.title(r"Prikaz karakterističnega časa za različne $\lambda$")
plt.plot(lam,karakteristicni)
def funkcija(x,k,m):
    return k*x+m
from scipy.optimize import curve_fit
plt.ylabel('t')
plt.xlabel(r'$\lambda$')
plt.legend()
plt.grid()
plt.show()
p0=[1,1]
popt, _ = curve_fit(funkcija, np.log(lam), np.log(karakteristicni),p0)
k,m=popt
y_line=k*np.log(lam)+m
plt.title(r"Prikaz karakterističnega časa; fit premice")
plt.plot(np.log(lam),y_line, label="$\log(\lambda)=k\log(t)+m$; $k=%.2f$, $m=%.2f$"%(k,m))
plt.plot(np.log(lam),np.log(karakteristicni))
plt.ylabel('log(t)')
plt.xlabel(r'log($\lambda$)')
plt.legend()
plt.grid()
plt.show()"""
karakteristicni=[]
lam=np.linspace(10,1000,50)
A0_line=[1,10,100]
for f in A0_line:
    lam=np.linspace(10,100,500)
    karakteristicni=[]
    for i in tqdm(lam):
        lines = ["-","--",":"]
        linecycler = cycle(lines)
        w = odeint(model,w0,ti,args=(i,))
        w0 = [x0, y0, f, f, B0, C0]
        x,y,z,A,B,C=w[:,0],w[:,1], w[:,2], w[:,3], w[:,4], w[:,5]
        indexation=0
        for j in range(len(x)):
            if x[j]<y[j] and indexation==0:
                karakteristicni.append(ti[j])
                indexation+=1
    def funkcija(x,k,m):
        return k*x+m
    from scipy.optimize import curve_fit
    p0=[1,1]
    popt, _ = curve_fit(funkcija, np.log(lam), np.log(karakteristicni),p0)
    k,m=popt
    y_line=k*np.log(lam)+m
    plt.title(r"Prikaz karakterističnega časa; fit premice")
    plt.plot(np.log(lam),y_line, label="$A_0=%.2f$ $\log(\lambda)=k\log(t)+m$; $k=%.2f$, $m=%.2f$"%(f,k,m))
    plt.plot(np.log(lam),np.log(karakteristicni))
plt.ylabel('log(t)')
plt.xlabel(r'log($\lambda$)')
plt.legend()
plt.grid()
plt.show()

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