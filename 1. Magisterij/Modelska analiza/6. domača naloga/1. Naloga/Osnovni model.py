import numpy as np 
import os
import pandas as pd
import gdal
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib import cm

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(THIS_FOLDER + "\\farmakoloski.dat",dtype=float)
x,y,sigma=data.T
def parameter_maker(x,y,sigma):
    A=np.zeros((2,2))
    for i in range(len(A)):
        for j in range(len((A[0]))):
            if i==0 and j==0:
                A_11=0
                for k in range(len(sigma)):
                    A_11+=(sigma[k]/y[k]**2)**(-2)
                A[i][j]=A_11
            elif (i+j)==1:
                A_21=0
                for k in range(len(sigma)):
                    A_21+=1/(x[k]*(sigma[k]/y[k]**2)**2)
                A[i][j]=A_21
            elif i==1 and j==1:
                A_22=0
                for k in range(len(sigma)):
                    A_22+=1/(x[k]**2*(sigma[k]/y[k]**2)**2)
                A[i][j]=A_22
    b=np.zeros(2)
    for i in range(len(b)):
        if i==0:
            for k in range(len(x)):
                b[i]+=1/(y[k]*(sigma[k]/y[k]**2)**2)
        if i==1:
            for k in range(len(x)):
                b[i]+=1/(y[k]*x[k]*(sigma[k]/y[k]**2)**2)

    A_inv=np.linalg.inv(A)
    y_0_no,a_no=np.linalg.inv(A).dot(b)
    y_0=1/y_0_no
    a=a_no*y_0
    return y_0,a

A=np.zeros((2,2))
for i in range(len(A)):
    for j in range(len((A[0]))):
        if i==0 and j==0:
            A_11=0
            for k in range(len(sigma)):
                A_11+=(sigma[k]/y[k]**2)**(-2)
            A[i][j]=A_11
        elif (i+j)==1:
            A_21=0
            for k in range(len(sigma)):
                A_21+=1/(x[k]*(sigma[k]/y[k]**2)**2)
            A[i][j]=A_21
        elif i==1 and j==1:
            A_22=0
            for k in range(len(sigma)):
                A_22+=1/(x[k]**2*(sigma[k]/y[k]**2)**2)
            A[i][j]=A_22
b=np.zeros(2)
for i in range(len(b)):
    if i==0:
        for k in range(len(x)):
            b[i]+=1/(y[k]*(sigma[k]/y[k]**2)**2)
    if i==1:
        for k in range(len(x)):
            b[i]+=1/(y[k]*x[k]*(sigma[k]/y[k]**2)**2)


A_inv=np.linalg.inv(A)
A_normal=np.zeros((2,2))
min=0
max=0
fig, ax = plt.subplots()
for i in range(len(A)):
    for j in range(len(A)):
        A_normal[i][j]=A[i][j]
        A[i][j]=np.log10(A[i][j])
        if A[i][j]>max:
            max=A[i][j]
        if A[i][j]<min:
            min=A[i][j]
fig.suptitle("Prikaz matrike A v logaritemski skali")
im=ax.imshow(A, vmin=min, vmax=max, cmap="Reds", aspect='auto')
fig.colorbar(im)
for (j,i),label in np.ndenumerate(A_normal):
    ax.text(i,j,label,ha='center',va='center')
plt.show()
y_0_no,a_no=np.linalg.inv(A).dot(b)
y_0=1/y_0_no
a=a_no*y_0


def f(t,y_0,a):
   return y_0*t/(t+a)

def g(t,y_0,a):
    return 1/y_0 + a/y_0/t

t = np.linspace(0, 1000, 10000)
chi = 0
for i in range(len(y)):
    chi+= ((1/y[i] - g(x[i],y_0,a))/(sigma[i]/y[i]**2))**2
plt.title(r"Prilagajanje funkcije za podatke farmakologije pri $\chi=%.2f$" %(chi))
plt.errorbar(x, y, sigma, label= "Meritve", color= "b",  fmt='.k', capsize=3)
plt.plot(t, f(t,y_0,a), color='red',label=r"Fit $y=\frac{y_0x}{x+a}$, $a=%.2f, y_0=%.2f$" %(a,y_0) )
plt.grid()
plt.legend()
plt.show()

plt.title(r"Prilagajanje funkcije za podatke farmakologije pri $\chi=%.2f$" %(chi))
plt.errorbar(x[0:5], y[0:5], sigma[0:5], label= "Meritve", color= "b",  fmt='.k', capsize=3)
plt.plot(t, f(t,y_0,a), color='red',label=r"Fit $y=\frac{y_0x}{x+a}$, $a=%.2f, y_0=%.2f$" %(a,y_0) )
plt.grid()
plt.xlim(-5.5,25)
plt.ylim(-5.5,65)
plt.legend()
plt.show()


y_0,a = parameter_maker(x[0:4],y[0:4],sigma[0:4])

t = np.linspace(0, x[4]+10, 10000)
chi = 0
for i in range(len(y[0:4])):
    chi+= ((1/y[i] - g(x[i],y_0,a))/(sigma[i]/y[i]**2))**2


plt.title(r"Smiselnost števila podatkov")
plt.errorbar(x, y, sigma, label= "Meritve", color= "b",  fmt='.k', capsize=3)
t = np.linspace(0, 1000, 10000)
for i in range(len(x)):
    if i>0:
        y_0,a = parameter_maker(x[0:i+1],y[0:i+1],sigma[0:i+1])
        chi = 0
        for j in range(len(y[0:i+1])):
            chi+= ((1/y[j] - g(x[j],y_0,a))/(sigma[j]/y[j]**2))**2
        plt.plot(t, f(t,y_0,a), color=plt.cm.gist_rainbow(i/len(x)),label=r"Št podatkov $%.1f$, $a=%.2f, y_0=%.2f, \chi=%.2f$" %(i, a,y_0,chi) )
plt.grid()
plt.legend()
plt.xlim(-5.5,1050)
plt.ylim(-5.5,150)
plt.show()


plt.title(r"Smiselnost števila podatkov v log skali")
plt.errorbar(x, y, sigma, label= "Meritve", color= "b",  fmt='.k', capsize=3)
t = np.linspace(0, 1000, 10000)
for i in range(len(x)):
    if i>0:
        y_0,a = parameter_maker(x[0:i+1],y[0:i+1],sigma[0:i+1])
        chi = 0
        for j in range(len(y[0:i+1])):
            chi+= ((1/y[j] - g(x[j],y_0,a))/(sigma[j]/y[j]**2))**2
        plt.plot(t, f(t,y_0,a), color=plt.cm.gist_rainbow(i/len(x)),label=r"Št podatkov $%.1f$, $a=%.2f, y_0=%.2f, \chi=%.2f$" %(i, a,y_0,chi) )
plt.grid()
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# X = y_0
# Y = a 
y_0 = 104.6831
dy0 = 0.0009
a = 21.19
da = 0.05
def f(t):
   return y_0*t/(t+a)

veca = [1/y_0, a/y_0]
def g(t, X, Y):
    return 1/X +Y/X/t

t = np.linspace(0, 1000, 10000)

chi = 0
stevec = 0


def chi(X,Y):
    c = 0
    for i in range(len(y)):
        c += ((1/y[i] - g(x[i], X,Y))/(sigma[i]/y[i]**2))**2
        #print(c)
    return c
N = 50
y_0values = np.linspace(y_0-N*50*dy0,y_0+N*50*dy0, N*10)
a_values = np.linspace (a-N*da,a+N*da,N*10)
 
X, Y = np.meshgrid(y_0values, a_values)
Z = chi(X,Y)



fig = plt.figure()
fig.suptitle(r"Prikaz $\chi^2$ v odvisnosti od $y_0$ in $a$")
ax = plt.axes(projection='3d')
#ax.axes.set_zlim3d(bottom=0, top=20)
ax.set_xlabel('$y_0$')
ax.set_ylabel('$a$')
ax.set_zlabel(r'$\chi^2$')
ax.scatter3D(y_0, a+0.05, chi(y_0,a), s = 50 , color ='r', label="$y_0=%.2f$, $a=%.2f$" %(y_0,a))
ax.plot_surface(X,Y,Z, cmap=cm.coolwarm)
ax.legend()

plt.show()