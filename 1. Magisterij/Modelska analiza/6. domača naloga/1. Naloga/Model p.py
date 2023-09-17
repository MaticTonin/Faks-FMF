from scipy.optimize import curve_fit
import plotly.express as px
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

y_0 = 104.6831
a = 21.19


def f(t, y_0, a, p):
   return y_0*t**p/(t**p+a**p)

def parameter_maker(x,y,sigma):
    popt, pcov = curve_fit(f, x, y, method ='lm', sigma = sigma, absolute_sigma= True)
    y_0,a,p=popt
    return y_0,a,p,pcov

def f_0(t, y_0, a):
   return y_0*t/(t+a)
t = np.linspace(0, 1000, 10000)


popt, pcov = curve_fit(f, x, y, method ='lm', sigma = sigma, absolute_sigma= True)
y_0,a,p=popt
yerr = sigma


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
y_0_1=1/y_0_no
a_1=a_no*y_0
t = np.linspace(0, 1000, 10000)
chi = 0
stevec = 0
for i in y:
    chi = chi + ((i - f(x[stevec], *popt))/sigma[stevec])**2
    #print(chi, stevec)
    stevec = stevec + 1
chi = round(chi, 2)

plt.title(r"Prilagajanje funkcije za podatke farmakologije pri $\chi=%.2f$" %(chi))
plt.errorbar(x, y, sigma, label= "Meritve", color= "b",  fmt='.k', capsize=3)
plt.plot(t, f(t,y_0,a,p), color='red',label=r"Fit $y=\frac{y_0x^p}{x^p+a^p}$, $a=%.2f, y_0=%.2f, p=%.2f$" %(a,y_0,p))
plt.grid()
plt.legend()
plt.show()

plt.title(r"Primerjava funkcij za podatke farmakologije pri $\chi=%.2f$" %(chi))
plt.errorbar(x, y, sigma, label= "Meritve", color= "b",  fmt='.k', capsize=3)
plt.plot(t, f(t,y_0,a,p), color='red',label=r"Fit $y=\frac{y_0x^p}{x^p+a^p}$, $a=%.2f, y_0=%.2f, p=%.2f$" %(a,y_0,p))
plt.plot(t, f_0(t,y_0_1,a_1), color='blue',label=r"Fit $y=\frac{y_0x}{x+a}$, $a=%.2f, y_0=%.2f$" %(a_1,y_0_1))
plt.grid()
plt.legend()
plt.show()

plt.title(r"Prilagajanje funkcije za podatke farmakologije pri $\chi=%.2f$" %(chi))
plt.errorbar(x[0:5], y[0:5], sigma[0:5], label= "Meritve", color= "b",  fmt='.k', capsize=3)
plt.plot(t, f(t,y_0,a,p), color='red',label=r"Fit $y=\frac{y_0x^p}{x^p+a^p}$, $a=%.2f, y_0=%.2f, p=%.2f$" %(a,y_0,p))
plt.grid()
plt.xlim(-5.5,25)
plt.ylim(-5.5,65)
plt.legend()
plt.show()


plt.title(r"Smiselnost števila podatkov v log skali")
plt.errorbar(x, y, sigma, label= "Meritve", color= "b",  fmt='.k', capsize=3)
t = np.linspace(0, 1000, 10000)
i=0
for i in range(len(x)):
    if i>3:
        y_0,a,p,pcov= parameter_maker(x[0:i+1],y[0:i+1],sigma[0:i+1])
        popt=y_0,a,p
        chi = 0
        stevec = 0
        for j in y:
            chi = chi + ((j - f(x[stevec], *popt))/sigma[stevec])**2
    #print(chi, stevec)
            stevec = stevec + 1
        chi = round(chi, 2)
        plt.plot(t, f(t,y_0,a,p), color=plt.cm.gist_rainbow(i/len(x)),label=r"Št podatkov $%.1f$, $a=%.2f, y_0=%.2f, $p=%.2f$, \chi=%.2f$" %(i, a,y_0,p,chi) )
plt.grid()
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.show()


plt.title(r"Smiselnost števila podatkov")
plt.errorbar(x, y, sigma, label= "Meritve", color= "b",  fmt='.k', capsize=3)
t = np.linspace(0, 1000, 10000)
i=0
for i in range(len(x)):
    if i>3:
        y_0,a,p,pcov= parameter_maker(x[0:i+1],y[0:i+1],sigma[0:i+1])
        popt=y_0,a,p
        chi = 0
        stevec = 0
        for j in y:
            chi = chi + ((j - f(x[stevec], *popt))/sigma[stevec])**2
    #print(chi, stevec)
            stevec = stevec + 1
        chi = round(chi, 2)
        plt.plot(t, f(t,y_0,a,p), color=plt.cm.gist_rainbow(i/len(x)),label=r"Št podatkov $%.1f$, $a=%.2f, y_0=%.2f, $p=%.2f$, \chi=%.2f$" %(i, a,y_0,p,chi) )
plt.grid()
plt.legend()
plt.xlim(-5.5,1050)
plt.ylim(-5.5,150)
plt.show()

y_0_no,a_no=np.linalg.inv(A).dot(b)
y_0_1=1/y_0_no
a_1=a_no*y_0
y_0,a,p,pcov= parameter_maker(x,y,sigma)
plt.title("Prikaz razlike med podatki in fitoma")
plt.plot(x, abs(y-f(y,y_0,a,p)), color='red',label=r"Fit $y=\frac{y_0x^p}{x^p+a^p}$, $a=%.2f, y_0=%.2f, p=%.2f$" %(a,y_0,p))
plt.plot(x, abs(y-f_0(y,y_0_1,a_1)), color='blue',label=r"Fit $y=\frac{y_0x}{x+a}$, $a=%.2f, y_0=%.2f$" %(a_1,y_0_1))
plt.grid()
plt.legend()
plt.show()

plt.title("Prikaz razlike med podatki in fitoma")
plt.plot(x, abs(y-f(y,y_0,a,p)), color='red',label=r"Fit $y=\frac{y_0x^p}{x^p+a^p}$, $a=%.2f, y_0=%.2f, p=%.2f$" %(a,y_0,p))
plt.plot(x, abs(y-f_0(y,y_0_1,a_1)), color='blue',label=r"Fit $y=\frac{y_0x}{x+a}$, $a=%.2f, y_0=%.2f$" %(a_1,y_0_1))
plt.grid()

plt.yscale("log")
plt.legend()
plt.show()

plt.title("Prikaz relativne razlike med podatki in fitoma")
plt.plot(x, abs(y-f(y,y_0,a,p))/y, "x-",color='red',label=r"Fit $y=\frac{y_0x^p}{x^p+a^p}$, $a=%.2f, y_0=%.2f, p=%.2f$" %(a,y_0,p))
plt.plot(x, abs(y-f_0(y,y_0_1,a_1))/y,"x-", color='blue',label=r"Fit $y=\frac{y_0x}{x+a}$, $a=%.2f, y_0=%.2f$" %(a_1,y_0_1))
plt.grid()
plt.legend()
plt.show()

plt.title("Prikaz relativne razlike med podatki in fitoma")
plt.plot(x, abs(y-f(y,y_0,a,p))/y, "x-",color='red',label=r"Fit $y=\frac{y_0x^p}{x^p+a^p}$, $a=%.2f, y_0=%.2f, p=%.2f$" %(a,y_0,p))
plt.plot(x, abs(y-f_0(y,y_0_1,a_1))/y, "x-",color='blue',label=r"Fit $y=\frac{y_0x}{x+a}$, $a=%.2f, y_0=%.2f$" %(a_1,y_0_1))
plt.grid()

plt.yscale("log")
plt.legend()
plt.show()

plt.title("Prikaz razlike med fitoma podatkov")
plt.plot(x, abs(f_0(y,y_0_1,a_1)-f(y,y_0,a,p)), color='red')
plt.grid()
plt.legend()
plt.show()

plt.title("Prikaz razlike med fitoma podatkov")
plt.plot(x, abs(f_0(y,y_0_1,a_1)-f(y,y_0,a,p)), color='red')
plt.grid()
plt.yscale("log")
plt.legend()
plt.show()


fig, ax = plt.subplots()
fig.suptitle("Prikaz matrike A v logaritemski skali")
min=0
max=0
def cor(m,n):
    return pcov[m][n]/np.sqrt(pcov[m][m]*pcov[n][n])

print(cor(0,0))
print(cor(0,1))
print(cor(0,2))
print(cor(1,2))
A_normal=np.zeros((3,3))
cov_matrix=np.zeros((3,3))
for i in range(len(pcov)):
    for j in range(len(pcov)):
        if i==j:
            cov_matrix[i][j]=cor(0,0)
        elif (i==1 and j==0) or (i==0 and j ==1):
            cov_matrix[i][j]=cor(0,1)
        elif (i==2 and j==0) or (i==0 and j ==2):
            cov_matrix[i][j]=cor(0,2)
        elif (i==2 and j==1) or (i==1 and j==2):
            cov_matrix[i][j]=cor(0,2)
        cov_matrix[i][j]=round(cov_matrix[i][j], 3)
max=0
min=0
for i in range(len(cov_matrix)):
    for j in range(len(cov_matrix)):
        A_normal[i][j]=cov_matrix[i][j]
        if cov_matrix[i][j]>max:
            max=cov_matrix[i][j]
        if cov_matrix[i][j]<min:
            min=cov_matrix[i][j]
fig.suptitle("Kovariančna matrika za fit s parametrom p")
im=ax.imshow(cov_matrix, vmin=min, vmax=max, cmap="winter", aspect='auto')
fig.colorbar(im)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(["y_0", "a", "p"])
ax.set_yticklabels(["y_0", "a", "p"])
for (j,i),label in np.ndenumerate(A_normal):
    ax.text(i,j,label,ha='center',va='center')
plt.show()

popt, pcov = curve_fit(f_0, x, y, method ='lm', sigma = sigma, absolute_sigma= True)
y_0,a = popt
yerr = sigma  
fig, ax = plt.subplots()
min=0
max=0
def cor(m,n):
    return pcov[m][n]/np.sqrt(pcov[m][m]*pcov[n][n])

print(cor(0,0))
print(cor(0,1))
print(cor(1,0))
print(cor(1,1))
A_normal=np.zeros((2,2))
cov_matrix=np.zeros((2,2))
for i in range(len(pcov)):
    for j in range(len(pcov)):
        if i==j:
            cov_matrix[i][j]=cor(0,0)
        elif (i==1 and j==0) or (i==0 and j==1):
            cov_matrix[i][j]=cor(0,1)
        cov_matrix[i][j]=round(cov_matrix[i][j], 3)

max=0
min=0
for i in range(len(cov_matrix)):
    for j in range(len(cov_matrix)):
        A_normal[i][j]=cov_matrix[i][j]
        if cov_matrix[i][j]>max:
            max=cov_matrix[i][j]
        if pcov[i][j]<min:
            min=cov_matrix[i][j]
fig.suptitle("Kovariančna matrika za navaden fit")
im=ax.imshow(cov_matrix, vmin=min, vmax=max, cmap="winter", aspect='auto')
fig.colorbar(im)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["y_0", "a"])
ax.set_yticklabels(["y_0", "a"])
for (j,i),label in np.ndenumerate(A_normal):
    ax.text(i,j,label,ha='center',va='center')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

popt, pcov = curve_fit(f, x, y, method ='lm', sigma = sigma, absolute_sigma= True)
y_0,a,p=popt
yerr = sigma

dy0 = 0.0009
da = 0.05
def chi(X,Y):
    chi = 0
    stevec = 0
    for j in y:
        chi = chi + ((j - f(x[stevec], X,Y,p))/sigma[stevec])**2
    #print(chi, stevec)
        stevec = stevec + 1
    return chi
N = 50
y_0values = np.linspace(y_0-N*50*dy0,y_0+N*50*dy0, N*10)
a_values = np.linspace (a-N*da,a+N*da,N*10)
p_values = np.linspace (p-N*da,p+N*10*da,N*10)
 
X, Y = np.meshgrid(y_0values, a_values)
Z = chi(X,Y)

fig = plt.figure()
fig.suptitle(r"Prikaz $\chi^2$ v odvisnosti od $y_0$ in $a$")
ax = plt.axes(projection='3d')
#ax.axes.set_zlim3d(bottom=0, top=20)
ax.set_xlabel('$y_0$')
ax.set_ylabel('$a$')
ax.set_zlabel(r'$\chi^2$')
ax.scatter3D(y_0, a+0.05, chi(y_0,a),s = 50 , color ='r', label="$y_0=%.2f$, $a=%.2f$" %(y_0,a))
ax.plot_surface(X,Y,Z,cmap=cm.coolwarm)
ax.legend()
plt.show()

X, Y = np.meshgrid(y_0values, p_values)
def chi(X,Y):
    chi = 0
    stevec = 0
    for j in y:
        chi = chi + ((j - f(x[stevec], X,a,Y))/sigma[stevec])**2
    #print(chi, stevec)
        stevec = stevec + 1
    return chi
Z = chi(X,Y)

fig = plt.figure()
fig.suptitle(r"Prikaz $\chi^2$ v odvisnosti od $y_0$ in $p$")
ax = plt.axes(projection='3d')
#ax.axes.set_zlim3d(bottom=0, top=20)
ax.set_xlabel('$y_0$')
ax.set_ylabel('$p$')
ax.set_zlabel(r'$\chi^2$')
ax.scatter3D(y_0, p+0.05, chi(y_0,p),s = 50 , color ='r', label="$y_0=%.2f$, $p=%.2f$" %(y_0,p))
ax.plot_surface(X,Y,Z,cmap=cm.coolwarm)
ax.legend()
plt.show()

X, Y = np.meshgrid(a_values, p_values)
def chi(X,Y):
    chi = 0
    stevec = 0
    for j in y:
        chi = chi + ((j - f(x[stevec], y_0,X,Y))/sigma[stevec])**2
    #print(chi, stevec)
        stevec = stevec + 1
    return chi
Z = chi(X,Y)

fig = plt.figure()
fig.suptitle(r"Prikaz $\chi^2$ v odvisnosti od $a$ in $p$")
ax = plt.axes(projection='3d')
#ax.axes.set_zlim3d(bottom=0, top=20)
ax.set_xlabel('$a$')
ax.set_ylabel('$p$')
ax.set_zlabel(r'$\chi^2$')
ax.scatter3D(a, p, chi(a,p),s = 50 , color ='r', label="$a=%.2f$, $p=%.2f$" %(a,p))
ax.plot_surface(X,Y,Z,cmap=cm.coolwarm)
ax.legend()
plt.show()