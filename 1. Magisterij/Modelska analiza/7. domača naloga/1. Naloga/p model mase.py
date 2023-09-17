import fractions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
from tqdm import tqdm
import time
from matplotlib import cm
N=5
p=1
def creating(N,p):
    n=0
    x_line=[]
    y_line=[]
    z_line=[]
    x_generated=[]
    y_generated=[]
    z_generated=[]
    N=10**N
    color=[]
    for i in range(int(N)):
        x=np.random.rand()*2-1
        y=np.random.rand()*2-1
        z=np.random.rand()*2-1
        x_generated.append(x)
        y_generated.append(y)
        z_generated.append(z)
        if((np.sqrt(np.abs(x)) + np.sqrt(np.abs(y)) + np.sqrt(np.abs(z))) <= 1):
            n = n + 1*np.sqrt((x**2 + y**2 + z**2))**p
            x_line.append(x)
            y_line.append(y)
            z_line.append(z)
            color.append(i)
    sigma = np.sqrt(N-1)/N
    return x_line,y_line,z_line,x_generated,y_generated,z_generated, sigma, color,n
            
x_line,y_line,z_line,x_generated,y_generated,z_generated,sigma,color, n=creating(N,p)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz podatkov za $10^{%i}$, masa=$%.5f$" %(N,8*n/10**N))
pa=ax.scatter3D(x_line, y_line, z_line,c=color, cmap="rainbow")
cbar=fig.colorbar(pa)
ax.legend()
cbar.set_label('N', rotation=270)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_zlim([-1.1, 1.1])
plt.show()

"""
N_index=np.linspace(2,7,20)
masa=[]
timer=[]
sigma_line=[]
N_line=[]
for i in tqdm(N_index):
    t_tot=0
    start = time.time()
    x_line,y_line,z_line,x_generated,y_generated,z_generated,sigma,color, n=creating(i,p)
    end= time.time()
    masa.append(8*n/10**i)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(r"Prikaz podatkov: $10^{%i}$, masa=$%.5f$, p=$%i$" %(i,8*n/10**i,p))
    pa=ax.scatter3D(x_line, y_line, z_line,c=color, cmap="rainbow")
    cbar=fig.colorbar(pa)
    cbar.set_label('N', rotation=270)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    plt.show()
    N_line.append(10**i)
    sigma_line.append(sigma)
    t_tot+=end-start
    timer.append(t_tot)
    plt.errorbar(10**i, 8*n/10**i,sigma, fmt ='.'+"b", capsize=3)
plt.title("Prikaz odvisnosti mase od podatkov, $p=%i$" %(p))
plt.xscale("log")
plt.xlabel("$10^N$")
plt.ylabel("m")
plt.legend()
plt.grid()
plt.show()

plt.title("Prikaz odvisnosti časa od podatkov,$p=%i$" %(p))
plt.plot(N_line, timer)
plt.xscale("log")
plt.xlabel("$10^N$")
plt.ylabel("t[s]")
plt.legend()
plt.grid()
plt.show()"""
"""
N=5
p_line=np.linspace(0.1,2,40)
masa=[]
timer=[]
sigma_line=[]
for i in tqdm(p_line):
    t_tot=0
    start = time.time()
    x_line,y_line,z_line,x_generated,y_generated,z_generated,sigma,color, n=creating(N,i)
    end= time.time()
    masa.append(8*n/10**N)
    sigma_line.append(sigma)
    t_tot+=end-start
    timer.append(t_tot)
plt.plot(p_line, masa)
plt.title("Prikaz odvisnosti mase od izbire p, $N=10^{%i}$" %(N))
plt.xlabel("p")
plt.ylabel("m")
plt.legend()
plt.grid()
plt.show()"""

masa=[]
timer=[]
sigma_line=[]
p_line=np.linspace(0.1,2,30)
N_special=np.linspace(2,6,10)
for j in tqdm(N_special):
    masa=[]
    timer=[]
    sigma_line=[]
    for i in p_line:
        t_tot=0
        start = time.time()
        x_line,y_line,z_line,x_generated,y_generated,z_generated,sigma,color, n=creating(j,i)
        end= time.time()
        masa.append(8*n/10**j)
        sigma_line.append(sigma)
        t_tot+=end-start
        timer.append(t_tot)
    plt.plot(p_line, masa,label="N=$10^{%.2f}$" %(j), color=plt.cm.gist_rainbow(j/len(N_special)))
plt.title("Prikaz odvisnosti mase od izbire p, $N=10^{%i}$" %(N))
plt.xlabel("p")
plt.ylabel("m")
plt.legend()
plt.grid()
plt.show()

plt.plot(p_line, timer)
plt.show()

plt.errorbar(p_line, masa, sigma_line, fmt ='.r', capsize=3)
plt.show()
