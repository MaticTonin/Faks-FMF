import fractions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
from tqdm import tqdm

N=5
def creating(N):
    n=0
    x_line=[]
    y_line=[]
    z_line=[]
    x_generated=[]
    y_generated=[]
    z_generated=[]
    N=10**N
    color=[]
    for i in range(N):
        x=np.random.rand()*2
        y=np.random.rand()*2
        z=np.random.rand()*2-1
        x_generated.append(x)
        y_generated.append(y)
        z_generated.append(z)
        s=np.sqrt(((np.abs(x))/x))
        a=(np.abs((x/2))-((3*np.sqrt(33)-7)/112)*x**2-3+np.sqrt(1-(np.abs(np.abs(x)-2)-1)**2)-y)
        b=((x/7)**2*s*(np.abs(x)-3)+(y/3)**2*s*(y+((3*np.sqrt(33))/7))-1)
        c=(9*s*((1-np.abs(x))*(np.abs(x)-.75))-8*np.abs(x)-y)
        d=(3*np.abs(x)+.75*s*((.75-np.abs(x))*(np.abs(x)-.5))-y)
        e=(2.25*s*((.5-x)*(x+.5))-y)
        f=(((6*np.sqrt(10))/7)+(1.5-.5*np.abs(x))*s*(np.abs(x)-1)-((6*np.sqrt(10))/14)*np.sqrt(4-(np.abs(x)-1)**2)-y)
        if b<0 and a<0 and c>0 and d>0 and e>0 and f>0:
            n = n + 1
            x_line.append(x)
            y_line.append(y)
            z_line.append(z)
            color.append(i)
    sigma = np.sqrt(N-1)/N
    return x_line,y_line,z_line,x_generated,y_generated,z_generated, sigma, color,n
            
x_line,y_line,z_line,x_generated,y_generated,z_generated,sigma,color, n=creating(N)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz podatkov za $10^{%i}$, masa=$%.5f$" %(N,8*n/10**N))
p=ax.scatter3D(x_line, y_line, z_line,c=color, cmap="rainbow")
cbar=fig.colorbar(p)
ax.legend()
cbar.set_label('N', rotation=270)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_zlim([-1.1, 1.1])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz generiranih podatkov za $10^{%i}$" %(N))
p=ax.scatter3D(x_generated, y_generated, z_generated,c=np.linspace(0,10**N,10**N),cmap='rainbow')
cbar=fig.colorbar(p)
cbar.set_label('N', rotation=270)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")
plt.show()

N_index=[2,3,4,5,6,7,8,9,10]
masa=[]
timer=[]
sigma_line=[]
for i in tqdm(N_index):
    start = time.time()
    x_line,y_line,z_line,x_generated,y_generated,z_generated,sigma,color, n=creating(i)
    end= time.time()
    masa.append(8*n/10**i)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(r"Prikaz podatkov za $10^{%i}$, masa=$%.5f$" %(i,8*n/10**i))
    p=ax.scatter3D(x_line, y_line, z_line,c=color, cmap="rainbow")
    cbar=fig.colorbar(p)
    cbar.set_label('N', rotation=270)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    plt.show()
    sigma_line.append(sigma)
    t_tot+=end-start
    timer.append(t_tot)
plt.plot(N_index, masa)
plt.show()

plt.errorbar(N, masa, sigma_line, fmt ='.r', capsize=3)
plt.show()
    
