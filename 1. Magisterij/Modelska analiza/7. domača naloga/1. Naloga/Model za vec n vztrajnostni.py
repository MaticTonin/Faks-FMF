import fractions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
from tqdm import tqdm
import time
N=3

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
    for i in range(int(N)):
        x=np.random.rand()*2-1
        y=np.random.rand()*2-1
        z=np.random.rand()*2-1
        x_generated.append(x)
        y_generated.append(y)
        z_generated.append(z)
        if((np.sqrt(np.abs(x)) + np.sqrt(np.abs(y)) + np.sqrt(np.abs(z))) <= 1):
            n = n + 1*(x**2+y**2)
            x_line.append(x)
            y_line.append(y)
            z_line.append(z)
            color.append(i)
    sigma = np.sqrt(N-1)/N
    return x_line,y_line,z_line,x_generated,y_generated,z_generated, sigma, color,n
            
x_line,y_line,z_line,x_generated,y_generated,z_generated,sigma,color, n=creating(N)

N_index=[2,3,4]
N_index=np.linspace(2,6,20)
masa=[]
timer=[]
sigma_line=[]
N_line=[]
color=["b","g","r","y","o"]
j=0
for j in range(5):
    masa=[]
    timer=[]
    sigma_line=[]
    N_line=[]
    for i in tqdm(N_index):
        t_tot=0
        start = time.time()
        x_line,y_line,z_line,x_generated,y_generated,z_generated,sigma,color, n=creating(i)
        end= time.time()
        N_line.append(10**i)
        masa.append(8*n/10**i)
        sigma_line.append(sigma)
        t_tot+=end-start
        timer.append(t_tot)
    plt.plot(N_line, masa, "x-",color=plt.cm.gist_rainbow(j/5),label="Ponovitev: $%i$" %(j))
plt.title("Prikaz odvisnosti $J_{z,z}$ od podatkov")
plt.hlines(0.004232, 0, 10**(8-0.8), label="$J_{z,z}=%.7f$" %(0.004232),color="r")
plt.xscale("log")
plt.xlabel("$10^N$")
plt.ylabel("m")
plt.legend()
plt.grid()
plt.show()


plt.title("Prikaz odvisnosti časa od podatkov")
plt.plot(N_line, timer)
plt.xscale("log")
plt.xlabel("$10^N$")
plt.ylabel("t[s]")
plt.legend()
plt.grid()
plt.show()
