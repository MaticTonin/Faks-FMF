import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp
import mycode as mc
import sys
import sympy as smp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "okrogla.dat")

#data load and manipulation
x, z = np.loadtxt(my_file, delimiter="\t", unpack="True")

y = np.array([i-z[0]for i in z])
yerror = np.array([0.01 for i in y])

n = np.linspace(-56.1/2, 56.2/2, 1000)
def f(x, l, F):
    if x < 0: return F/2*(x+l/2) 
    else: return F/2*(l/2-x)

#plotting
#plt.ylim(-1e-4, 1e-3)
for i in x:
    j = i*9.8*1e-3
    u = np.array([])
    for k in n:
        num = f(k/100, 0.562, j)
        u = np.append(u, [num])
    plt.plot(n, u)

plt.xlabel("$x[cm]$")
plt.ylabel("$M[Nm]$")
plt.title("Navor glede na položaj v palici")

plt.savefig("navor.pdf")
plt.close()