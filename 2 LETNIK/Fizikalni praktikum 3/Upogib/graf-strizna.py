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

#plotting
#plt.ylim(-1e-4, 1e-3)
for i in x:
    j = i*9.8*1e-3
    F1 = np.linspace(j, j, 500)
    F2 = np.linspace(-j, -j, 500)
    F = np.append(F1, F2)
    plt.plot(n, F)

plt.xlabel("$x[cm]$")
plt.ylabel("$F[N]$")
plt.title("Strižna sila glede na položaj v palici")

plt.savefig("strizna.pdf")
plt.close()