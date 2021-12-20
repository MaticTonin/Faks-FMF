import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp
import mycode as mc
import sys
import sympy as smp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Kvadratna.dat")

#data load and manipulation
x, z = np.loadtxt(my_file, delimiter="\t", unpack="True")

y = np.array([i-z[0]for i in z])
yerror = np.array([0.01 for i in y])

#fitting function
#predznak za label





#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, -y, color="black", marker=".")
plt.errorbar(x, -y, yerr=yerror, label='meritve', barsabove="True", linestyle="None", color="black", capsize=4)
plt.plot(lin[0], lin[1], label=fitlabel)

plt.xlabel("$m[g]$")
plt.ylabel("$-u[mm]$")
plt.legend()
plt.title("Uklon oglate palice pri različnih obremenitvah")

plt.savefig("uklon-oglate.pdf")
plt.close()
