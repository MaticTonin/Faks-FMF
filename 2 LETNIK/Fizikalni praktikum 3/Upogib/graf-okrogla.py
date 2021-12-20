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

#fitting function
pars, sigs = mc.linfit(x, -y, yerr=yerror, sigmas="sigma_okrogla.dat")

fitlabel = "$%.5fmm/g %.5fmm, \sigma_k=%.1E$"%(pars[0], pars[1], np.sqrt(sigs[0][0]))
lin = mc.linfun(pars[0], pars[1], st=0, end=2100)

#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, -y, color="black", marker=".")
plt.errorbar(x, -y, yerr=yerror, label='meritve', barsabove="True", linestyle="None", color="black", capsize=4)
plt.plot(lin[0], lin[1], label=fitlabel)

plt.xlabel("$m[g]$")
plt.ylabel("$-u[mm]$")
plt.legend()
plt.title("Uklon okrogle palice pri različnih obremenitvah")

plt.savefig("uklon-okrogle.pdf")
plt.close()