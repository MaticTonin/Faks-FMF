import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Amplituda R=51.dat")

#data load and manipulation
x, z = np.loadtxt(my_file, delimiter=" ", unpack="True")

#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, z, color="black", marker=".")
#plt.errorbar(x, y, yerr=yerror, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)

plt.xlabel("$\\nu$[MHz]")
plt.ylabel("2U0[V]")

plt.title("Graf Amplitude v odvisnosti od frekvence")

plt.savefig("Amplituda R=51.pdf")
plt.close()
