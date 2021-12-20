import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "200,1000.dat")

#data load and manipulation
x, amp, dev = np.loadtxt(my_file, delimiter="  ", unpack="True")



#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, dev, color="black", marker=".")
#plt.errorbar(x, y, yerr=yerror, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)

plt.xlabel("$\\nu$[Hz]")
plt.ylabel("U")

plt.title("Odziv na frekvence med 0 in 1000Hz")

plt.savefig("resonance.pdf")
plt.close()
