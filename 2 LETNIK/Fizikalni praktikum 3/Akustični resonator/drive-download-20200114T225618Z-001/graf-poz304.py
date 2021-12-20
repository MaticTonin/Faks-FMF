import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "302,5.dat")

#data load and manipulation
x, amp, dev = np.loadtxt(my_file, delimiter="  ", unpack="True")

t = np.array([2.7+i*2 for i in range(len(dev))])

#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(t, dev, color="black", marker=".")
#plt.errorbar(x, y, yerr=yerror, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)

plt.xlabel("x[cm]")
plt.ylabel("U")

plt.title("Odziv glede na odmikih od roba, $\\nu$=304 Hz, $(n_x, n_y, n_z)=(1,0,0)$")

plt.savefig("f302,5.pdf")
plt.close()
