import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "448.45.dat")

#data load and manipulation
x, amp, dev = np.loadtxt(my_file, delimiter="  ", unpack="True")

t = np.array([2.7+i*2 for i in range(len(dev))])

#plotting
plt.ylim(0.05, 0.8)
plt.scatter(t, dev, color="black", marker=".")
#plt.errorbar(x, y, yerr=yerror, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)

plt.xlabel("x[cm]")
plt.ylabel("U")

plt.title("Odziv glede na odmikih od roba, $\\nu$=450Hz, $(n_x, n_y, n_z)=(0,1,0)$")

plt.savefig("f448.45.pdf")
plt.close()
