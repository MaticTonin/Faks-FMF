import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Karakteristika nihajnega kroga.txt")
my_file1 = os.path.join(THIS_FOLDER, "Karakteristika nihajnega kroga U2.txt")
#data load and manipulation
x0, y0, z0= np.loadtxt(my_file, delimiter="\t", unpack="True")
x1, y1= np.loadtxt(my_file1, delimiter="\t", unpack="True")


#plotting
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x0, y0, color="black", marker=".", label="U1")
ax.plot(x0, z0, color="red", marker=".", label="U2")


#ax.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
#ax.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

#ax.set_xlim(0.45, 0.57)
#ax.set_ylim(-1.3, 0.33)

plt.xlabel("$V[kHz]$")
plt.ylabel("$2U[V]$")
plt.legend()
plt.title("Graf odvisnosti amplitude od frekvence pri C=0, glavni vrh")

plt.savefig("Karakteristika za C=0.png")
plt.show()
plt.close()
