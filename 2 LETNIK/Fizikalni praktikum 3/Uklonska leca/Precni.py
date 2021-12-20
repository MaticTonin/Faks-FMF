import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Odmik vzdolz.txt")
my_file1 = os.path.join(THIS_FOLDER, "Izvor zamaknjen za 3 cm vzdolz.txt")


#data load and manipulation
x0, y0 = np.loadtxt(my_file, delimiter="\t", unpack="True")
x1, y1 = np.loadtxt(my_file1, delimiter="\t", unpack="True")



#plotting
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x0, y0, color="black", marker=".", label="x=0cm")
ax.plot(x1, y1, color="red", marker=".", label="x=3cm")

#ax.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
#ax.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-8.2,0))

#ax.set_xlim(0.45, 0.57)
#ax.set_ylim(-1.3, 0.33)

plt.xlabel("$Odmik[cm]$")
plt.ylabel("$2U0[V]$")
plt.legend()
plt.title("Graf odvisnosti amplitude od zamika, vzdolz")

plt.savefig("Amplituda,vzdolz,x=0.pdf")
plt.show()
plt.close()
