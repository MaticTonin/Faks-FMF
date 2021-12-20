import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file1 = os.path.join(THIS_FOLDER, "Temperatura-upor.txt")


#data load and manipulation
x1, y1 = np.loadtxt(my_file1, delimiter="\t", unpack="True")        #odmik detektorja v milimetrih, dvakratnik amplitude napetosti
x1=x1+273

#plotting
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x1, y1, color="red", marker=".", label="nacentriran detektor")
#ax.errorbar(x1, y1, yerr=6, label='meritve', barsabove="True", linestyle="None", color="red", capsize=2)
#ax.errorbar(x2, y2, yerr=yerror2, label='meritve', barsabove="True", linestyle="None", color="blue", capsize=2)

#ax.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
#ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

#ax.set_xlim(0.45, 0.57)
#ax.set_ylim(-1.3, 0.33)

#plt.axis([0.3, 0.7, -2, 4])
plt.xlabel("$T[K]$")
plt.ylabel("$U[mV]$")
plt.legend()
plt.title("Temperatura z napetostjo")

plt.show()
plt.close()
