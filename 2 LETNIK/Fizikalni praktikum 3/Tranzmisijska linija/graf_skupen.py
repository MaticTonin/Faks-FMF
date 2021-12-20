import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Frekvenca C=150.txt")
my_file1 = os.path.join(THIS_FOLDER, "Frekvenca C=330.txt")
my_file2 = os.path.join(THIS_FOLDER, "Frekvenca C=560.txt")
my_file3 = os.path.join(THIS_FOLDER, "Frekvenca C=820.txt")
my_file4 = os.path.join(THIS_FOLDER, "Frekvenca C=1150.txt")


#data load and manipulation
x0, y0= np.loadtxt(my_file, delimiter=" ", unpack="True")
x1, y1= np.loadtxt(my_file1, delimiter=" ", unpack="True")
x2, y2= np.loadtxt(my_file2, delimiter=" ", unpack="True")
x3, y3= np.loadtxt(my_file3, delimiter=" ", unpack="True")
x4, y4= np.loadtxt(my_file4, delimiter=" ", unpack="True")


#plotting
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x0, y0, color="black", marker=".", label="C=150")
ax.scatter(x1, y1, color="red", marker=".", label="C=330")
ax.scatter(x2, y2, color="blue", marker=".", label="C=560")
ax.scatter(x3, y3, color="yellow", marker=".", label="C=820")
ax.scatter(x4, y4, color="green", marker=".", label="C=1150")

#ax.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
#ax.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

#ax.set_xlim(0.45, 0.57)
#ax.set_ylim(-1.3, 0.33)

plt.xlabel("$V[kHz]$")
plt.ylabel("$2U[V]$")
plt.legend()
plt.title("Graf odvisnosti amplitude od frekvence, glavni vrh")

plt.savefig("Graf odvisnosti amplitude od frekvence ožji.pdf")
plt.show()
plt.close()
