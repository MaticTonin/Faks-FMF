import matplotlib.pyplot as plt
import scipy.optimize as sp
import numpy as np
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Napetosti.txt")

Uamp, Ur = np.loadtxt(my_file, delimiter = "\t", unpack ="True")    ##x=Napetost na celici  y=napetost na toku Ur
R=1000 #Ohm
w=100000 #Hz
C=[]
Uamp1=[]
for i in range(len(Uamp)):
    C.append(Ur/(R*w*Uamp)*10**(9))
for i in range(len(Uamp)):
    Uamp1.append(Uamp*(2)**(-1/2))
    
plt.plot(Uamp1, C, color="blue", marker=".")
#plt.errorbar(x, y, yerr=yerror, label='meritve', barsabove="True", linestyle="None", color="black", capsize=2)
plt.xlabel("$U RMS [V]$")
plt.ylabel("$C[nF]$")
#plt.axis([-1.7, 1.7, 0, 1])
#plt.legend()
plt.title("Kapacitivnost v odvisnodti napetosti")
plt.savefig("Kapacitivnost_napetost.png")
plt.show()
plt.close()
