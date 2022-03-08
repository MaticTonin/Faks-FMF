import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Jeklo.txt") 

#data load and manipulation
x,y = np.loadtxt(my_file, delimiter="\t", unpack="True")

y/=100


gmed=2730 #kg/m**3
med=0.0262 #m
voda=1489.5 #m/s

#fitting function
f = lambda x, a, b : a*x + b
args = [20, 0]
fit = sp.curve_fit(f, x, y, p0=args)

fitlabel = "$%.2E m*d %.2E$"%(fit[0][0], fit[0][1])
print(fitlabel)
t0 = np.linspace(0, 4, 100)

hit=(fit[0][0])**(-1)*med*voda
E=gmed*hit**2

#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, y, color="black", marker=".")
#plt.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
print("Hitrost je %.3E  \\frac{m}{s}" %hit)
print("E je %.3E  " %E)
plt.xlabel("n")
plt.ylabel("Pot[m]")
plt.legend()
print(fit[1])
plt.title("Hitrost signala v našem predmetu, Jeklo")
plt.savefig("Jeklo.png")
plt.show()
