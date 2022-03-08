import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Hitrost zvoka v predmetu.txt") 

#data load and manipulation
y,x = np.loadtxt(my_file, delimiter="\t", unpack="True")

y *=2
#fitting function
f = lambda x, a, b : a*x + b
args = [20, 0]
fit = sp.curve_fit(f, x, y, p0=args)

fitlabel = "$%.2E \\frac{m}{ms}*d %.2E$"%(fit[0][0], fit[0][1])
print(fitlabel)
t0 = np.linspace(0, 0.0348, 100)


#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, y, color="black", marker=".")
#plt.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)

plt.xlabel("Čas[ms]")
plt.ylabel("Pot[s]")
plt.legend()
print(fit[1])
plt.title("Hitrost signala v našem predmetu")
plt.savefig("Meritev casa v predmetu.png")
plt.show()
