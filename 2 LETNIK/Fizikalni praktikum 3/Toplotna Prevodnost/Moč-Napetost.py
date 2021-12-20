import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Moč-napetost.txt")

#data load and manipulation
y, x1 = np.loadtxt(my_file, delimiter="\t", unpack="True")
k = 0.0487
x = x1/k
y /= 0.024**2*np.pi

#fitting function
f = lambda x, a, b: a*x + b
args = [20, 0]
fit = sp.curve_fit(f, x, y, p0=args)
print(fit)
fitlabel = "$%d\\frac{W}{m^2K}\Delta T+%d\\frac{W}{m^2}, \sigma_k=%d$"%(fit[0][0]//100*100, fit[0][1]//100*100, np.sqrt(fit[1][0][0]))
print(fitlabel)
t0 = np.linspace(2, 8, 1000)


#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, y, color="black", marker=".")
plt.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)

plt.xlabel("$\Delta T[K]$")
plt.ylabel("$j[w/m^2]$")
plt.legend()
plt.title("$j(\Delta T)$")

plt.show()
plt.close()
