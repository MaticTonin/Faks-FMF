import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Temperatura-upor.txt")

#data load and manipulation
x1, x2, y = np.loadtxt(my_file, delimiter="\t", unpack="True")
x = x2 - x1
yerror = np.array([0.001 for i in y])


#fitting function
f = lambda x, a, b: a*x + b
args = [40, 0]
fit = sp.curve_fit(f, x, y, p0=args, sigma=yerror, absolute_sigma=True)
print(fit)
fitlabel = "$%.5fmVK^{-1}\Delta T%.5fmV, \sigma_k=%.5E$"%(fit[0][0], fit[0][1], np.sqrt(fit[1][0][0]))
print(fitlabel)
t0 = np.linspace(0, 40, 1000)


#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, y, color="black", marker=".")
plt.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)

plt.xlabel("$\Delta T[K]$")
plt.ylabel("$U[mV]$")
plt.legend()
plt.title("Umeritev termočlena")

plt.show()
plt.close()
