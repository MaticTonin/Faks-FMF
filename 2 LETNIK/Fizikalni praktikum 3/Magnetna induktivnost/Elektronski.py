import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Elektro2000.txt")

#data load and manipulation
x, y = np.loadtxt(my_file, delimiter="\t", unpack="True")

y /= 1000
y *= 0.01
y /= 200
y /= np.pi*0.009**2

#fitting function
f = lambda x, a, b: a*x + b
args = [20, 0]
fit = sp.curve_fit(f, x, y, p0=args)
print(fit)
fitlabel = "$%.2f\\frac{T}{A}I+%.2fT$"%(fit[0][0], fit[0][1])
print(fitlabel)
t0 = np.linspace(0, 5.5, 1000)


#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, y, color="black", marker=".")
plt.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)

plt.xlabel("I[A]")
plt.ylabel("B[T]")
plt.legend()
plt.title("B(I)")

plt.show()
plt.close()
