import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Induktivnost2000.txt")

#data load and manipulation
x, y = np.loadtxt(my_file, delimiter="\t", unpack="True")
x += 1.4
x /= 100

y /= 1000
y *= 0.01
y /= 2000
y /= np.pi*0.009**2

yerror = np.array([0.001 for i in y])

yerror /= 1000
yerror *= 0.01
yerror /= 2000
yerror /= np.pi*0.009**2

#fitting function
f = lambda x, a, b, c: a*(b+x**2)**(-3/2) + c
args = [1, 0, 0]
fit = sp.curve_fit(f, x, y, p0=args, sigma=yerror, absolute_sigma=True)
print(fit)
fitlabel = "$\\frac{%.2ETm^{\\frac{3}{2}}}{(%.2Em^2+h^2)^{\\frac{3}{2}}} + %.2E  $"%(fit[0][0], fit[0][1], fit[0][1])
print(fitlabel)
t0 = np.linspace(0, 0.45, 1000)


#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, y, color="black", marker=".")
#plt.errorbar(x, y, yerr=yerror, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
plt.plot(t0, f(t0, fit[0][0], fit[0][1], fit[0][2]), label=fitlabel)

plt.xlabel("h[m]")
plt.ylabel("B[T]")
plt.legend()
plt.title("B(h)")

plt.show()
plt.close()
