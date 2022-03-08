import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Izsev-elektrika.txt") 

#data load and manipulation
x,y = np.loadtxt(my_file, delimiter="\t", unpack="True")

y=y-0.5 #mW
y*=0.001 #W
y*=4*3.14* 0.21**2 #m
y/=0.0001 #m^2
print(y)
#fitting function
f = lambda x, a, b : a*x + b
args = [20, 0]
fit = sp.curve_fit(f, x, y, p0=args)

fitlabel = "$P= %.2E P_{elektrike} %.2E$"%(fit[0][0], fit[0][1])
print(fitlabel)
t0 = np.linspace(0, 32, 100)



#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, y, color="black", marker=".")
#plt.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
plt.xlabel("P_{elektrike} [W]")
plt.ylabel("P_{izsev} [W]")
plt.legend()

plt.title("Graf odvisnosti izseva od električne moči")
plt.savefig("Izsev-elektrika.png")
plt.show()
