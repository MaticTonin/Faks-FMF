import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp
import math

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Prepuščen-celoten.txt") 

#data load and manipulation
x, y= np.loadtxt(my_file, delimiter="\t", unpack="True")


#Da dobim temperaturo
x=(1.1)/(1.38*10**(-23)*x) #y

y=y+1-(15/3.14**4)*(-x**3*math.log10(1-math.e**(-x))+(6+6*x+3*x**2)*math.e**(-x))
print(y)
#fitting function
f = lambda x, a, b : a*x + b
args = [20, 0]
fit = sp.curve_fit(f, x, y, p0=args)

fitlabel = "$R= %.2E T %.2E$"%(fit[0][0], fit[0][1])
print(fitlabel)
t0 = np.linspace(400, 2100, 100)



#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, y, color="black", marker=".")
#plt.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
plt.xlabel("T [K]")
plt.ylabel("P/P(T)")
plt.legend()
plt.title("Graf odvisnosti razmerja moči od temperature, silicijev zastor")
plt.savefig("Prepuščen-Celoten.png")
plt.show()
