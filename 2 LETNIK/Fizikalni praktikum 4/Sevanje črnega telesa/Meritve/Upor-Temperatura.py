import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Upor-Temperatura.txt") 

#data load and manipulation
y,z,x = np.loadtxt(my_file, delimiter="\t", unpack="True")
#Da dobim izsevan svetlobni tok
x=x-0.5 #mW
x*=0.001 #W
x*=4*3.14* 0.21**2 #m
x/=0.0001 #m^2
#Da dobim upor
y=y/z #Ohm

#Da dobim temperaturo
T_0=2700. #K
P_0=30#W
x=T_0*(x/P_0)**(1/4)

print(y)
#fitting function
f = lambda x, a, b : a*x + b
args = [20, 0]
fit = sp.curve_fit(f, x, y, p0=args)

fitlabel = "$R= %.2E \cdot T \quad %.2E$"%(fit[0][0], fit[0][1])
print(fitlabel)
t0 = np.linspace(0, 3000, 100)



#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, y, color="black", marker=".")
#plt.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
plt.xlabel("T [K]")
plt.ylabel("R [Ohm]")
plt.legend()
print(x)
plt.title("Graf odvisnosti upora od temperature")
plt.savefig("Upor-Temperatura.png")
plt.show()
