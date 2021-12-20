import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "laser_podatki.txt") 

#data load and manipulation
x,y = np.loadtxt(my_file, delimiter="\t", unpack="True")
x_0=1584019423.2113843
sumI_0=0
x=x-x_0
for i in range(len(y)):
    sumI_0+=y[i]


avrg=sumI_0/len(y)*10**3
print("Povprečje toka je %.2E " %avrg)
print("\n")
print("\n")
#fitting function
y=y*10**(3)
f = lambda x, a, b : a*x + b
args = [20, 0]
fit = sp.curve_fit(f, x, y, p0=args)
fitlabel = "$%.2E \\frac{N}{A} * I + %.2E$"%(fit[0][0], fit[0][1])
print(fitlabel)
t0 = np.linspace(-3.551, 3.553, 1000)


#plotting
#plt.ylim(-1e-4, 1e-3)
plt.plot(x, y, color="black", marker=".")
#plt.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
#plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)

plt.xlabel("t[s]")
plt.ylabel("I[mA]")
plt.legend()
plt.title("Tok na laserju v odvisnosti od časa")
plt.savefig("Tok_čas,dioda.png")
plt.show()
