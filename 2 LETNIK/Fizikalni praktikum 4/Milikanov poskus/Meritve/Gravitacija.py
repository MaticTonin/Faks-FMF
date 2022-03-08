import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "metoda1_in_opis.txt") 

#data load and manipulation
a,b = np.loadtxt(my_file, delimiter="\t", unpack="True")
avgr=0
sumr=0
for i in range(len(a)):
    r= ((9*18.3*10**(-6)*a[i]*10**(-6))/(2*(973-1.194)*9.81))**(1/2)
    sumr+=r
    r=r*10**(6)
    print(r)
avgr=sumr/len(a)
print("Povprečen radij je %.2E " %avgr)
#fitting function

my_file = os.path.join(THIS_FOLDER, "Radiji-hitrosti.txt")
x,y = np.loadtxt(my_file, delimiter="\t", unpack="True")

f = lambda x, a, b : a*x + b
args = [20, 0]
fit = sp.curve_fit(f, x, y, p0=args)

fitlabel = "$%.2E \cdot r \quad %.2E$"%(fit[0][0], fit[0][1])
print(fitlabel)
t0 = np.linspace(0.1, 1.15, 100)


#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, y, color="black", marker=".")
#plt.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
plt.xlabel("Radij [mikro m]")
plt.ylabel("Hitrost [mikro m/s] ")
plt.legend()
print(fit[1])
plt.title("Hitrosti v odvisnosti od radija")
plt.savefig("Hitrosti.png")
plt.show()
