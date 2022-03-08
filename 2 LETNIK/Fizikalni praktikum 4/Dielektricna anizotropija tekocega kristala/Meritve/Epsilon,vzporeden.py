import matplotlib.pyplot as plt
import scipy.optimize as sp
import numpy as np
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Epsilon,vzporeden.txt")

T, C = np.loadtxt(my_file, delimiter = "\t", unpack ="True")    #x=Temperatura v K, y=Kapacitivnost v nF, DC
T+=273 #v Kelvine
C0=0.06 #nF
R=[]
for i in range(len(T)):
    R.append(100*(1+0.00385*(T[i]-273.16)))
print(R)

f = lambda x, a, b : 100*(1+a*(x - b))
args = [20, 0]
fit = sp.curve_fit(f, T, R, p0=args)
fitlabel = "$R(T)=100*(1+%.5f*(T[i]+ %.2f))$"%(fit[0][0], fit[0][1])
print(fitlabel)
t0 = np.linspace(300, 345, 100)
plt.scatter(T, R, color="black", marker=".")
plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
#plt.errorbar(x, y, yerr=yerror, label='meritve', barsabove="True", linestyle="None", color="black", capsize=2)
plt.xlabel("$T[K]$")
plt.ylabel("$R[Ohm]$")
#plt.axis([-1.7, 1.7, 0, 1])
plt.legend()
plt.title("Odvisnost upora od temperature, vzporedno")
plt.savefig("Upor_temperatura, vzporedno.png")
plt.show()
plt.close()

print("\n")
print("\n")
E=[]
for i in range(len(C)):
    E.append(C[i]/C0)
    
print(E)
plt.scatter(T, E, color="black", marker=".")
#plt.errorbar(x, y, yerr=yerror, label='meritve', barsabove="True", linestyle="None", color="black", capsize=2)
plt.xlabel("$T[K]$")
plt.ylabel("$E$")
#plt.axis([-1.7, 1.7, 0, 1])
plt.legend()
plt.title("Dielektrična konstanta v odvisnosti od temperature, vzporedno")
plt.savefig("Dielektričnost_temperatura, vzporedno.png")
plt.show()
plt.close()
