import matplotlib.pyplot as plt
import scipy.optimize as sp
import numpy as np
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "kerr_podatki.txt")

x, y = np.loadtxt(my_file, delimiter = " ", unpack ="True")    #x=kot v stopinjah, y=tok [A], DC
          #iz A v mA
     #pretvorba iz stopinj v radiane

I0 = 0.409                      #tok na laserju v mA...napaka 0.015 mA
y /= I0                         #da dobimo razmerje tokov oz moči

L=1.5*10**(-3) #m
d=1.4*10*(-3) #m

yerror = []
for i in y:
    yerror.append(((0.015/0.409)+0.05)*i)
    

def P(x, P1, B, D_0):
    return P1*(np.sin(np.pi*B*L*x**2*d**(-2)+D_0/2))**2 
args = [0, 0, 0]
fit = sp.curve_fit(P, x, y, p0=args, sigma=yerror)
print(fit)
fitlabel = "$(%.4E \pm %.3f) \cdot sin^2(U^2  \cdot \np.pi L/d^2 \\cdot (%.4f \pm %.3f)  + %.4f \pm %.3f) $"%(fit[0][1], np.sqrt(fit[1][1][1]), fit[0][2], np.sqrt(fit[1][2][2]), fit[0][0], np.sqrt(fit[1][0][0]))
print(fitlabel)
t0 = np.linspace(x[0], 0.5*np.pi, 1000)

plt.scatter(x, y, color="black", marker=".")
plt.errorbar(x, y, yerr=yerror, label='meritve', barsabove="True", linestyle="None", color="black", capsize=2)
plt.plot(t0, P(t0, fit[0][0], fit[0][1], fit[0][2]), label=fitlabel, color="red")

plt.xlabel("$U [V]$")
plt.ylabel("$I/I_0$")
plt.axis([0, 1050, 0, 0.3])
plt.legend()
plt.title("Prepuščena moč skozi polarizator pri različnih kotih")
plt.savefig("Izkoristek_kot,Kerrova celica.png")
plt.show()
plt.close()

