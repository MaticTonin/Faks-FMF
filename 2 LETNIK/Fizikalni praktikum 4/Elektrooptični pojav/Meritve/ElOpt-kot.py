import matplotlib.pyplot as plt
import scipy.optimize as sp
import numpy as np
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "2polarizator.txt")

x, y = np.loadtxt(my_file, delimiter = " ", unpack ="True")    #x=kot v stopinjah, y=tok [A], DC
y *= 1000           #iz A v mA
x = x*np.pi/180     #pretvorba iz stopinj v radiane

I0 = 0.409                      #tok na laserju v mA...napaka 0.015 mA
y /= I0                         #da dobimo razmerje tokov oz moči

yerror = []
for i in y:
    yerror.append(((0.015/0.409)+0.05)*i)
    

def P(x, P0, P1, D):
    return P1*(np.sin(2*x+D))**2 + P0
args = [0, 0, 0]
fit = sp.curve_fit(P, x, y, p0=args, sigma=yerror)
print(fit)
fitlabel = "$(%.4f \pm %.3f) \cdot sin^2(2\\beta + %.4f \pm %.3f) + %.4f \pm %.3f$"%(fit[0][1], np.sqrt(fit[1][1][1]), fit[0][2], np.sqrt(fit[1][2][2]), fit[0][0], np.sqrt(fit[1][0][0]))
print(fitlabel)
t0 = np.linspace(x[0], 0.5*np.pi, 1000)

plt.scatter(x, y, color="blue", marker=".")
plt.errorbar(x, y, yerr=yerror, label='meritve', barsabove="True", linestyle="None", color="black", capsize=2)
plt.plot(t0, P(t0, fit[0][0], fit[0][1], fit[0][2]), label=fitlabel, color="red")

plt.xlabel("$\\beta [rad]$")
plt.ylabel("$I/I_0$")
plt.axis([-1.7, 1.7, 0, 0.25])
plt.legend()
plt.title("Prepuščena moč skozi polarizatorja pri različnih kotih")
plt.savefig("Izkoristek_kot,2_polarizator.png")
plt.show()
plt.close()
