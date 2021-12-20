import os
import numpy as np
import numpy.linalg as lin
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.optimize as sp
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data_file = THIS_FOLDER + "\Diagram.txt"
data= np.loadtxt(data_file, delimiter="\t")

t=data.T[0]
Y_max=data.T[1]
a=24.7
b=107.2
d=1.5
k=(a+b)/a
S=24.9
constants=1/(4*np.pi*k**2)*S**2

square_Y=Y_max**2
y=constants/square_Y

yerror=0.0005*constants
print(yerror)

f = lambda x, k, n : k*x+n
args = [0, 0]
fit = sp.curve_fit(f, t, y, p0=args)
y_anal=[]
for i in t:
    y_anal.append(0.84*10**(-5)*i+fit[0][1])
print(fit[0][0],fit[0][1])
fitlabel = r"$\frac{1}{4\pi k^2}\frac{S^2}{Y^2_{max}}(t)=%.5f \cdot 10^{-6} t+ %.2f \cdot 10^{-3}$"%(fit[0][0]*10**6, fit[0][1]*10**3)
plt.title("Prikaz odvisnosti višine stolpca od časa, končni časi")
plt.plot(t,y, "x-")
plt.plot(t,y_anal, label="Analitična")
plt.errorbar(t, y, yerr=yerror, label="Napaka= %.5f" %(yerror), barsabove="True", linestyle="None", color="black", capsize=2)
plt.plot(t, f(t, fit[0][0], fit[0][1]), label=fitlabel)
plt.xlabel("t[s]")
plt.legend()
plt.ylabel(r"$\frac{1}{4\pi k^2}\frac{S^2}{Y^2_{max}}[/]$")
plt.show()