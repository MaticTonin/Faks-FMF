import matplotlib.pyplot as plt
import scipy.optimize as sp
import numpy as np
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "lc_elip.txt")

Y, I = np.loadtxt(my_file, delimiter = " ", unpack ="True")    #Y=kot v stopinjah, I=tok [A], DC
I *= 1000           #iz A v mA
Y = Y*np.pi/180     #pretvorba iz stopinj v radiane

I0 = 0.409                      #tok na laserju v mA...napaka 0.015 mA
I /= I0                         #da dobimo razmerje tokov oz moči
plt.scatter(Y, I, color="black", marker=".")
#plt.errorbar(x, y, yerr=yerror, label='meritve', barsabove="True", linestyle="None", color="black", capsize=2)
plt.xlabel("$\\gamma$")
plt.ylabel("$I/I0$")
#plt.axis([-1.7, 1.7, 0, 1])
plt.legend()
plt.title("Prepuščena moč skozi našo celico pri različnih kotih")
plt.savefig("Izkoristek_kot,elipsa.png")
plt.show()
x = []
y = []

i = 0
while i < len(Y):
        if 26 <i:
            x.append(-I[i]*np.cos(Y[i]))
            y.append(-I[i]*np.sin(Y[i]))
        else:
            x.append(I[i]*np.cos(Y[i]))
            y.append(I[i]*np.sin(Y[i]))
        i += 1
'''yerror = []
for i in I:
    yerror.append(((0.015/0.409)+0.05)*i)'''
print(x)
plt.scatter(x, y, color="black", marker=".")
#plt.errorbar(x, y, yerr=yerror, label='meritve', barsabove="True", linestyle="None", color="black", capsize=2)
print(x,    y) 
plt.xlabel("$x$")
plt.ylabel("$y$")
#plt.axis([-1.7, 1.7, 0, 1])
plt.legend()
plt.title("Prepuščena moč skozi polarizator pri različnih kotih")
plt.savefig("Elipsa.png")
plt.show()
plt.close()
