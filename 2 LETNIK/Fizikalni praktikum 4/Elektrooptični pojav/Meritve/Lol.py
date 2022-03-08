import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "lc_elip.txt") 

#data load and manipulation
x, y = np.loadtxt(my_file, delimiter=" ", unpack="True")
    
y /= 100
x /= 1000
I0 = 0.409
y/=I0
y*=1000
izkoristek=[]
for i in y:
    izkoristek.append(i)
print(izkoristek)


#plotting
#plt.ylim(-1e-4, 1e-3)
plt.scatter(x, y, color="black", marker=".")
#plt.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
#plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)

plt.xlabel("I[A]")
plt.ylabel("F[N]")
plt.legend()
plt.title("F(I)")
plt.axis([-0.5, 0.5, 0, 0.0001])
#plt.savefig("Sila v odvisnosti toka.png")
plt.show()
