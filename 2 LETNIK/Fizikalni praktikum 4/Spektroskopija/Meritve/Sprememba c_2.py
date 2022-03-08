import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
#NEON
my_file7 = os.path.join(THIS_FOLDER, "Neon,napake.txt")
pi1_Ne, lamb=np.loadtxt(my_file7, delimiter="\t", unpack="True")
print("NEON")
lam5=[]
lam51=[]
lam1=[]
delta=[]
for i in range(len(pi1_Ne)):
    lam1.append(lamb[i])
    lam5.append(((-7.6056+np.sqrt(7.6056**2+4*0.135*(69.5616-pi1_Ne[i])))/(-2*0.135))**2)
    lam51.append(((-7.6156+np.sqrt(7.6156**2+4*0.136*(69.7-pi1_Ne[i])))/(-2*0.136))**2)
for i in range(len(lam5)):
    print("Valovna dolžina \u03C6_1 je: %.4f nm" %lam5[i])
for i in range(len(lam5)):
    delta.append(lam51[i]-lam5[i])
print("\n")
print("Barva & Tablična $\lambda$ & $\lambda$ & Napaka $\lambda$ & $\Delta \lambda$ \\\ ")
print("\hline")
print("\hline")
for i in range(len(lam5)):
    print(" // & %.2f nm & %.2f nm & %.2f nm & %.2f nm \\\ " %(lam1[i], lam5[i], lam51[i], delta[i]))
    print("\\hline")
print("\n")







