import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import os
import shutil


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
files=["\\U=15kW.txt","\\U=20kW.txt","\\U=25kW.txt","\\U=30kW.txt", "\\U=35kW.txt"]
#files=["\\U=15kW.txt","\\U=20kW.txt","\\U=25kW.txt"]
max_X=[]
max_Y=[]
for i in files:
    data = np.loadtxt(THIS_FOLDER + i, dtype=float)
    plt.title("Prikaz odvisnosti napetosti na kondenzatorju v odvisnost od napetosti na celici")
    plt.xlabel("$U_C[V]$")
    plt.ylabel("$U_E[V]$")
    U_c, I_c=data.T
    for j in range(len(U_c)):
        if abs(I_c[j-1]-I_c[j])<=0.1 and j>0:
            max_X.append(U_c[j])
            max_Y.append(I_c[j])
            break
    plt.plot(U_c,I_c, "x-", label=i[1:])
plt.legend()
plt.show()
f = lambda x, k, n : k*x+n
args = [0, 0]
fit = sp.curve_fit(f,max_X, max_Y, p0=args)
fitlabel = r"$U_E=%.5f  U_c %.2f $"%(fit[0][0], fit[0][1])
plt.title("Umeritev naprave z zamikom")
print(max_X)
plt.plot(max_X, f(np.array(max_X), fit[0][0], fit[0][1]), label=fitlabel)
plt.title("Prikaz odvisnosti maksimalne napetosti na kondenzatorju v odvisnost od napetosti na celici")
plt.xlabel("$U_C[V]$")
plt.ylabel("$U_E[V]$")
plt.plot(max_X,max_Y, "x")
plt.legend()
plt.show()


data=np.loadtxt(THIS_FOLDER + "\\Itenziteta, enkratna.txt", dtype=float)
I_x,I_z=data.T
eta_avg=0
eta_sum=0
eta=[]
for i in range(len(I_x)):
    eta.append(abs(I_x[i]-I_z[i])/(I_z[i]+I_x[i]))
    eta_sum+=(abs(I_x[i]-I_z[i])/(I_z[i]+I_x[i]))
eta_avg=eta_sum/len(I_x)
print(eta_avg)
eta_plt=[]
for i in range(len(I_x)):
    eta_plt.append(eta_avg)
plt.subplot(2,1,1)
plt.title("Prikaz izračuna polarizacije za posamezno postavitev")
plt.plot(I_x,eta_plt, "-", color="black", label="Avg: %.4f" %(eta_avg))
plt.plot(I_x,eta, "x", label="Enkratna")
plt.legend()
data=np.loadtxt(THIS_FOLDER + "\\Itenziteta, dvakratna.txt", dtype=float)
I_x,I_z=data.T
eta_avg=0
eta_sum=0
eta=[]
for i in range(len(I_x)):
    eta.append(abs(I_x[i]-I_z[i])/(I_z[i]+I_x[i]))
    eta_sum+=(abs(I_x[i]-I_z[i])/(I_z[i]+I_x[i]))
eta_avg=eta_sum/len(I_x)
eta_plt=[]
for i in range(len(I_x)):
    eta_plt.append(eta_avg)
plt.subplot(2,1,2)
plt.plot(I_x,eta_plt, "-", color="black", label="Avg: %.4f" %(eta_avg))
plt.plot(I_x,eta, "x", label="Sipalna")
plt.legend()
plt.show()
print(eta_avg)