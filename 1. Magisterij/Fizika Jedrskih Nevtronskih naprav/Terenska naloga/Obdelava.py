import numpy as np
import numpy.linalg as lin
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.optimize as sp
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data_file = THIS_FOLDER + "\Vaja 1.txt"
data1= np.loadtxt(data_file)
LOWER, N1, t1=data1.T
data_file= THIS_FOLDER + "\Vaja 2.txt"
data2= np.loadtxt(data_file)
print(data2)
t2, N2, bio=data2.T

#Prvi del vaje

A=[]
for i in range(len(N1)):
    A.append(np.log(N1[i]/t1[i]))

plt.title("Prikaz odvisnosti detekcij z večanjem spodnje meje")
plt.plot(LOWER,A,"x-")
plt.legend()
plt.xlabel(r"$\log(U_{LOWER})[V]$")
plt.ylabel(r"$A[1/s]$")
plt.show()
lower_abs=[]
A_abs=[]
for i in range(len(LOWER)):
    if i!=0:
        lower_abs.append(LOWER[i])
        A_abs.append(abs(A[i]-A[i-1]))

plt.title("Prikaz odvisnosti detekcij z večanjem spodnje meje")
plt.plot(lower_abs,A_abs,"x-")
plt.legend()
plt.xlabel(r"$\log(U_{LOWER})[V]$")
plt.ylabel(r"$A[1/s]$")
plt.show()

#Drugi del vaje
"""
f1= lambda x, M0, T1 : M0*(1-np.exp(-x/T1))
#f1= lambda x, M0, T1 : M0*np.exp(-2*x/T1)
args = [10, 3.69]
fit1 = sp.curve_fit(f1, tau_rocna, U, p0=args)
fitlabel = r"$U=%.2f(1-exp(\frac{\tau}{%.2f}) $"%(fit1[0][0], fit1[0][1])
plt.title("Meritev ionizirane vode")
plt.plot(tau_rocna,U,"x-")
#plt.plot(tau1,U1,"x-")
#plt.plot(tau_rocna, f1(tau_rocna, 8, 3.69,fit1[0][2]), label=fitlabel)
plt.plot(tau_rocna, f1(tau_rocna, fit1[0][0], fit1[0][1]), label=fitlabel)
plt.legend()
plt.xlabel(r"$\tau[ms]$")
plt.ylabel(r"$U[V]$")
plt.show()


tau_rocna=tau1
tau_rocna=f(tau1,fit[0][0],fit[0][1])*0.001
f12= lambda x, M0, T1 : M0*(1-np.exp(-x/T1))
#f1= lambda x, M0, T1 : M0*np.exp(-2*x/T1)
args = [1,1]
fit12 = sp.curve_fit(f12, tau_rocna, U1, p0=args)
fitlabel = r"$U=%.2f(1-exp(\frac{\tau}{%.2f}) $"%(fit12[0][0], fit12[0][1])
plt.title("Meritev vodovodne vode")
plt.plot(tau_rocna,U1,"x-")
#plt.plot(tau1,U1,"x-")
#plt.plot(tau_rocna, f1(tau_rocna, 8.1, 3540), label=fitlabel)
plt.plot(tau_rocna, f12(tau_rocna, fit12[0][0], fit12[0][1]), label=fitlabel)
plt.legend()
plt.xlabel(r"$\tau[ms]$")
plt.ylabel(r"$U[V]$")
plt.show()"""