import numpy as np
import numpy.linalg as lin
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.optimize as sp
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data_file = THIS_FOLDER + "\Meritev T1.txt"
data1= np.loadtxt(data_file)
tau, tau_rocna, U=data1.T
data_file = THIS_FOLDER + "\Meritev Vodovodna.txt"
data2= np.loadtxt(data_file)
print(data2)
tau1, U1=data2.T

f = lambda x, k, n : k*x+n
args = [0, 0]
fit = sp.curve_fit(f, tau, tau_rocna, p0=args)
fitlabel = r"$\tau=%.5f  \tau+ %.2f $"%(fit[0][0], fit[0][1])
plt.title("Umeritev naprave z zamikom")
plt.plot(tau,tau_rocna,"x-")
plt.plot(tau, f(tau, fit[0][0], fit[0][1]), label=fitlabel)
plt.legend()
plt.xlabel(r"$\tau[ms]$")
plt.ylabel(r"$\tau_{rocna}[ms]$")
plt.show()



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
plt.show()