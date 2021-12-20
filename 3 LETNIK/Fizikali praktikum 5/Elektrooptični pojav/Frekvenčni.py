import numpy as np
import numpy.linalg as lin
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.optimize as sp
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data_file = THIS_FOLDER + "\Modulacija.txt"
data= np.loadtxt(data_file)
data=data.T
U=data[0]
X_0=data[1]
Y_0=data[2]
plt.title("Prikaz odvisnosti komponente X in Y v odvisnosti od napetosti U")
plt.plot(U,X_0, label="X komponenta" )
plt.plot(U,Y_0, label="Y komponenta" )
plt.ylabel("X[mV]/Y[mV]")
plt.xlabel("U[V]")
plt.legend()
plt.show()
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data_file = THIS_FOLDER + "\Frekvencni.txt"
data= np.loadtxt(data_file)
data=data.T
nu=data[0]
X_0=data[1]
Y_0=data[2]
X_90=data[3]
Y_90=data[4]
print(len(Y_90))
plt.title("Prikaz odvisnosti komponente X in Y v fazi spreminjanja frekvenc")
plt.plot(X_0,Y_0, color="Green",label="Faza pri 0°")
plt.plot(X_0[0],Y_0[0], "x", color="Black",  label=r"Faza pri 0°, $\nu=%.2f$" %(nu[0]) )
plt.plot(X_90[0],Y_90[0], "x", color="Black",  label=r"Faza pri 90°, $\nu=%.2f$" %(nu[0]) )
plt.plot(X_0[10],Y_0[10], "x", color="Red",  label=r"Faza pri 0°, $\nu=%.2f$" %(nu[10]) )
plt.plot(X_90[10],Y_90[10], "x", color="Red",  label=r"Faza pri 90°, $\nu=%.2f$" %(nu[10]) )
plt.plot(X_0[30],Y_0[30], "x", color="Purple",  label=r"Faza pri 0°, $\nu=%.2f$" %(nu[30]) )
plt.plot(X_90[30],Y_90[30], "x", color="Purple",  label=r"Faza pri 90°, $\nu=%.2f$" %(nu[30]) )
plt.plot(X_0[50],Y_0[50], "x", color="Orange",  label=r"Faza pri 0°, $\nu=%.2f$" %(nu[50]) )
plt.plot(X_90[50],Y_90[50], "x", color="Orange",  label=r"Faza pri 90°, $\nu=%.2f$" %(nu[50]) )
plt.plot(X_90,Y_90, color="Blue",label="Faza pri 90°")
plt.xlabel("X[V]")
plt.ylabel("Y[V]")
plt.legend()
plt.show()


plt.subplot(2,1,1)
plt.title("Prikaz odvisnosti komponente X in Y v fazi spreminjanja frekvenc")
plt.plot(X_0,Y_0, color="Green",label="Faza pri 0°")
plt.plot(X_0[0],Y_0[0], "x", color="Black",  label=r"Faza pri 0°, $\nu=%.2f$" %(nu[0]) )
plt.plot(X_0[10],Y_0[10], "x", color="Red",  label=r"Faza pri 0°, $\nu=%.2f$" %(nu[10]) )
plt.plot(X_0[30],Y_0[30], "x", color="Purple",  label=r"Faza pri 0°, $\nu=%.2f$" %(nu[30]) )
plt.plot(X_0[50],Y_0[50], "x", color="Orange",  label=r"Faza pri 0°, $\nu=%.2f$" %(nu[50]) )
plt.xlabel("X[V]")
plt.ylabel("Y[V]")
plt.legend()
plt.subplot(2,1,2)
plt.plot(X_90,Y_90, color="Blue",label="Faza pri 90°")
plt.plot(X_90[0],Y_90[0], "x", color="Black",  label=r"Faza pri 90°, $\nu=%.2f$" %(nu[0]) )
plt.plot(X_90[10],Y_90[10], "x", color="Red",  label=r"Faza pri 90°, $\nu=%.2f$" %(nu[10]) )
plt.plot(X_90[30],Y_90[30], "x", color="Purple",  label=r"Faza pri 90°, $\nu=%.2f$" %(nu[30]) )
plt.plot(X_90[50],Y_90[50], "x", color="Orange",  label=r"Faza pri 90°, $\nu=%.2f$" %(nu[50]) )
plt.xlabel("X[V]")
plt.ylabel("Y[V]")
plt.legend()
plt.show()

f = lambda x, k, n : k*x+n
args = [0, 0]
fit = sp.curve_fit(f, nu[:30], np.array(Y_0[:30])/np.array(X_0[:30]), p0=args)
fitlabel = r"$\frac{Y}{X}(\nu)=%.5f \nu+ %.2f $"%(fit[0][0], fit[0][1])
plt.title("Prikaz diagrama odvisnosti razmerja Y/X pri določeni frekvenci")
plt.plot(nu[:30],np.array(Y_0[:30])/np.array(X_0[:30]),label="Faza pri 0°")
plt.plot(nu[:30], f(nu[:30], fit[0][0], fit[0][1]), label=fitlabel)
plt.xlabel(r"\nu[Hz]")
plt.ylabel(r"Y/X")
plt.legend()
plt.show()