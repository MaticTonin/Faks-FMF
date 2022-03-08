import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file1 = os.path.join(THIS_FOLDER, "Napetost-temperatura,0,5.txt")
my_file2 = os.path.join(THIS_FOLDER, "Napetost-temperatura,0,57.txt")
 
#data load and manipulation
T1, dT1, IC11, IC12= np.loadtxt(my_file1, delimiter="\t", unpack="True")
T2, dT2, IC21, IC22= np.loadtxt(my_file2, delimiter="\t", unpack="True")

#TOKOVI BREZ EKSPONENTNEGA FITA
T1+=273
#Napetost pri 0.5 V
plt.plot(T1, IC11, color="black", marker=".")
plt.xlabel("T [K]")
plt.ylabel("I_C [mA]")
plt.title("Meritev odvisnosti kolektorskega toka od temperature,0.5 V")
plt.savefig("Napetost-temperatura,0,5.png")
plt.show()
#Napetost pri 0.57 V
T2+=273
plt.plot(T2, IC22, color="blue", marker=".")
plt.xlabel("T [K]")
plt.ylabel("I_C [mA]")
plt.title("Meritev odvisnosti kolektorskega toka od temperature,0.5 V")
plt.savefig("Napetost-temperatura,0,57.png")
plt.show()

#EKSPONENTNI FIT
e_0=1.6*10**(-19) #As
#Napetost pri 0.5 V
UBE1=0.5
#f = lambda T1, kB1, IS1 : UBE1*e_0/(kB1)*1/T1 + IS1
#args = [1.38*10**(-23), 0]
#fit = sp.curve_fit(f, T1, np.log(IC11/0.7), p0=args)
#fitlabel = "$%.2E \\frac{e_0}{T} * U_{BE} + %.2E$"%(fit[0][0], fit[0][1])
#print(fitlabel)
#t0 = np.linspace(290, 350, 1000)
plt.plot(T1, np.log(IC11/0.7), color="green", marker=".")
#plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
plt.xlabel("T [K]")
plt.legend()
plt.ylabel("I_C [mA]")
plt.title("Meritev odvisnosti kolektorskega toka od temperature,0.5 V")
plt.savefig("Napetost-temperatura_fut,0,5.png")
plt.show()


#Napetost pri 0.57 V
UBE2=0.57
#f = lambda T1, kB1, IS1 : UBE1*e_0/(kB1)*1/T1 + IS1
#args = [1.38*10**(-23), 0]
#fit = sp.curve_fit(f, T1, np.log(IC11/0.7), p0=args)
#fitlabel = "$%.2E \\frac{e_0}{T} * U_{BE} + %.2E$"%(fit[0][0], fit[0][1])
#print(fitlabel)
#t0 = np.linspace(290, 350, 1000)
plt.plot(T2, np.log(IC22/0.7), color="blue", marker=".")
#plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
plt.xlabel("T [K]")
plt.legend()
plt.ylabel("I_C [mA]")
plt.title("Meritev odvisnosti kolektorskega toka od temperature,0.57 V")
plt.savefig("Napetost-temperatura_fut,0,57.png")
plt.show()


