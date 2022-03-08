import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file1 = os.path.join(THIS_FOLDER, "Napetost-tok,24.txt")
my_file2 = os.path.join(THIS_FOLDER, "Napetost-tok,43.txt")
my_file3 = os.path.join(THIS_FOLDER, "Napetost-tok,60.txt") 
#data load and manipulation
UBE1, IC11, IC12, IC13 = np.loadtxt(my_file1, delimiter="\t", unpack="True")
UBE2, IC21, IC22, IC23 = np.loadtxt(my_file2, delimiter="\t", unpack="True")
UBE3, IC31, IC32, IC33 = np.loadtxt(my_file3, delimiter="\t", unpack="True")

#TOKOVI BREZ EKSPONENTNEGA FITA

#Tok pri 24 °C
plt.plot(UBE1, IC11, color="black", marker=".")
plt.xlabel("U_BE [V]")
plt.ylabel("I_C [mA]")
plt.title("Kolektorski tok v odvisnosti od napetosti med bazo in emitorjem, 24 °C")
plt.savefig("Napetost-tok,24.png")
plt.show()
#Tok pri 43 °C
plt.plot(UBE2, IC22, color="green", marker=".")
plt.xlabel("U_BE [V]")
plt.ylabel("I_C [mA]")
plt.title("Kolektorski tok v odvisnosti od napetosti med bazo in emitorjem, 43 °C")
plt.savefig("Napetost-tok,43.png")
plt.show()
#Tok pri 60 °C
plt.plot(UBE3, IC33, color="blue", marker=".")
plt.xlabel("U_BE [V]")
plt.ylabel("I_C [mA]")
plt.title("Kolektorski tok v odvisnosti od napetosti med bazo in emitorjem, 60 °C")
plt.savefig("Napetost-tok,60.png")
plt.show()

#EKSPONENTNI FIT
e_0=1.6*10**(-19) #As
#Tok pri 24 °C
T1=297 #K             
f = lambda UBE1, kB1, IS1 : UBE1*e_0/(kB1*T1) + IS1
args = [1.38*10**(-23), 0]
fit = sp.curve_fit(f, UBE1, np.log(IC11/0.0004), p0=args)
fitlabel = "$%.2E \\frac{e_0}{T} * U_{BE} + %.2E$"%(fit[0][0], fit[0][1])
print(fitlabel)
t0 = np.linspace(0.4, 0.6, 1000)
plt.plot(UBE1, np.log(IC11/0.0004), color="green", marker=".")
plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
plt.xlabel("U_BE [V]")
plt.legend()
plt.ylabel("I_C [mA]")
plt.title("Kolektorski tok v odvisnosti od napetosti med bazo in emitorjem, 24 °C")
plt.savefig("Napetost-tok_fit,24.png")
plt.show()   

#Tok pri 43 °C
T2=316 #K             
f = lambda UBE2, kB2, IS2 : UBE2*e_0/(kB2*T2) + IS2
args = [1.38*10**(-23), 0]
fit = sp.curve_fit(f, UBE2, np.log(IC22/0.0016), p0=args)
fitlabel = "$%.4E \\frac{e_0}{T} * U_{BE} + %2.E$"%(fit[0][0], fit[0][1])
print(fitlabel)
t0 = np.linspace(0.4, 0.6, 1000)
plt.plot(UBE2, np.log(IC22/0.0016), color="green", marker=".")
plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
plt.xlabel("U_BE [V]")
plt.legend()
plt.ylabel("I_C [mA]")
plt.title("Kolektorski tok v odvisnosti od napetosti med bazo in emitorjem, 43 °C")
plt.savefig("Napetost-tok_fit,43.png")
plt.show()

#Tok pri 60 °C
T3=333 #K             
f = lambda UBE3, kB3, IS3 : UBE3*e_0/(kB3*T3) + IS3
args = [1.38*10**(-23), 0]
fit = sp.curve_fit(f, UBE3, np.log(IC33/0.0132), p0=args)
fitlabel = "$%.2E \\frac{e_0}{T} * U_{BE} + %.2E$"%(fit[0][0], fit[0][1])
print(fitlabel)
t0 = np.linspace(0.4, 0.6, 1000)
plt.plot(UBE3, np.log(IC33/0.0132), color="green", marker=".")
plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
plt.xlabel("U_BE [V]")
plt.legend()
plt.ylabel("I_C [mA]")
plt.title("Kolektorski tok v odvisnosti od napetosti med bazo in emitorjem, 60 °C")
plt.savefig("Napetost-tok_fit,60.png")
plt.show()  
