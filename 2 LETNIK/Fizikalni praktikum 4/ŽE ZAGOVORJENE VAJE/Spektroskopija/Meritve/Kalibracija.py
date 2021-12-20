import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file1 = os.path.join(THIS_FOLDER, "Hg spekter.txt")
my_file2 = os.path.join(THIS_FOLDER, "H2 spekter.txt")
my_file3 = os.path.join(THIS_FOLDER, "Skupni spekter.txt")
my_file4 = os.path.join(THIS_FOLDER, "Volfram.txt")
#data load and manipulation
Hg, pi_Hg= np.loadtxt(my_file1, delimiter="\t", unpack="True")
H2, pi_H2= np.loadtxt(my_file2, delimiter="\t", unpack="True")
Hg2, pi_Hg2= np.loadtxt(my_file3, delimiter="\t", unpack="True")
pi1_Vf, pi2_Vf=np.loadtxt(my_file4, delimiter="\t", unpack="True")
#EMISIJE BREZ FITA
#Hg spekter
plt.plot(Hg, pi_Hg, color="black", marker=".")
plt.xlabel("\u03BB [nm]")
plt.ylabel("\u03C6 [°]")
plt.title("Meritev odvisnotsi kota od valovne dolžine, Hg")
plt.savefig("Hg spekter.png")
#plt.show()

#H2 spekter
plt.plot(H2, pi_H2, color="black", marker=".")
plt.xlabel("\u03BB [nm]")
plt.ylabel("\u03C6 [°]")
plt.title("Meritev odvisnotsi kota od valovne dolžine, H_2")
plt.savefig("H2 spekter.png")
#plt.show()

#EKSPONENTNI FIT

#Hg spekter
f = lambda Hg, c1_Hg, c2_Hg, c3_Hg : c1_Hg+c2_Hg*Hg+c3_Hg*(Hg)**(1/2)
args = [0, 0, 0]
fit = sp.curve_fit(f, Hg, pi_Hg, p0=args)
fitlabel = "$%.2f + %.2f \cdot \u03BB  + %.2f \cdot np.sqrt(\u03BB)$"%(fit[0][0], fit[0][1], fit[0][2])
print(fitlabel)
t0 = np.linspace(435,579, 1000)
plt.plot(Hg, pi_Hg, color="green", marker=".")
plt.plot(t0, f(t0, fit[0][0], fit[0][1], fit[0][2]), label=fitlabel)
plt.legend()
plt.xlabel("\u03BB [nm]")
plt.ylabel("\u03C6 [°]")
plt.title("Meritev odvisnotsi kota od valovne dolžine, Hg ,fit")
plt.savefig("Hg spekter,fit.png")
#plt.show()

#H2 spekter
f = lambda H2, c1_H2, c2_H2, c3_H2 : c1_H2+c2_H2*H2+c3_H2*(H2)**(1/2)
args = [68, -0.14, 7.68]
fit = sp.curve_fit(f, H2, pi_H2, p0=args)
fitlabel = "$%.5f + %.5f \cdot \u03BB  + %.5f \cdot np.sqrt(\u03BB)$"%(fit[0][0], fit[0][1], fit[0][2])
print(fitlabel)
t0 = np.linspace(410,656, 1000)
plt.plot(H2, pi_H2, color="green", marker=".")
plt.plot(t0, f(t0, fit[0][0], fit[0][1], fit[0][2]), label=fitlabel)
plt.legend()
plt.xlabel("\u03BB [nm]")
plt.ylabel("\u03C6 [°]")
plt.title("Meritev odvisnotsi kota od valovne dolžine, H2 ,fit")
plt.savefig("H2 spekter,fit.png")
#plt.show()

#Volfram žarnica
print("VOLFRAM ŽARNICA")
lam1=[]
lam2=[]
lamtest=[]
for i in range(len(pi1_Vf)):
    lam1.append(((-7.6056+np.sqrt(7.6056**2+4*0.135*(69.5616-pi1_Vf[i])))/(-2*0.135))**2)
    lam2.append(((-7.6056+np.sqrt(7.6056**2+4*0.135*(69.5616-pi2_Vf[i])))/(-2*0.135))**2)
for i in range(len(lam1)):
    print("Valovna dolžina \u03C6_1 je: %.4f nm" %lam1[i])
print("\n")
for i in range(len(lam2)):
    print("Valovna dolžina \u03C6_2 je: %.4f nm" %lam2[i])
print("\n")

#Varčna žarnica
my_file5 = os.path.join(THIS_FOLDER, "Varčna.txt")
pi1_Va=np.loadtxt(my_file5, delimiter="\t", unpack="True")
print("VARČNA ŽARNICA")
lam3=[]
for i in range(len(pi1_Va)):
    lam3.append(((-7.6056+np.sqrt(7.6056**2+4*0.135*(69.5616-pi1_Va[i])))/(-2*0.135))**2)
for i in range(len(lam3)):
    print("Valovna dolžina \u03C6_1 je: %.4f nm" %lam3[i])
print("\n")
for i in range(len(lam3)):
    print(" // & %.4f nm & %.2f \\\ " %(lam3[i], pi1_Va[i]))
    print("\\hline")
print("\n")
#NO2
my_file6 = os.path.join(THIS_FOLDER, "NO2.txt")
pi1_NO2=np.loadtxt(my_file6, delimiter="\t", unpack="True")
print("NO_2")
lam4=[]
for i in range(len(pi1_NO2)):
    lam4.append(((-7.6056+np.sqrt(7.6056**2+4*0.135*(69.5616-pi1_NO2[i])))/(-2*0.135))**2)
for i in range(len(lam4)):
    print("Valovna dolžina \u03C6_1 je: %.4f nm" %lam4[i])
print("\n")
for i in range(len(lam4)):
    print(" // & %.4f nm & %.2f \\\ " %(lam4[i], pi1_NO2[i]))
    print("\\hline")
print("\n")
#NEON
my_file7 = os.path.join(THIS_FOLDER, "Neon.txt")
pi1_Ne=np.loadtxt(my_file7, delimiter="\t", unpack="True")
print("NEON")
lam5=[]
for i in range(len(pi1_Ne)):
    lam5.append(((-7.6056+np.sqrt(7.6056**2+4*0.135*(69.5616-pi1_Ne[i])))/(-2*0.135))**2)
for i in range(len(lam5)):
    print("Valovna dolžina \u03C6_1 je: %.4f nm" %lam5[i])
print("\n")
for i in range(len(lam5)):
    print(" // & %.4f nm & %.2f \\\ " %(lam5[i], pi1_Ne[i]))
    print("\\hline")
print("\n")
#HELIJ
my_file8 = os.path.join(THIS_FOLDER, "Helij.txt")
pi1_He=np.loadtxt(my_file8, delimiter="\t", unpack="True")
print("HELIJ")
lam6=[]
for i in range(len(pi1_He)):
    lam6.append(((-7.6056+np.sqrt(7.6056**2+4*0.135*(69.5616-pi1_He[i])))/(-2*0.135))**2)
for i in range(len(lam6)):
    print("Valovna dolžina \u03C6_1 je: %.4f nm \\\ " %lam6[i])
print("\n")
#LED
my_file9= os.path.join(THIS_FOLDER, "LED.txt")
pi1_LED, pi2_LED, pi3_LED=np.loadtxt(my_file9, delimiter="\t", unpack="True")
print("LED")
lam7=[]
lam8=[]
lam9=[]
for i in range(len(pi1_LED)):
    lam7.append(((-7.6056+np.sqrt(7.6056**2+4*0.135*(69.5616-pi1_LED[i])))/(-2*0.135))**2)
    lam8.append(((-7.6056+np.sqrt(7.6056**2+4*0.135*(69.5616-pi2_LED[i])))/(-2*0.135))**2)
    lam9.append(((-7.6056+np.sqrt(7.6056**2+4*0.135*(69.5616-pi3_LED[i])))/(-2*0.135))**2)
for i in range(len(lam7)):
    print("Valovna dolžina \u03C6_1 je: %.4f nm \\\ " %lam7[i])
    print("Valovna dolžina \u03C6_2 je: %.4f nm \\\ " %lam8[i])
    print("Valovna dolžina \u03C6_3 je: %.4f nm \\\ " %lam9[i])
print("\n")

print(" // \lambda_1 & \lambda_2 & \lambda_3 & \phi_1 & \phi_2 & \phi_3 \\\ ")

for i in range(len(lam7)):
    print(" // %.4f nm & %.4f nm & %.4f nm & %.2f & %.2f & %.2f \\\ " %(lam7[i], lam8[i], lam9[i], pi1_LED[i], pi2_LED[i], pi3_LED[i]))
    print("\\hline")
print("\n")
#Skupni spekter
f = lambda Hg2, c1_Hg2, c2_Hg2, c3_Hg2 : c1_Hg2+c2_Hg2*Hg2+c3_Hg2*(Hg2)**(1/2)
args = [68, -0.14, 7.68]
fit = sp.curve_fit(f, Hg2, pi_Hg2, p0=args)
fitlabel = "$%.4f + %.4f \cdot \u03BB  + %.4f \cdot np.sqrt(\u03BB)$"%(fit[0][0], fit[0][1], fit[0][2])
print(fitlabel)
t0 = np.linspace(410,656, 6000)
plt.plot(Hg2, pi_Hg2, color="green", marker=".")
plt.plot(t0, f(t0, fit[0][0], fit[0][1], fit[0][2]), label=fitlabel)
plt.legend()
plt.xlabel("\u03BB [nm]")
plt.ylabel("\u03C6 [°]")
plt.title("Meritev odvisnotsi kota od valovne dolžine, skupni ,fit")
plt.savefig("Skupni spekter,fit.png")
plt.show()






