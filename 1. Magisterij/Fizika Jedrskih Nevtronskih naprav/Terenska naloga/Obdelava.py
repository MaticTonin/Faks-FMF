
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
t2, N2, bio=data2.T

#Prvi del vaje

A=[]
line1=[]
line2=[]
line3=[]
line4=[]
line5=[]
xdiscrimination=[]
discrimination=[]
for i in range(len(N1)):
    A.append(np.log(N1[i]/t1[i]))
    xdiscrimination.append(0.61)
    discrimination.append(8.14)
    line3.append(8.3)
    line2.append(7.3)
    line1.append(0.5)
    line4.append(9.0)
    line5.append(10.0)

plt.title("Prikaz odvisnosti detekcij z večanjem spodnje meje")
#plt.plot(LOWER,A,"x-")
plt.errorbar(LOWER,A,yerr=0.1,xerr=0.1, fmt='.--', ecolor="black", color="red", alpha=0.4)
plt.fill_betweenx(LOWER+2, 0, line1, color='gray', alpha=0.4)
plt.text(0,6, r"$X$ rays")
plt.text(9.1,10, r"He$^{4}$ reaction")
plt.text(9.1,9.2, r"E=2.78 MeV")
plt.text(7.35,10, r"Li$^{7}$ reaction")
plt.text(7.35,9.2, r"E=2.30 MeV")
plt.fill_betweenx(LOWER+2, line2, line3, color='gray', alpha=0.4)
plt.fill_betweenx(LOWER+2, line4, line5, color='gray', alpha=0.4)
plt.plot(line1,A, "--", color="gray")
plt.plot(line2,A, "--", color="gray")
plt.plot(line3,A, "--", color="gray")
plt.plot(line4,A, "--", color="gray")
plt.plot(line5,A, "--", color="gray")
plt.plot(LOWER,discrimination, "--", color="black", alpha=0.7)
plt.plot(xdiscrimination,A, "--", color="black", alpha=0.7)
plt.text(0.8,8.65, r"Nivo diskriminacije")
plt.text(0.8,8.25, r"T(0.61 V,8,14 [1/s])")
plt.legend()

plt.xlabel(r"$U_{LOWER}[V]$")
plt.ylabel(r"$\log(A)[1/s]$")
plt.show()
lower_abs=[]
A_abs=[]
line1=[]
line2=[]
line3=[]
line4=[]
line5=[]
xlines=np.linspace(0,1, len(LOWER)-1)
for i in range(len(LOWER)):
    if i!=0:
        lower_abs.append(LOWER[i])
        A_abs.append(abs(A[i]-A[i-1]))
        line3.append(8.3)
        line2.append(7.3)
        line1.append(0.5)
        line4.append(9.0)
        line5.append(10.0)
lower_abs=np.array(lower_abs)
A_abs=np.array(A_abs)/max(A_abs)
plt.title("Prikaz odvisnosti detekcij znotraj intervalov")
plt.plot(lower_abs,A_abs,"x-")
plt.plot(lower_abs,A_abs,".", color="red")
#plt.errorbar(lower_abs,A_abs,yerr=0.05,xerr=0.05, fmt='.--', ecolor="black", color="red", alpha=0.4)
plt.text(0,0.3, r"$X$ rays")
plt.text(9.05,0.8, r"He$^{4}$ reaction")
plt.text(9.05,0.95, r"E=2.78 MeV")
plt.text(7.35,0.8, r"Li$^{7}$ reaction")
plt.text(7.35,0.95, r"E=2.30 MeV")
plt.fill_betweenx(xlines, 0, line1, color='gray', alpha=0.4)
plt.fill_betweenx(xlines, line2, line3, color='gray', alpha=0.4)
plt.fill_betweenx(xlines, line4, line5, color='gray', alpha=0.4)
plt.plot(line1,xlines, "--", color="gray")
plt.plot(line2,xlines, "--", color="gray")
plt.plot(line3,xlines, "--", color="gray")
plt.plot(line4,xlines, "--", color="gray")
plt.plot(line5,xlines, "--", color="gray")
plt.legend()
plt.xlabel(r"$U_{LOWER}[V]$")
plt.ylabel(r"$A[1/s]$")
plt.show()


#Drugi del vaje
U=np.array(N2)
U=N2-12.6
U=N2/50
tau_rocna=bio
f1= lambda x, M0, T1 : M0*np.exp(-x*T1)
#f1= lambda x, M0, T1 : M0*np.exp(-2*x/T1)
args = [26,0.0032]
fit1 = sp.curve_fit(f1, tau_rocna, U, p0=args)
fitlabel = r"$U=%.2f \exp( -%.6f \cdot t) $"%(fit1[0][0], fit1[0][1])
plt.title("Meritev aktivacije Vanadija")
plt.errorbar(tau_rocna,U,yerr=2,xerr=1, fmt='.--', ecolor="black", color="red", alpha=0.7)
#plt.plot(tau1,U1,"x-")
#plt.plot(tau_rocna, f1(tau_rocna, 8, 3.69,fit1[0][2]), label=fitlabel)
plt.plot(tau_rocna, f1(tau_rocna, fit1[0][0], fit1[0][1]), label=fitlabel)
plt.legend()
plt.xlabel(r"$t[s]$")
plt.ylabel(r"$A[1/s]$")
plt.show()


U=np.array(N2)
U=N2-12.6
U=N2/50
delta=np.linspace(-1.5,1.5,len(U))
U1=[]
U2=[]

for i in range(len(U)):
    U1.append(U[i]-delta[i])
tau_rocna=bio
f1= lambda x, M0, T1 : M0*np.exp(-x*T1)
#f1= lambda x, M0, T1 : M0*np.exp(-2*x/T1)
args = [26,0.0032]
fit2 = sp.curve_fit(f1, tau_rocna, U1, p0=args)
fitlabel1 = r"Glavni fit: $U=%.2f \exp( -%.6f \cdot t) $, $t_{\frac{1}{2}}=%.6f$ min"%(fit1[0][0], fit1[0][1], np.log(2)/(fit1[0][1]*60))
fitlabel2 = r"Fit z napako: $U=%.2f \exp( -%.6f \cdot t) $, $t_{\frac{1}{2}}=%.6f$ min"%(fit2[0][0], fit2[0][1], np.log(2)/(fit2[0][1]*60))
plt.title("Meritev aktivacije Vanadija, z napačnim fitom")
plt.errorbar(tau_rocna,U,yerr=2,xerr=1, fmt='.--', ecolor="black", color="red", alpha=0.7)
#plt.plot(tau1,U1,"x-")
#plt.plot(tau_rocna, f1(tau_rocna, 8, 3.69,fit1[0][2]), label=fitlabel)
plt.plot(tau_rocna, f1(tau_rocna, fit1[0][0], fit1[0][1]), label=fitlabel1)
plt.plot(tau_rocna, f1(tau_rocna, fit2[0][0], fit2[0][1]), label=fitlabel2)

plt.legend()
plt.xlabel(r"$t[s]$")
plt.ylabel(r"$A[1/s]$")
plt.show()


t=np.linspace(0,3600, 3600)
plt.title("Prikaz aktivacije Vanadija po 1h")
A_back=[]
A=f1(t, fit1[0][0], fit1[0][1])
t_min=0
A_min=0
for i in range(len(t)):
    A_back.append(0.25)
for i in range(len(t)):
    if A[i]<A_back[i]:
        t_min=t[i]
        A_min=A[i]
        break
        

#plt.plot(tau1,U1,"x-")
#plt.plot(tau_rocna, f1(tau_rocna, 8, 3.69,fit1[0][2]), label=fitlabel)
plt.plot(t, A, "-", label=fitlabel, color="green")
plt.plot(t, A_back, "--", label="Ozadje A=0.25 1/s", color="gray", alpha=0.7)
plt.scatter(t_min,A_min, color="red", label="Meja aktivacije")
plt.text(1800, 2.5, "Točka:")
plt.text(1600, 1.5, "t=%.2f, A=%.4f" %(t_min,A_min))
plt.legend()
plt.xlabel(r"$t[s]$")
plt.ylabel(r"$A[1/s]$")
plt.show()
