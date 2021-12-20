import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import os
import shutil


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
#files=["\\U=15kW.txt","\\U=20kW.txt","\\U=25kW.txt"]

data = np.loadtxt(THIS_FOLDER + "\\Gamma.txt", dtype=float)
plt.title("Meritev števila trkov pri določeni energiji")
U_c, I_c=data.T
width=[]
for i in range(len(U_c)-1):
    width.append(abs(U_c[i+1]-U_c[i]))
width.append(1)
print(len(width))
print((len(U_c)))
plt.title("Meritev števila trkov pri določeni energiji")
plt.bar(U_c,I_c, width=width,label="Meritev Na22",edgecolor="black",align="edge")
ax = plt.gca()
ax.set_xticks([1.4,3.4,4.9])
ax.set_xticklabels(["511 keV","1274 keV","1813 keV"])
plt.legend()
plt.show()

dataCs = np.loadtxt(THIS_FOLDER + "\\Cs-137.mca", dtype=float)
dataCo = np.loadtxt(THIS_FOLDER + "\\CO-60.mca", dtype=float)
dataNa = np.loadtxt(THIS_FOLDER + "\\Na22.mca", dtype=float)
databack = np.loadtxt(THIS_FOLDER + "\\Back.mca", dtype=float)
plt.title("Meritev števila trkov pri določeni energiji")
U_cNa=dataNa.T
U_Co=dataCo.T
U_Cs=dataCs.T
U_cNaB=dataNa.T-databack.T
U_CoB=dataCo.T-databack.T
U_CsB=dataCs.T-databack.T
n=[]
for i in range(len(U_cNa)):
    n.append(i*2.29)
print(n,U_cNa)

plt.title("Meritev števila trkov pri določeni energiji z ozadjem in brez ozadja")
plt.bar(n,U_cNa, width=1*2.29,label="Meritev Na-22",align="edge")
plt.bar(n,U_cNaB, width=1*2.29,label="Meritev Na-22 brez ozadja",align="edge")
ax = plt.gca()
ax.set_xticks([511,1274,1813])
ax.set_xticklabels(["511 keV","1274 keV","1813 keV"])
plt.legend()
plt.show()

nC=[]
for i in range(len(U_cNa)):
    nC.append(i*1.7)

plt.title("Meritev števila trkov pri določeni energiji z ozadjem in brez ozadja")
plt.bar(nC,U_Co, width=1*1.85,label="Meritev Co-60",align="edge")
plt.bar(nC,U_CoB, width=1*1.85,label="Meritev Co-60 brez ozadja",align="edge")
ax = plt.gca()
#ax.set_xticks([1484,1683])
#ax.set_xticklabels(["1484 keV","1683 keV"])
plt.legend()
plt.show()


nS=[]
for i in range(len(U_cNa)):
    nS.append(i*1.8)

plt.title("Meritev števila trkov pri določeni energiji z ozadjem in brez ozadja")
plt.bar(nS,U_Cs, width=1*1.8,label="Meritev Cs-137",align="edge")
plt.bar(nS,U_CsB, width=1*1.8,label="Meritev Cs-137 brez ozadja",align="edge")
ax = plt.gca()
#ax.set_xticks([835])
#ax.set_xticklabels(["835 keV"])
plt.legend()
plt.show()

plt.title("Meritev števila trkov pri določeni energiji brez ozadja")
plt.bar(n,U_cNaB, width=1*2.29,label="Meritev Na-22",align="edge")
plt.bar(nC,U_CoB, width=1*1.85,label="Meritev Co-60",align="edge")
plt.bar(nS,U_CsB, width=1*1.8,label="Meritev Cs-137",align="edge")
ax = plt.gca()
#ax.set_xticks([511,835,1274,1484,1683,1813])
#ax.set_xticklabels(["511 keV","835 keV","1274 keV","1484 keV","1683 keV","1813keV"])
plt.legend()
plt.show()

vrhnNa=n[180:260]
vrhoviNa=U_cNaB[180:260]

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

vrhnNa = np.asarray(vrhnNa)
vrhoviNa = np.asarray(vrhoviNa)
# Define the Gaussian function
def Gauss(x, A, B,x_0,c):
    y = A*np.exp(-1*(x-x_0)**2/(2*B**2))+c
    return y
parametersNa1, covarianceNa1 = curve_fit(Gauss, vrhnNa, vrhoviNa,p0=[30000,1,510,0])

fit_A1 = parametersNa1[0]
fit_B1 = parametersNa1[1]
fit_x_01 = parametersNa1[2]
fit_c1= parametersNa1[3]


vrhnNa3=n[730:850]
vrhoviNa3=U_cNaB[730:850]
vrhnNa3 = np.asarray(vrhnNa3)
vrhoviNa3 = np.asarray(vrhoviNa3)
parametersNa3, covarianceNa3 = curve_fit(Gauss, vrhnNa3, vrhoviNa3,p0=[558,20,1813,0])

fit_A3 = parametersNa3[0]
fit_B3 = parametersNa3[1]
fit_x_03 = parametersNa3[2]
fit_c3 = parametersNa3[3]
 

vrhnNa2=n[510:600]
vrhoviNa2=U_cNaB[510:600]
vrhnNa2 = np.asarray(vrhnNa2)
vrhoviNa2 = np.asarray(vrhoviNa2)
parametersNa2, covarianceNa2 = curve_fit(Gauss, vrhnNa2, vrhoviNa2,p0=[2500,20,1270,0])

fit_A2 = parametersNa2[0]
fit_B2 = parametersNa2[1]
fit_x_02 = parametersNa2[2]
fit_c2 = parametersNa2[3]


vrhnCo1=n[600:700]
vrhoviCo1=U_CoB[600:700]
vrhnCo1 = np.asarray(vrhnCo1)
vrhoviCo1 = np.asarray(vrhoviCo1)
parametersCo1, covarianceCo1 = curve_fit(Gauss, vrhnCo1, vrhoviCo1,p0=[4600,20,1484,0])

fit_ACo1 = parametersCo1[0]
fit_BCo1 = parametersCo1[1]
fit_x_0Co1 = parametersCo1[2]
fit_cCo1 = parametersCo1[3]

vrhnCo2=n[685:810]
vrhoviCo2=U_CoB[685:810]
vrhnCo2 = np.asarray(vrhnCo2)
vrhoviCo2 = np.asarray(vrhoviCo2)
parametersCo2, covarianceCo2 = curve_fit(Gauss, vrhnCo2, vrhoviCo2,p0=[3000,20,1638,0])

fit_ACo2 = parametersCo2[0]
fit_BCo2 = parametersCo2[1]
fit_x_0Co2 = parametersCo2[2]
fit_cCo2 = parametersCo2[3]
 
fit_y1 = Gauss(vrhnNa, fit_A1, fit_B1, fit_x_01, fit_c1)
fit_y2 = Gauss(vrhnNa2, fit_A2, fit_B2, fit_x_02, fit_c2)
fit_y3 = Gauss(vrhnNa3, fit_A3, fit_B3, fit_x_03, fit_c3)
fit_yCo1 = Gauss(vrhnCo1, fit_ACo1, fit_BCo1, fit_x_0Co1, fit_cCo1)
fit_yCo2 = Gauss(vrhnCo2, fit_ACo2, fit_BCo2, fit_x_0Co2, fit_cCo2)
plt.title("Meritev števila trkov pri določeni energiji, fitanje Gaussove krivulje")
plt.plot(vrhnNa, vrhoviNa, 'o', label="Vrh Natrija 1")
plt.plot(vrhnNa, fit_y1, '-', label=r'Gaussov fit Vrh Natrija 1, $\sigma$=%.5f,$E$=%.5f' %(fit_B1, fit_x_01))
plt.plot(vrhnNa2, vrhoviNa2, 'o', label="Vrh Natrija 2")
plt.plot(vrhnNa2, fit_y2, '-', label=r'Gaussov fit Vrh Natrija 2, $\sigma$=%.5f,$E$=%.5f' %(fit_B2, fit_x_02))
plt.plot(vrhnNa3, vrhoviNa3, 'o', label="Vrh Natrija 3")
plt.plot(vrhnNa3, fit_y3, '-', label=r'Gaussov fit Vrh Natrija 3, $\sigma$=%.5f,$E$=%.5f' %(fit_B3, fit_x_03))
plt.plot(vrhnCo1, vrhoviCo1, 'o', label="Vrh Kobalta 1")
plt.plot(vrhnCo1, fit_yCo1, '-', label=r'Gaussov fit Vrh Kobalta 1, $\sigma$=%.5f,$E$=%.5f' %(fit_BCo1, fit_x_0Co1))
plt.plot(vrhnCo2, vrhoviCo2, 'o', label="Vrh Kobalta 1")
plt.plot(vrhnCo2, fit_yCo2, '-', label=r'Gaussov fit Vrh Kobalta 1, $\sigma$=%.5f,$E$=%.5f' %(fit_BCo2, fit_x_0Co2))
plt.legend()
ax = plt.gca()
ax.set_xticks([fit_x_01,fit_x_02,fit_x_03,fit_x_0Co1,fit_x_0Co2])
ax.set_xticklabels([str(round(fit_x_01,1))+"keV",str(round(fit_x_02,1))+"keV",str(round(fit_x_03,1))+"keV",str(round(fit_x_0Co1,1))+"keV",str(round(fit_x_0Co2,1))+"keV"])
plt.show()
energy=[fit_x_01,fit_x_02,fit_x_0Co1,fit_x_0Co2,fit_x_03]
sigma=[fit_B1,fit_B2,fit_BCo1,fit_BCo2,fit_B3]
def Kvadrat(x,A):
    y= A / np.sqrt(x)
    return y
energy= np.asarray(energy)
sigma = np.asarray(sigma)
parameters, covariance = curve_fit(Kvadrat, energy, sigma/energy,p0=[1])
fit_y1 = Kvadrat(n[100:],parameters[0])
plt.title("Prikaz odvisnosti ločljivosti vrhov od energije")
plt.plot(energy, sigma/energy, 'o', label="Podatki")
plt.plot(n[100:], fit_y1, '-', label=r"Fit $R=\frac{%.4f}{\sqrt{E}}$" %(parameters[0]))
plt.legend()
plt.show()