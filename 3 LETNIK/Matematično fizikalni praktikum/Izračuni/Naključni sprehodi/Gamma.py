import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation
from scipy.optimize import curve_fit

muFZ=[1.5,1.7,2, 2.2, 2.5, 3, 3.5, 4]
gammaFZ=[3.899448407, 2.814994669215, 2.0097504755, 1.705273316177, 1.443856248487304, 1.21158111495, 1.22157248, 1.10978323]
muWZ=[1.5,1.7,2, 2.2, 2.5, 3, 3.5, 4]
gammaWZ=[1.8992587,1.69877786, 1.656984465644, 1.720024822621229, 1.91862263125815, 1.81103999, 1.700008094151, 1.001371897]
muF=[1.5, 1.7, 2, 2.2, 2.5, 3, 3.5, 4]
gammaF=np.zeros(8)
muW=[1.5, 1.6, 1.7, 2, 2.5, 3, 3.5, 3.7, 4, 4.5]
gammaW=[1.97740718, 1.96445199, 1.946511022681188,1.7048777028370377,1.417214409530227, 1.1750294111081208, 1.0771141156637074, 1.05344584, 1.025122, 1.0053253]
xw=np.linspace(1.5, 4,10000)
xf=np.linspace(1.5, 4,10000)
gammaw=np.zeros(10000)
gammaf=np.zeros(10000)
for i in range(len(xw)):
    if xw[i]<=2:
        gammaw[i]=2
    if 3>xw[i]>2:
        gammaw[i]=4-xw[i]
    if xw[i]>=3:
        gammaw[i]=1
    if 1.5<= xf[i]<3:
        gammaf[i]=2/(xf[i]-1)
    if xf[i]>3:
        gammaf[i]=1


plt.plot(xf,gammaf,'-',color="black", label= "Pričakovane vrednosti Flight")
plt.plot(xw,gammaw,'-',color="green", label= "Pričakovane vrednosti Walk")
plt.plot(muWZ, gammaWZ,'-',color="blue",label= "Walk z zamikom")
plt.plot(muFZ, gammaFZ,'-',color="red",label= "Flight z zamikom")
plt.title('Odvisnost $gamma$ funkcije od parametra $mu$ za walk in flight ob zaostanku')
plt.xlabel("$\mu$")
plt.ylabel("$\gamma$")
plt.legend()
plt.savefig('Odvisnost $gamma$ funkcije od parametra $mu$ za walk in flight ob zaostanku(1)')
plt.show()
def korak_r_fi(x, y, mi, v, t0):
    epsilon =1e-6
    rho0 = np.random.uniform(0., 1.)
    rho1 = np.random.uniform(0., 1.)
    fi= 2. * np.pi * rho0
    l= (epsilon ** (1. - mi) - rho1 * epsilon ** (1. - mi)) **(1. / (1.- mi))
    rho1 = np.random.uniform(0., 1.)
    return x + l * np.cos(fi), y + l * np.sin(fi), fi, l, t, t0 + t

def polet_r_fi(x,y, mi, t0, tkonst):
    epsilon=10e-6
    rho0 = np.random.uniform(0., 1.)
    rho1 = np.random.uniform(0., 1.)
    fi= 2. * np.pi * rho0
    l= (epsilon ** (1. - mi) - rho1 * epsilon ** (1. - mi)) **(1. / (1.- mi))
    rho1 = np.random.uniform(0., 1.)
    t=tkonst
    return x + l * np.cos(fi), y + l * np.sin(fi), fi, l, t, t0 + t

def flight(mikro, stt, stk, tkonst):
    #DEFINIRANJE TOČK, KI BODO KORAKALE
    n_sprehod=np.zeros((stt, stk, 6)) #ki vsebujejo 6 parametrov x, y, phi, l, t, t0
    #Sestava matrik za korakanje toč
    for i in range(stt):  #posamezne točke
        for j in range(stk-1): #korakanje posamezne točke
            n_sprehod[i,j+1]=polet_r_fi(n_sprehod[i, j, 0], n_sprehod[i, j, 1], mikro, n_sprehod[i, j, 5], tkonst)
    #DEFINIRANJE VEKTORJEV VREDNOSTI MA
    c = 1000 #številka časovnih intervalov
    ckonc=0.1 #max velikost posameznega intervala
    td=np.linspace(0., ckonc, c) # delitev časovnih intervalov 
    ti = [td[n] + (td[n + 1] - td[n]) / 2. for n in range(c - 1)]
    #Računanje MAD v času
    count = 0
    MAD = np.zeros(c) #mediane
    tdata = np.zeros(c) #časi
    MADel= 500 #število meritev za izračun MAD, saj nas zanima zgolj večje vrednosti
    for i in range(c-1):
        indikator = np.where((n_sprehod[:,:,5] > td[i]) & (n_sprehod[:,:,5] <=td[i+1]))
        #ker nas zanimajo zgolj časovno dolgi prehodi in dobimo indekse njih
        if len(indikator[0]) >= MADel: #zanimajo nas bolj končni prehodi
            razdalja =(n_sprehod[indikator[0], indikator[1], 0]**2. +n_sprehod[indikator[0], indikator[1], 1] **2.)**0.5
            MAD[count] = median_absolute_deviation(razdalja)
            #ustvarja tabelo MAD za razdalje, ki imajo dolge časovne intervale in končne prehode
            tdata[count]= ti[i] #umeščanje v časovni interval
            count = count + 1
    return MAD, tdata, count

def walking(mikro, v, stt, stk):
    #DEFINIRANJE TOČK, KI BODO KORAKALE
    n_sprehod=np.zeros((stt, stk, 6)) #ki vsebujejo 6 parametrov x, y, phi, l, t, t0
    #Sestava matrik za korakanje toč
    for i in range(stt):  #posamezne točke
        for j in range(stk-1): #korakanje posamezne točke
            n_sprehod[i,j+1]=korak_r_fi(n_sprehod[i, j, 0], n_sprehod[i, j, 1], mikro, 5., n_sprehod[i, j, 5])
    #DEFINIRANJE VEKTORJEV VREDNOSTI MA
    c = 1000 #številka časovnih intervalov
    ckonc=0.01 #max velikost posameznega intervala
    td=np.linspace(0., ckonc, c) # delitev časovnih intervalov
    ti = [td[n] + (td[n + 1] - td[n]) / 2. for n in range(c - 1)]
    #Računanje MAD v času
    count = 0
    MAD = np.zeros(c) #mediane
    tdata = np.zeros(c) #časi
    MADel= 500 #število meritev za izračun MAD, saj nas zanima zgolj večje vrednosti
    for i in range(c-1):
        indikator = np.where((n_sprehod[:,:,5] > td[i]) & (n_sprehod[:,:,5] <=td[i+1]))
        #ker nas zanimajo zgolj časovno dolgi prehodi in dobimo indekse njih
        if len(indikator[0]) >= MADel: #zanimajo nas bolj končni prehodi
            razdalja =(n_sprehod[indikator[0], indikator[1], 0]**2. +n_sprehod[indikator[0], indikator[1], 1] **2.)**0.5
            MAD[count] = median_absolute_deviation(razdalja)
            #ustvarja tabelo MAD za razdalje, ki imajo dolge časovne intervale in končne prehode
            tdata[count]= ti[i] #umeščanje v časovni interval
            count = count + 1
    return MAD, tdata, count
    
#PLOT funkcije
#funkcija za fitanje: len(varianca)*2 = (gamma)*len(čas) + n
#MAD1=walking(2, 1, 1000, 1000)[0]
#tdata1=walking(2, 1, 1000, 1000)[1]
#count1=walking(2, 1, 1000, 1000)[2]

MAD1=flight(1.5, 1000,1000, 0.0001)[0]
tdata1=flight(1.5, 1000,1000, 0.0001)[1]
count1=flight(1.5, 1000,1000, 0.0001)[2]

MAD2=flight(1.7, 1000,1000, 0.0001)[0]
tdata2=flight(1.7, 1000,1000, 0.0001)[1]
count2=flight(1.7, 1000,1000, 0.0001)[2]

MAD3=flight(2, 1000,1000, 0.0001)[0]
tdata3=flight(2, 1000,1000, 0.0001)[1]
count3=flight(2, 1000,1000, 0.0001)[2]

MAD4=flight(2.2, 1000,1000, 0.0001)[0]
tdata4=flight(2.2, 1000,1000, 0.0001)[1]
count4=flight(2.2, 1000,1000, 0.0001)[2]

MAD5=flight(2.5, 1000,1000, 0.0001)[0]
tdata5=flight(2.5, 1000,1000, 0.0001)[1]
count5=flight(2.5, 1000,1000, 0.0001)[2]

MAD6=flight(3, 1000,1000, 0.0001)[0]
tdata6=flight(3, 1000,1000, 0.0001)[1]
count6=flight(3, 1000,1000, 0.0001)[2]

MAD7=flight(3.5, 1000,1000, 0.0001)[0]
tdata7=flight(3.5, 1000,1000, 0.0001)[1]
count7=flight(3.5, 1000,1000, 0.0001)[2]

MAD8=flight(4, 1000,1000, 0.0001)[0]
tdata8=flight(4, 1000,1000, 0.0001)[1]
count8=flight(4, 1000,1000, 0.0001)[2]

def fitfunkcija(xdata, k, n):
    return (k)*xdata + n #saj smo fukcijo logoritmirali

muFZ=[1.5,1.7,2, 2.2, 2.5, 3, 3.5, 4]
gammaFZ=[3.899448407, 2.814994669215, 2.0097504755, 1.705273316177, 1.443856248487304, 1.21158111495, 1.22157248, 1.10978323]
muWZ=[1.5,1.7,2, 2.2, 2.5, 3, 3.5, 4]
gammaWZ=[1.8992587,1.69877786, 1.656984465644, 1,720024822621229, 1.91862263125815, 1.81103999, 1.700008094151, 1.001371897]
muF=[1.5, 1.7, 2, 2.2, 2.5, 3, 3.5, 4]
gammaF=np.zeros(8)
muW=[1.5, 1.6, 1.7, 2, 2.5, 3, 3.5, 3.7, 4, 4.5]
gammaW=[1.97740718, 1.96445199, 1.946511022681188,1.7048777028370377,1.417214409530227, 1.1750294111081208, 1.0771141156637074, 1.05344584, 1.025122, 1.0053253]

par1, cov1 = curve_fit(fitfunkcija, np.log(tdata1[0:count1]), np.log(MAD1[0:count1])*2.)
gammaF[0]=float(par1[0])
par2, cov2 = curve_fit(fitfunkcija, np.log(tdata2[0:count2]), np.log(MAD2[0:count2])*2.)
gammaF[1]=float(par2[0])
par3, cov3 = curve_fit(fitfunkcija, np.log(tdata3[0:count3]), np.log(MAD3[0:count3])*2.)
gammaF[2]=float(par3[0])
par4, cov4 = curve_fit(fitfunkcija, np.log(tdata4[0:count4]), np.log(MAD4[0:count4])*2.)
gammaF[3]=float(par4[0])
par5, cov5 = curve_fit(fitfunkcija, np.log(tdata5[0:count5]), np.log(MAD5[0:count5])*2.)
gammaF[4]=float(par5[0])
par6, cov6 = curve_fit(fitfunkcija, np.log(tdata6[0:count6]), np.log(MAD6[0:count6])*2.)
gammaF[5]=float(par6[0])
par7, cov7 = curve_fit(fitfunkcija, np.log(tdata7[0:count7]), np.log(MAD7[0:count7])*2.)
gammaF[6]=float(par7[0])
par8, cov8 = curve_fit(fitfunkcija, np.log(tdata8[0:count8]), np.log(MAD8[0:count8])*2.)
gammaF[7]=float(par8[0])
print(muF)
print(gammaF)
#par1, cov1 = curve_fit(fitfunkcija, np.log(tdata1[0:count1]), np.log(MAD1[0:count1])*2.)
par2, cov2 = curve_fit(fitfunkcija, np.log(tdata2[0:count2]), np.log(MAD2[0:count2])*2.)
#ker vemo, da je MAD^2 sorazmeren z varianco^2, ta pa je sorazmerna z t
xw=np.linspace(1.5, 4,10000)
xf=np.linspace(1.5, 4,10000)
gammaw=np.zeros(10000)
gammaf=np.zeros(10000)
for i in range(len(xw)):
    if xw[i]<=2:
        gammaw[i]=2
    if 3>xw[i]>2:
        gammaw[i]=4-xw[i]
    if xw[i]>=3:
        gammaw[i]=1
    if 1.5<= xf[i]<3:
        gammaf[i]=2/(xf[i]-1)
    if xf[i]>3:
        gammaf[i]=1

plt.plot(xf,gammaf,'-',color="red", label= "Pričakovane vrednosti")
plt.plot(muF, gammaF,'o',color="blue",)
plt.title('Odvisnost $\gamma$ funkcije od parametra $\mu$ za flight')
plt.xlabel("$\mu$")
plt.ylabel("$\gamma$")
plt.legend()
plt.savefig('Odvisnost $gamma$ funkcije od parametra $mu$ za flight(1)')
plt.show()

plt.plot(xw,gammaw,'-',color="red", label= "Pričakovane vrednosti")
plt.plot(muW, gammaW,'o',color="blue",label="Dejanske Vrednosti")
plt.title('Odvisnost $\gamma$ funkcije od parametra $\mu$ za walk')
plt.xlabel("$\mu$")
plt.ylabel("$\gamma$")
plt.legend()
plt.savefig('Odvisnost $gamma$ funkcije od parametra $mu$ za walk(1)')
plt.show()


plt.plot(xf,gammaf,'-',color="black", label= "Pričakovane vrednosti Flight")
plt.plot(xw,gammaw,'-',color="green", label= "Pričakovane vrednosti Walk")
plt.plot(muW, gammaW,'-',color="blue",label= "Walk")
plt.plot(muF, gammaF,'-',color="red",label= "Flight")
plt.title('Odvisnost $gamma$ funkcije od parametra $mu$ za walk in flight')
plt.xlabel("$\mu$")
plt.ylabel("$\gamma$")
plt.legend()
plt.savefig('Odvisnost $gamma$ funkcije od parametra $mu$ za walk in flight(1)')
plt.show()

plt.plot(xf,gammaf,'-',color="black", label= "Pričakovane vrednosti Flight")
plt.plot(xw,gammaw,'-',color="green", label= "Pričakovane vrednosti Walk")
plt.plot(muWZ, gammaWZ,'-',color="blue",label= "Walk z zamikom")
plt.plot(muFZ, gammaFZ,'-',color="red",label= "Flight z zamikom")
plt.title('Odvisnost $gamma$ funkcije od parametra $mu$ za walk in flight ob zaostanku')
plt.xlabel("$\mu$")
plt.ylabel("$\gamma$")
plt.legend()
plt.savefig('Odvisnost $gamma$ funkcije od parametra $mu$ za walk in flight ob zaostanku(1)')
plt.show()

