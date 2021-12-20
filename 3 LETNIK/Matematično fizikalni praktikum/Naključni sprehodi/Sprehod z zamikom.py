import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation
from scipy.optimize import curve_fit

def korak_r_fi(x, y, mi, v, t0):
    epsilon =1e-6
    rho0 = np.random.uniform(0., 1.)
    rho1 = np.random.uniform(0., 1.)
    fi= 2. * np.pi * rho0
    l= (epsilon ** (1. - mi) - rho1 * epsilon ** (1. - mi)) **(1. / (1.- mi))
    rho1 = np.random.uniform(0., 1.)
    tc= (epsilon ** (1. - mi) - rho1 * epsilon ** (1. - mi)) **(1. / (1.- mi))
    t = 2 / v+tc
    return x + l * np.cos(fi), y + l * np.sin(fi), fi, l, t, t0 + t
def polet_r_fi(x,y, mi, t0, tkonst):
    epsilon=10e-6
    rho0 = np.random.uniform(0., 1.)
    rho1 = np.random.uniform(0., 1.)
    fi= 2. * np.pi * rho0
    l= (epsilon ** (1. - mi) - rho1 * epsilon ** (1. - mi)) **(1. / (1.- mi))
    rho1 = np.random.uniform(0., 1.)
    tc= (epsilon ** (1. - mi) - rho1 * epsilon ** (1. - mi)) **(1. / (1.- mi))
    t=tkonst + tc
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
    ckonc=0.01 #max velikost posameznega intervala
    td=np.linspace(0., ckonc, c) # delitev časovnih intervalov  #stevilo meritev za izracun MAD
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

MAD1=flight(1.5, 1000,100, 0.0001)[0]
tdata1=flight(1.5, 1000,100, 0.0001)[1]
count1=flight(1.5, 1000,100, 0.0001)[2]

MAD2=flight(1.7, 1000,100, 0.0001)[0]
tdata2=flight(1.7, 1000,100, 0.0001)[1]
count2=flight(1.7, 1000,100, 0.0001)[2]

MAD3=flight(2, 1000,100, 0.0001)[0]
tdata3=flight(2, 1000,100, 0.0001)[1]
count3=flight(2, 1000,100, 0.0001)[2]

MAD4=flight(2.2, 1000,100, 0.0001)[0]
tdata4=flight(2.2, 1000,100, 0.0001)[1]
count4=flight(2.2, 1000,100, 0.0001)[2]

MAD5=flight(2.5, 1000,100, 0.0001)[0]
tdata5=flight(2.5, 1000,100, 0.0001)[1]
count5=flight(2.5, 1000,100, 0.0001)[2]

MAD6=flight(3, 1000,100, 0.0001)[0]
tdata6=flight(3, 1000,100, 0.0001)[1]
count6=flight(3, 1000,100, 0.0001)[2]

MAD7=flight(3.5, 1000,100, 0.0001)[0]
tdata7=flight(3.5, 1000,100, 0.0001)[1]
count7=flight(3.5, 1000,100, 0.0001)[2]

MAD8=flight(4, 1000,100, 0.0001)[0]
tdata8=flight(4, 1000,100, 0.0001)[1]
count8=flight(4, 1000,100, 0.0001)[2]

def fitfunkcija(xdata, k, n):
    return (k)*xdata + n #saj smo fukcijo logoritmirali
gamma=[]
par1, cov1 = curve_fit(fitfunkcija, np.log(tdata1[0:count1]), np.log(MAD1[0:count1])*2.)
gamma[0]=float(par1[0])
par2, cov2 = curve_fit(fitfunkcija, np.log(tdata2[0:count2]), np.log(MAD2[0:count2])*2.)
gamma[1]=float(par2[0])
par3, cov3 = curve_fit(fitfunkcija, np.log(tdata3[0:count3]), np.log(MAD3[0:count3])*2.)
gamma[2]=float(par3[0])
par4, cov4 = curve_fit(fitfunkcija, np.log(tdata4[0:count4]), np.log(MAD4[0:count4])*2.)
gamma[3]=float(par4[0])
par5, cov5 = curve_fit(fitfunkcija, np.log(tdata5[0:count5]), np.log(MAD5[0:count5])*2.)
gamma[4]=float(par5[0])
par6, cov6 = curve_fit(fitfunkcija, np.log(tdata6[0:count6]), np.log(MAD6[0:count6])*2.)
gamma[5]=float(par6[0])
par7, cov7 = curve_fit(fitfunkcija, np.log(tdata7[0:count7]), np.log(MAD7[0:count7])*2.)
gamma[6]=float(par7[0])
par8, cov8 = curve_fit(fitfunkcija, np.log(tdata8[0:count8]), np.log(MAD8[0:count8])*2.)
gamma[7]=float(par8[0])


#par1, cov1 = curve_fit(fitfunkcija, np.log(tdata1[0:count1]), np.log(MAD1[0:count1])*2.)

#ker vemo, da je MAD^2 sorazmeren z varianco^2, ta pa je sorazmerna z t

#np.amax vrne največjo vrednost časovnega intervala
#np.amin pa najmanjšo vrednost časovnega intervala
#r1 = np.linspace(np.amax(np.log(tdata1[0:count1])), np.amin(np.log(tdata1[0:count1])))
#plt.plot(np.log(tdata1[0:count1]), np.log(MAD1[0:count1] ** 2),'o',color="black",)
#plt.plot(r1, fitfunkcija(r, par1[0], par1[1]),'-',color='red', label=str(par1[0]))

r1 = np.linspace(np.amax(np.log(tdata1[0:count1])), np.amin(np.log(tdata1[0:count1])))
plt.plot(np.log(tdata1[0:count1]), np.log(MAD1[0:count1] ** 2),'o',color="green",)
plt.plot(r2, fitfunkcija(r1, par1[0], par1[1]),'-',color='blue', label=str(par1[0]))
plt.title('Flight z zamikom pri paramterih $\mu=1.5$, N=1000')
plt.xlabel("ln[t]")
plt.ylabel("2ln[MAD]")
plt.legend()
plt.savefig('Flight z zamikom pri paramterih $mu=1.5$, N=1000 K=1000.png')
plt.show()

r2 = np.linspace(np.amax(np.log(tdata2[0:count2])), np.amin(np.log(tdata2[0:count2])))
plt.plot(np.log(tdata2[0:count2]), np.log(MAD2[0:count2] ** 2),'o',color="green",)
plt.plot(r2, fitfunkcija(r2, par2[0], par2[1]),'-',color='blue', label=str(par2[0]))
plt.title('Flight z zamikom pri paramterih $mu=1.7$, N=1000')
plt.xlabel("ln[t]")
plt.ylabel("2ln[MAD]")
plt.legend()
plt.savefig('Flight z zamikom pri paramterih $mu=1.7$, N=1000 K=1000.png')
plt.show()
