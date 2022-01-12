import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation
from scipy.optimize import curve_fit
#Funkcija za generacijo naslednjega koraka
#Funkcija sprejme star korak (x, y, t0) in mikro ter konst. hitrosti
#Funkcija vrne nov korak (x, y, phi, l, t1, t01, t2, t02, t3, t03, t4, t04)
#t je čas koraka, t0 je celotni čas, ki smo da porabili do tega koraka
def korak(x, y, t01, t02, t03, t04, mikro, nu, vkonst, tkonst):
    epsilon = 1e-6

    ro = np.random.uniform(0., 1.)
    phi = 2. * np.pi * ro
    ro = np.random.uniform(0., 1.)
    #3 razlicne funkcije iz Inverse Transform Samplinga
    #l = (ro * (mikro - 1.)) ** (1. / (1. - mikro))
    #l = (ro * (mikro - 1.) + epsilon ** (1. - mikro)) ** (1. / (1. - mikro))
    l = (ro * epsilon ** (1. - mikro)) ** (1. / (1. - mikro))
    
    ro = np.random.uniform(0., 1.)
    tc = (ro * epsilon ** (1. - nu)) ** (1. / (1. - nu))
    t1 = l/vkonst
    t2 = tkonst
    t3 = t1 + tc
    t4 = t2 + tc
    return x + l * np.cos(phi), y + l * np.sin(phi), phi, l, t1, t01+t1, t2, t02+t2, t3, t03+t3, t4, t04+t4

#funkcija za fitanje: len(varianca)*2 = (gamma)*len(čas) + n

def fitfunkcija(xdata, k, n):
    return (k)*xdata + n



def izdelava(mikro, nu, vkonst, tkonst,s, k, c,tfin):
#zgeneriram vse korake za vse sprehode
#n_sprehod[i, j] vsebuje x, y, phi, l, t1, t01, t2, t02, t3, t03, t4, t04
    n_sprehod = np.zeros((s, k, 12))
    for i in range(s):
        for j in range(k - 1):
            n_sprehod[i, j + 1] = korak(n_sprehod[i, j, 0], n_sprehod[i, j, 1], n_sprehod[i, j, 5], n_sprehod[i, j, 7], n_sprehod[i, j, 9], n_sprehod[i, j, 11], mikro, nu, vkonst, tkonst)

#definiramo potrebne vektorje
    tt = np.linspace(0., tfin, c) #vektor časovnih intervalov
    ti = [tt[n] + (tt[n + 1] - tt[n]) / 2. for n in range(c - 1)] #vektor polovičk med intervali
#mediane in časi
    MAD1 = np.zeros(c)
    tdata1 = np.zeros(c)
    MAD2 = np.zeros(c)
    tdata2 = np.zeros(c)
    MAD3 = np.zeros(c)
    tdata3 = np.zeros(c)
    MAD4 = np.zeros(c)
    tdata4 = np.zeros(c) 


#racunanje MAD v časovnih intervalih
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    MADel=500
    MADe2=500
    for i in range(c - 1):
        #returns indices
        indices1 = np.where((n_sprehod[:, :, 5] > tt[i]) & (n_sprehod[:, :, 5] <= tt[i + 1])) 
        indices2 = np.where((n_sprehod[:, :, 7] > tt[i]) & (n_sprehod[:, :, 7] <= tt[i + 1])) 

    #upoštevamo samo MAD izračunan iz večih vrednosti
        if len(indices1[0]) >= MADel: 
            razdalja = (n_sprehod[indices1[0], indices1[1], 0] ** 2. + n_sprehod[indices1[0], indices1[1], 1]**2.)**0.5
            MAD1[count1] = median_absolute_deviation(razdalja)
            tdata1[count1] = ti[i]
            count1 = count1 + 1
        if len(indices2[0]) >= MADe2: 
            razdalja = (n_sprehod[indices2[0], indices2[1], 0] ** 2. + n_sprehod[indices2[0], indices2[1], 1]**2.)**0.5
            MAD2[count2] = median_absolute_deviation(razdalja)
            tdata2[count2] = ti[i]
            count2 = count2 + 1
    par1, cov1 = curve_fit(fitfunkcija, np.log(tdata1[0:count1]), np.log(MAD1[0:count1])*2.)
    par_1=par1[0]
    par2, cov2 = curve_fit(fitfunkcija, np.log(tdata2[0:count2]), np.log(MAD2[0:count2])*2.)
    par_2=par2[0]
    return  par_1, par_2 

#fitanje
nu=np.linspace(1.2,4,1)
mu1=np.linspace(1.5,1.9,10)
mu=[]
gammaF=[]
gammaW=[]
for i in range(len(mu1)):
    print(i)
    mu.append(mu1[i])
    gammaF.append(0)
    gammaW.append(0)
    gammaF[i]=(izdelava(mu1[i], 0, 1, 0.0001 ,1000, 1000, 1000,0.01)[1])
    gammaW[i]=(izdelava(mu1[i], 0, 1, 0.0001 ,1000, 1000, 1000,0.01)[0])
    
mu2=np.linspace(2.1,2.9,10)
for i in range(len(mu2)):
    print(i)
    mu.append(mu2[i])
    gammaF.append(0)
    gammaW.append(0)
    gammaF[i+10]=(izdelava(mu2[i], 0, 1, 0.0001 ,1000, 1000, 1000,0.01)[1])
    gammaW[i+10]=(izdelava(mu2[i], 0, 1, 0.0001 ,1000, 1000, 1000,0.01)[0])

mu3=np.linspace(3.1,4.5,10)
for i in range(len(mu3)):
    print(i)
    mu.append(mu3[i])
    gammaF.append(0)
    gammaW.append(0)
    gammaF[i+20]=(izdelava(mu3[i], 0, 1, 0.0001 ,1000, 1000, 1000,0.01)[1])
    gammaW[i+20]=(izdelava(mu3[i], 0, 1, 0.0001 ,1000, 1000, 1000,0.01)[0])

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

plt.plot(xf,gammaf,'-',color="black", label= "Pričakovane vrednosti")
plt.plot(mu,gammaF,'o',color="red", label= "Vrednosti Flight")
plt.title('Odvisnost $gamma$ funkcije od parametra $mu$ flight')
plt.xlabel("$\mu$")
plt.ylabel("$\gamma$")
plt.legend()
plt.savefig('Odvisnost $gamma$ funkcije od parametra $mu$ za walk in flight ob zaostanku(2)')
plt.show()
plt.plot(xw,gammaw,'-',color="black", label= "Pričakovane vrednosti")
plt.plot(mu,gammaW,'o',color="red", label= "Vrednosti Walk")
plt.title('Odvisnost $gamma$ funkcije od parametra $mu$ walk')
plt.xlabel("$\mu$")
plt.ylabel("$\gamma$")
plt.legend()
plt.savefig('Odvisnost $gamma$ funkcije od parametra $mu$ za walk in flight ob zaostanku(3)')
plt.show()
#plotanje











