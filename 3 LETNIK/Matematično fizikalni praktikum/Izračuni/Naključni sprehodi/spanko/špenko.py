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



#definirane konstante
mikro = 1.5
nu = 1.5
vkonst = 5.
tkonst = 0.0001 #100 korakov v 0.01
s = 10000 #stevilo sprehodov
k = 100 #stevilo korakov
c = 1000 #stevilo casovnih intervalov c - 1
#parametri za optimalnost fitanja
tfin = 0.01 #časovno območje intervalov 
MADel = 500 #stevilo meritev za izracun MAD

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
for i in range(c - 1):
    #returns indices
    indices1 = np.where((n_sprehod[:, :, 5] > tt[i]) & (n_sprehod[:, :, 5] <= tt[i + 1])) 
    indices2 = np.where((n_sprehod[:, :, 7] > tt[i]) & (n_sprehod[:, :, 7] <= tt[i + 1])) 
    indices3 = np.where((n_sprehod[:, :, 9] > tt[i]) & (n_sprehod[:, :, 9] <= tt[i + 1])) 
    indices4 = np.where((n_sprehod[:, :, 11] > tt[i]) & (n_sprehod[:, :, 11] <= tt[i + 1])) 

    #upoštevamo samo MAD izračunan iz večih vrednosti
    if len(indices1[0]) >= MADel: 
        razdalja = (n_sprehod[indices1[0], indices1[1], 0] ** 2. + n_sprehod[indices1[0], indices1[1], 1]**2.)**0.5
        MAD1[count1] = median_absolute_deviation(razdalja)
        tdata1[count1] = ti[i]
        count1 = count1 + 1
    if len(indices2[0]) >= MADel: 
        razdalja = (n_sprehod[indices2[0], indices2[1], 0] ** 2. + n_sprehod[indices2[0], indices2[1], 1]**2.)**0.5
        MAD2[count2] = median_absolute_deviation(razdalja)
        tdata2[count2] = ti[i]
        count2 = count2 + 1
    if len(indices3[0]) >= MADel: 
        razdalja = (n_sprehod[indices3[0], indices3[1], 0] ** 2. + n_sprehod[indices3[0], indices3[1], 1]**2.)**0.5
        MAD3[count3] = median_absolute_deviation(razdalja)
        tdata3[count3] = ti[i]
        count3 = count3 + 1
    if len(indices4[0]) >= MADel: 
        razdalja = (n_sprehod[indices4[0], indices4[1], 0] ** 2. + n_sprehod[indices4[0], indices4[1], 1]**2.)**0.5
        MAD4[count4] = median_absolute_deviation(razdalja)
        tdata4[count4] = ti[i]
        count4 = count4 + 1


#fitanje
par1, cov1 = curve_fit(fitfunkcija, np.log(tdata1[0:count1]), np.log(MAD1[0:count1])*2.)
print (par1)
print (cov1)
par2, cov2 = curve_fit(fitfunkcija, np.log(tdata2[0:count2]), np.log(MAD2[0:count2])*2.)
print (par2)
print (cov2)
par3, cov3 = curve_fit(fitfunkcija, np.log(tdata3[0:count3]), np.log(MAD3[0:count3])*2.)
print (par3)
print (cov3)
par4, cov4 = curve_fit(fitfunkcija, np.log(tdata4[0:count4]), np.log(MAD4[0:count4])*2.)
print (par4)
print (cov4)

#plotanje
plt.subplot(2, 2, 1)
r1 = np.linspace(np.amax(np.log(tdata1[0:count1])), np.amin(np.log(tdata1[0:count1])))
plt.plot(np.log(tdata1[0:count1]), np.log(MAD1[0:count1])*2., 'o')
line1, = plt.plot(r1, fitfunkcija(r1, par1[0], par1[1]), '-', label=str(par1[0]))
plt.xlabel("ln[t]")
plt.ylabel("2ln[MAD]")
plt.grid(True)
plt.legend(handles=[line1])
plt.title("Walk " + str(mikro) + " mikro")

plt.subplot(2, 2, 2)
r2 = np.linspace(np.amax(np.log(tdata2[0:count2])), np.amin(np.log(tdata2[0:count2])))
plt.plot(np.log(tdata2[0:count2]), np.log(MAD2[0:count2])*2., 'o')
line2, =plt.plot(r2, fitfunkcija(r2, par2[0], par2[1]), '-', label=str(par2[0]))
plt.xlabel("ln[t]")
plt.ylabel("2ln[MAD]")
plt.grid(True)
plt.legend(handles=[line2])
plt.title("Flight " + str(mikro) + " mikro")

plt.subplot(2, 2, 3)
r3 = np.linspace(np.amax(np.log(tdata3[0:count3])), np.amin(np.log(tdata3[0:count3])))
plt.plot(np.log(tdata3[0:count3]), np.log(MAD3[0:count3])*2., 'o')
line3, =plt.plot(r3, fitfunkcija(r3, par3[0], par3[1]), '-', label=str(par3[0]))
plt.xlabel("ln[t]")
plt.ylabel("2ln[MAD]")
plt.grid(True)
plt.legend(handles=[line3])
plt.title("Walk with trapping time " + str(mikro) + " mikro")

plt.subplot(2, 2, 4)
r4 = np.linspace(np.amax(np.log(tdata4[0:count4])), np.amin(np.log(tdata4[0:count4])))
plt.plot(np.log(tdata4[0:count4]), np.log(MAD4[0:count4])*2., 'o')
line4, =plt.plot(r4, fitfunkcija(r4, par4[0], par4[1]), '-', label=str(par4[0]))
plt.xlabel("ln[t]")
plt.ylabel("2ln[MAD]")
plt.grid(True)
plt.legend(handles=[line4])
plt.title("Flight with trapping time " + str(mikro) + " mikro")

plt.tight_layout()
plt.show()



















