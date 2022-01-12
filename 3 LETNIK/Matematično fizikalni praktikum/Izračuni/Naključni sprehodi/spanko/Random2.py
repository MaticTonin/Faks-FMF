import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation
from scipy.optimize import curve_fit



#Funkcija za generacijo naslednjega koraka
#Funkcija sprejme star korak (x, y, t0) in mikro ter konst. hitrosti
#Funkcija vrne nov korak (x, y, phi, l, t, t0)
#t je čas koraka, t0 je celotni čas, ki smo da porabili do tega koraka
def korak(x, y, t0, mikro, v):
    epsilon = 1e-6

    ro = np.random.uniform(0., 1.)
    phi = 2. * np.pi * ro
    ro = np.random.uniform(0., 1.)

    #3 razlicne funkcije iz Inverse Transform Samplinga
    #l = (ro * (mikro - 1.)) ** (1. / (1. - mikro))
    #l = (ro * (mikro - 1.) + epsilon ** (1. - mikro)) ** (1. / (1. - mikro))
    l = (ro * epsilon ** (1. - mikro)) ** (1. / (1. - mikro))
    t = l/v
    return x + l * np.cos(phi), y + l * np.sin(phi), phi, l, t, t0 + t

#funkcija za fitanje: len(varianca)*2 = (gamma)*len(čas) + n
def fitfunkcija(xdata, k, n):
    return (k)*xdata + n



#definirane konstante
mikro = 2
v = 5.
s = 10000 #stevilo sprehodov
k = 100 #stevilo korakov
c = 1000 #stevilo casovnih intervalov c - 1
#parametri za optimalnost fitanja
cfin = 0.01 #časovno območje intervalov 
MADel = 500 #stevilo meritev za izracun MAD

#zgeneriram vse korake za vse sprehode
#n_sprehod[i, j] vsebuje x, y, phi, l, t, t0
n_sprehod = np.zeros((s, k, 6))
for i in range(s):
    for j in range(k - 1):
        n_sprehod[i, j + 1] = korak(n_sprehod[i, j, 0], n_sprehod[i, j, 1], n_sprehod[i, j, 5], mikro, v)

#definiramo potrebne vektorje
tt = np.linspace(0., cfin, c) #vektor časovnih intervalov
ti = [tt[n] + (tt[n + 1] - tt[n]) / 2. for n in range(c - 1)] #vektor polovičk med intervali
MAD = np.zeros(c) #mediane
tdata = np.zeros(c) #časi


#racunanje MAD v časovnih intervalih
count1 = 0
for i in range(c - 1):
    indices = np.where((n_sprehod[:, :, 5] > tt[i]) & (n_sprehod[:, :, 5] <= tt[i + 1])) #returns indices

    if len(indices[0]) >= MADel: #upoštevamo samo MAD izračunan iz večih vrednosti
        razdalja = (n_sprehod[indices[0], indices[1], 0] ** 2. + n_sprehod[indices[0], indices[1], 1]**2.)**0.5
        MAD[count1] = median_absolute_deviation(razdalja)
        tdata[count1] = ti[i]
        count1 = count1 + 1


#fitanje
par, cov = curve_fit(fitfunkcija, np.log(tdata[0:count1]), np.log(MAD[0:count1])*2.)
print (par)
print (cov)

#plotanje
r = np.linspace(np.amax(np.log(tdata[0:count1])), np.amin(np.log(tdata[0:count1])))
plt.plot(np.log(tdata[0:count1]), np.log(MAD[0:count1] ** 2), 'o')
plt.plot(r, fitfunkcija(r, par[0], par[1]), '-')

plt.show()

