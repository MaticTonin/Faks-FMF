import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.linalg as la
from scipy import fftpack
from scipy.fftpack import fft, fftfreq
import spectrum as spc
from spectrum import *
from scipy import optimize as opt


cm = plt.get_cmap('rainbow')
v1 = 100
v2 = 150
w1 = v1*2*np.pi
w2 = v2*2*np.pi

sr = 1000
x = np.linspace(0,1,sr)
val2 = np.zeros(len(x))
for i in range(len(x)):
    val2[i] = np.sin(x[i]*w1) + np.sin(x[i]*w2)

plt.subplot(2,1,1)
plt.title("Ločljivosti metode.")
plt.plot(x,val2, label = r"$\sin(t*{%.2f})+\sin(t*{%.2f})$" %(w1,w2))
plt.xlabel("t")
plt.ylabel("Amp")
plt.grid()
plt.legend()




N = len(val2)
val_fft = fft(val2)
x_fft = fftfreq(N, 1/N)[:N//2]

bar = 0
pl = [64,65,66,70]
plt.subplot(2,2,3)
plt.plot(x_fft, np.abs(val_fft[0:N//2])**2, label = 'FT',color = cm(1.*bar/(len(pl)+1)))


for p in pl:
    bar = bar+1
    a, rho, k = spc.aryule(val2, p)
    PSD = arma2psd(a, NFFT=N)
    plt.plot(x_fft, PSD[:N//2]**2, label = 'p = '+str(p), color = cm(1.*bar/(len(pl)+1)))

    vrh, pr = sig.find_peaks(PSD[:N//2]**2,width = 1, height = 10**4)
    #vrh, pr = sig.find_peaks(np.abs(val_fft[0:N//2])**2,width = 1, height = 10**4)
    
plt.yscale('log')
plt.ylabel(r'PSD') 
plt.xlabel(r'$\nu$')
plt.legend()
plt.grid()





sr = 1000
x = np.linspace(0,1,sr)
val2 = np.zeros(len(x))

plt.subplot(2,2,4)
for v2 in range(101, 160):
    w2 = v2*2*np.pi
    for i in range(len(x)):
        val2[i] = np.sin(x[i]*w1) + np.sin(x[i]*w2)

    p = 2
    while 1 == 1:
        a, rho, k = spc.aryule(val2, p)
        PSD = arma2psd(a, NFFT=N)
        vrh, pr = sig.find_peaks(PSD[:N//2]**2,width = 1, height = 10**4)
        if p > 50:
            print('limit!')
            break
        if len(vrh) < 2:
            p = p + 1
        else:
            if len(vrh) > 2:
                plt.scatter(v2-v1, p, color = 'r', marker = '.')
                print('wtf')
            else:
                plt.scatter(v2-v1, p, color = 'k', marker = '.')
        
            break

plt.xlabel(r'$\Delta \nu$')
plt.ylabel('p')
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.linalg as la
import spectrum as spc
from spectrum import *

#število sample točko
#N = 512 deli 1
#N = 256 deli 2
#N = 128 deli 4
#N = 64  deli 8
deli = 1
N = int(512/deli)

p = 3

cm = plt.get_cmap('nipy_spectral')
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data = THIS_FOLDER+r"\val2.dat"
#branje datoteke
f=open(data ,"r")

lines=f.readlines()
val2=np.array([])

#naredimo array iz N meritev
t=0

for y in lines:
    if t >= N:
        break
    val2 = np.append(val2, float(y.split('\n')[0]))
    t = t + 1  
f.close()

#sample spacing
x = np.linspace(0,len(val2), len(val2))
pji = 100
ps = np.linspace(2,pji,pji-1)
#[2,3,4,5,6,7,8,9,10,11,12,13]
GS = np.zeros(len(ps))

st = -1
for p in ps:
    p = int(p)
    st += 1
    stolpec = np.zeros(p)
    R = np.zeros(p)
    for i in range(p):
        rola = np.roll(val2, i+1)
        rola2 = np.roll(val2, i)
        for j in range(len(val2)):
            R[i] = R[i] + val2[j] * rola[j]
            stolpec[i] = stolpec[i] + val2[j]*rola2[j]

    RM = la.toeplitz(stolpec)

    a, rho, k = spc.aryule(val2, p)

    G2 = RM[0,0]
    for i in range(len(a)):
        G2 += a[i]*R[i]
    GS[st] = G2

plt.stem(ps, GS, markerfmt = 'or', linefmt='--k')
plt.axhline(GS[-1]*1.05, color = 'b', ls='--')
plt.axhline(GS[-1], color = 'b')
plt.axhline(GS[-1]*0.95, color = 'b', ls='--')
plt.fill_between(ps, GS[-1]*0.95, GS[-1]*1.05, color = 'b', alpha = 0.5)
plt.ylabel(r'$G^2$')
plt.xlabel('p')
plt.grid()
plt.show()





from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.linalg as la
from scipy import fftpack
from scipy.fftpack import fft, fftfreq
import spectrum as spc

#število sample točko
#N = 512 deli 1
#N = 256 deli 2
#N = 128 deli 4
#N = 64  deli 8
deli = 1
N = int(512/deli)

p = 3
t = 0
import os
from tqdm import tqdm
import numpy.linalg as lin
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data = THIS_FOLDER+r"\val2.dat"
f=open(data ,"r")
val2=np.array([])
lines=f.readlines()
t=0
for y in lines:
    if t >= N:
        break
    val2 = np.append(val2, float(y.split('\n')[0]))
    t = t + 1  
f.close()

data = THIS_FOLDER+r"\val3.dat"
f=open(data ,"r")
val3=np.array([])
lines=f.readlines()
t=0 
for y in lines:
    if t >= N:
        break
    val3 = np.append(val3, float(y.split('\n')[0]))
    t = t + 1  
f.close()

#sample spacing
x = np.linspace(0,len(val2), len(val2))

val_fft = fft(val2)
x_fft = fftfreq(N, 1/512)[:N//2]
plt.subplot(2,2,1)
plt.title("Prikaz spektra val2")
plt.plot(x_fft, np.abs(val_fft[0:256])**2, label = 'FT')
p=4
for p in[4,8,16, 32, 64]:
    a, rho, k = spc.aryule(val2, p)
    PSD = arma2psd(a, NFFT=512)
    plt.plot(x_fft, PSD[:N//2]**2, label = 'p = '+str(p))
plt.grid()
plt.yscale('log')
plt.ylabel(r'PSD') 
plt.xlabel('frekvenca')
plt.legend()
x = np.linspace(0,len(val3), len(val3))

val_fft = fft(val3)
x_fft = fftfreq(N, 1/512)[:N//2]

plt.subplot(2,2,2)
plt.title("Prikaz spektra val3")
plt.plot(x_fft, np.abs(val_fft[0:256])**2, label = 'FT')
p=4
for p in[4,8,16,32,64]:
    a, rho, k = spc.aryule(val3, p)
    PSD = arma2psd(a, NFFT=512)
    plt.plot(x_fft, PSD[:N//2]**2, label = 'p = '+str(p))
plt.yscale('log')
plt.ylabel(r'PSD') 
plt.xlabel('frekvenca')
plt.legend()
plt.grid()

plt.subplot(2,2,3)

for p in [4,8,16,32,64]:
    a, rho, k = spc.aryule(val2, p)
    r = np.roots([1, *a])
    plt.scatter([np.real(i) for i in r], [np.imag(i) for i in r], label="p="+str(p), marker=".")

fi = np.linspace(0, 2*np.pi, 300)
plt.plot(np.cos(fi), np.sin(fi))
plt.xlabel("Re")
plt.ylabel("Im")
plt.legend()
plt.grid()

plt.subplot(2,2,4)

for p in [4,8,16,32, 64]:
    a, rho, k = spc.aryule(val3, p)
    r = np.roots([1, *a])
    plt.scatter([np.real(i) for i in r], [np.imag(i) for i in r], label="p="+str(p), marker=".")

fi = np.linspace(0, 2*np.pi, 300)
plt.plot(np.cos(fi), np.sin(fi))
plt.xlabel("Re")
plt.ylabel("Im")
plt.legend()
plt.grid()

plt.show()



import os
from tqdm import tqdm
import numpy.linalg as lin
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data = THIS_FOLDER+r"\co2.dat"
f=open(data ,"r")
lines=f.readlines()
val2=np.array([])
x=np.array([])


for y in lines:
    val2 = np.append(val2, float(y.split('\n')[0].split(' ')[1]))
    x = np.append(x, float(y.split('\n')[0].split(' ')[0]))
f.close()

for i in range(len(val2)):
    if val2[i] == -99.99:
        n = 1
        while val2[i-n] == -99.9:
            n = n +1 
        val2[i] = val2[i-n]



def kf(x, a, b, c ):
    return a*x**2 + b*x + c

def lf(x, b, c ):
    return  b*x + c

def ef(x, a, b, c,d ):
    return a*np.exp(b*(x+d)) + c


par, cov = opt.curve_fit(kf, x, val2)
par2, cov = opt.curve_fit(lf, x, val2)
par3, cov = opt.curve_fit(ef, x, val2, p0=[0.1,0.1,320,-1960])

plt.subplot(2,1,1)
plt.title("Prikaz podatkov CO2 in fita funkcije")
plt.plot(x, val2)
plt.plot(x,kf(x,*par), label = '$%.2f x^2 + %.2f x+ %.2f$' %(par[0], par[1], par[2]))
plt.plot(x,lf(x,*par2), label = '$ %.2f x+ %.2f$' %(par2[0], par2[1]))

plt.grid()
plt.legend()
plt.xlabel('t')
plt.ylabel('Koncentracija')


for i in range(len(val2)):
    val2[i] = val2[i] - lf(x[i],*par2)
plt.subplot(2,1,2)
plt.title("Odštevanje lineranega odziva")
plt.plot(x, val2)
plt.grid()
plt.legend()
plt.xlabel('t')
plt.ylabel('Koncentracija')
plt.show()



N = len(val2)
val_fft = fft(val2)
x_fft = fftfreq(N, 1/N)[:N//2]

bar = 0
pl = [64,128, 256, 512]
plt.subplot(2,1,1)
plt.title("Prikaz frekvenčnega spektra CO2")
plt.plot(x_fft, np.abs(val_fft[0:N//2])**2, label = 'FT',color = cm(1.*bar/(len(pl)+1)))

p=4

for p in pl:
    bar = bar+1
    a, rho, k = spc.aryule(val2, p)
    PSD = arma2psd(a, NFFT=N)
    plt.plot(x_fft, PSD[:N//2]**2, label = 'p = '+str(p), color = cm(1.*bar/(len(pl)+1)))

plt.yscale('log')
plt.ylabel(r'PSD') 
plt.xlabel(r'$\nu$')
plt.legend()
plt.grid()


plt.subplot(2,1,2)

#[2,3,4,5,6,7,8,9,10,11,12,13]
bar = -1
for p in pl:
    bar = bar + 1
    a, rho, k = spc.aryule(val2, p)

    r = np.roots([1, *a])

    plt.scatter([np.real(i) for i in r], [np.imag(i) for i in r], label="p = "+str(p) , color = cm(1.*bar/len(pl)))

print(a)

fi = np.linspace(0, 2*np.pi, 300)
plt.plot(np.cos(fi), np.sin(fi), color = 'k')
plt.legend()
plt.grid()
plt.show()


val2 = val2[100:200]
N = len(val2)
val_fft = fft(val2)
x_fft = fftfreq(N, 1/N)[:N//2]
x = np.linspace(val2[0], val2[len(val2)-1], len(val2))
bar = 0
pl = [4,8,16,32]
plt.subplot(2,2,2)
plt.title("Prikaz frekvenčnega spektra CO2")
plt.plot(x_fft, np.abs(val_fft[0:N//2])**2, label = 'FT',color = cm(1.*bar/(len(pl)+1)))


plt.subplot(2,2,1)
plt.title("Prikaz obrezanega spektra CO2")
plt.plot(x,val2)
plt.xlabel("t")
plt.ylabel("Amp")
plt.grid()

p=4

for p in pl:
    bar = bar+1
    a, rho, k = spc.aryule(val2, p)
    PSD = arma2psd(a, NFFT=N)
    plt.plot(x_fft, PSD[:N//2]**2, label = 'p = '+str(p), color = cm(1.*bar/(len(pl)+1)))

plt.yscale('log')
plt.ylabel(r'PSD') 
plt.xlabel(r'$\nu$')
plt.legend()
plt.grid()


plt.subplot(2,1,2)

#[2,3,4,5,6,7,8,9,10,11,12,13]
bar = -1
for p in pl:
    bar = bar + 1
    a, rho, k = spc.aryule(val2, p)

    r = np.roots([1, *a])

    plt.scatter([np.real(i) for i in r], [np.imag(i) for i in r], label="p = "+str(p) , color = cm(1.*bar/len(pl)))

print(a)

fi = np.linspace(0, 2*np.pi, 300)
plt.plot(np.cos(fi), np.sin(fi), color = 'k')
plt.legend()
plt.grid()
plt.show()
#[2,3,4,5,6,7,8,9,10,11,12,13]
#val_fft = fft(val2)
#x_fft = fftfreq(N, 1/512)[:N//2]
#
#plt.plot(x_fft, np.abs(val_fft[0:256])**2, label = 'FT')
#
#p=4
#for p in[4,8,16]:
#    a, rho, k = spc.aryule(val2, p)
#    PSD = arma2psd(a, NFFT=512)
#    plt.plot(x_fft, PSD[:N//2], label = 'p = '+str(p))
#
#
#plt.yscale('log')
#plt.ylabel(r'PSD') 
#plt.xlabel('frekvenca')
#plt.legend()
#plt.grid()
#plt.show()