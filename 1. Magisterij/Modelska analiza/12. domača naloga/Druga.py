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
#branje datoteke
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
file_name = THIS_FOLDER+r"\luna.dat"
f=open(file_name ,"r")
lines=f.readlines()
val2=np.array([])
x=np.array([])
#0 datum
#1 rekt
#2 dekl
t = -1 
for y in lines:
    t = t +1
    val2 = np.append(val2, float(y.split('\n')[0].split(' ')[1]))
    x = np.append(x, 1995+t/365)
f.close()
#x, y, val2 = np.genfromtxt(file_name).T
x = np.linspace(0,len(val2), len(val2))
plt.figure(1)
plt.title("Prikaz podatkov rektascenzije lune")
plt.xlabel('Čas')
plt.ylabel('Borza')
plt.plot(x,val2)
plt.grid()


pji = 10
ps = np.linspace(2,pji,(pji)//10)
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



N = len(val2)
val_fft = fft(val2)
x_fft = fftfreq(N, 1/N)[:N//2]

bar = 0
pl = [10,20,50,100,200]

p=200

val3 = np.copy(val2)
for i in range(N//2, N):
    val3[i] = 0
plt.figure(2)
plt.title("Prikaz spektra FT")
plt.plot(x,val2, label = 'meritve')
plt.plot(x,val3, label = 'model, p = '+str(p))
plt.legend()
plt.show()
plt.figure(3)
plt.subplot(2,2,1)
plt.plot(x,val2, label = 'Meritev')
for p in pl:
    bar = bar+1
    a, rho, k = spc.aryule(val3[:i], p)
    for j in a:
        if np.abs(j)>1:
            print('fuck me')
    a = a[::-1]
    PSD = arma2psd(a, NFFT=N)
    #plt.figure(4)
    #plt.plot(x_fft, PSD[:N//2]**2, label = 'p = '+str(p))
    for i in range(N//2, N):
        val3[i] = -np.dot(a, val3[i-p:i] )
    plt.plot(x,val3, color = cm(1.*bar/len(pl)))

plt.title("Napoved rektascenzije lune")
plt.xlabel('Čas')
plt.ylabel('Borza')
plt.grid()
plt.legend()

plt.subplot(2,2,2)
plt.title("Frekvenčni spekter")
plt.plot(x_fft, np.abs(val_fft[0:N//2])**2, label = 'FT')
plt.yscale('log')
plt.ylabel(r'PSD') 
plt.xlabel('Frekvenca')
plt.legend()
plt.grid()


plt.subplot(2,1,2)
#[2,3,4,5,6,7,8,9,10,11,12,13]
bar = 0
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




