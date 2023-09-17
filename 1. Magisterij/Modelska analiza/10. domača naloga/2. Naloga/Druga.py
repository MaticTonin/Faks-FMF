import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.fftpack import fft, fftfreq
from scipy.signal import windows
from scipy.signal import find_peaks
from scipy.signal import peak_widths
import scipy.optimize as opt
import os

#število sample točko
#N = 512 deli 1
#N = 256 deli 2
#N = 128 deli 4
#N = 64  deli 8

def okno(i):
    if i < 100:
        return 0
    if i > 350: 
        return 0
    else:
        return 1

deli = 1
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
N = int(512/deli)


ime = "C:\\Users\\Blaž Levpušček\\Documents\\FMF\\Mag\MA\\110\\druga\\signal2.dat"
cm = plt.get_cmap('nipy_spectral')
#branje datoteke
filename="signal3.dat"
filename_ref="signal0ref.txt"
f=open(THIS_FOLDER +"\\"+filename,"r")
g=open(THIS_FOLDER +"\\"+filename_ref,"r")

lines=f.readlines()
lines_ref=g.readlines()
sig=[]
hamm = windows.hamming(N)
cos = windows.cosine(N)


#naredimo array iz N meritev
t=0
for y in lines:
    if t >= N:
        break
    sig = np.append(sig, float(y.split('\n')[0]))
    t = t + 1  
f.close()

t=0
sig_ref=[]
for y in lines_ref:
    if t >= N:
        break
    sig_ref = np.append(sig_ref, float(y.split('\n')[0]))
    t = t + 1  
g.close()
print(sig_ref)



#sample spacing
x = np.linspace(0,len(sig), len(sig))
plt.figure(1)

plt.xlabel('t')


def r(t,len):
    if t<len/2:
        return (1/(2*16) *np.exp(-np.abs(t)/16))
    else:
        return (1/(2*16) *np.exp(-np.abs(len-t)/16))
plt.figure(3)
plt.plot(np.array([r(t,len(sig)) for t in range(len(sig))]), label='+0')
def r(t,len):
    if t<len/2:
        return (1/(2*16) *np.exp(-np.abs(t)/16))
    else:
        return (1/(2*16) *np.exp(-np.abs(len+1-t)/16))

plt.plot(np.array([r(t,len(sig)) for t in range(len(sig))]), label='+1')
plt.legend()
plt.xlabel('t')
plt.ylabel('r(t)')
plt.grid()
plt.figure(1)
C = np.fft.rfft(sig)
Co = np.fft.rfft(sig*hamm)
Ccos = np.fft.rfft(sig*cos)
Cx = np.fft.rfft([sig[i]*okno(i) for i in range(len(sig))] )

R = np.fft.rfft(np.array([r(t,len(sig)) for t in range(len(sig))]))

def konst(x, a):
    return a

def exp(x,k, c):
    return k*x + c

if ime == "C:\\Users\\Blaž Levpušček\\Documents\\FMF\\Mag\MA\\110\\druga\\signal0.dat":
    meja = 66 #sig 0
if ime == "C:\\Users\\Blaž Levpušček\\Documents\\FMF\\Mag\MA\\110\\druga\\signal1.dat":
    meja = 50 #sig 1
if ime == "C:\\Users\\Blaž Levpušček\\Documents\\FMF\\Mag\MA\\110\\druga\\signal2.dat":
    meja = 20 #sig 2
if ime == "C:\\Users\\Blaž Levpušček\\Documents\\FMF\\Mag\MA\\110\\druga\\signal3.dat":
    meja = 15 #sig 3

popt_konst, _ =opt.curve_fit(konst,x[:257] ,np.abs(C[meja:])**2)
popt_eks, _ =opt.curve_fit(exp,x[:meja] ,np.log(np.abs(C[:meja])**2))

popt_konst_o, _ =opt.curve_fit(konst,x[:257] ,np.abs(Co[meja:])**2)
popt_eks_o, _ =opt.curve_fit(exp,x[:meja] ,np.log(np.abs(Co[:meja])**2))

popt_konst_cos, _ =opt.curve_fit(konst,x[:257] ,np.abs(Ccos[meja:])**2)
popt_eks_cos, _ =opt.curve_fit(exp,x[:meja] ,np.log(np.abs(Ccos[:meja])**2))

popt_konst_x, _ =opt.curve_fit(konst,x[:257] ,np.abs(Cx[meja:])**2)
popt_eks_x, _ =opt.curve_fit(exp,x[:meja] ,np.log(np.abs(Cx[:meja])**2))

S = np.exp(x[:257]*popt_eks[0]+popt_eks[1])
So = np.exp(x[:257]*popt_eks_o[0]+popt_eks_o[1])
Scos = np.exp(x[:257]*popt_eks_cos[0]+popt_eks_cos[1])
Sx = np.exp(x[:257]*popt_eks_x[0]+popt_eks_x[1])

M = popt_konst*np.ones(len(x[:257]))
Mo = popt_konst_o*np.ones(len(x[:257]))
Mcos = popt_konst_cos*np.ones(len(x[:257]))
Mx = popt_konst_x*np.ones(len(x[:257]))

F = S/(S + M)
Fo = So/(So + Mo)
Fcos = Scos/(Scos + Mcos)
Fx = Sx/(Sx + Mx)

print(popt_eks)
print(popt_konst)
U = F*C/R
Uo = Fo*Co/R
Ucos = Fcos*Ccos/R
Ux = Fx*Cx/R

u_1=np.fft.irfft(C/R)
u = np.fft.irfft(U)
uo = np.fft.irfft(Uo)
ucos = np.fft.irfft(Ucos)
ux = np.fft.irfft(Ux)
plt.title("Prikaz signala za "+str(filename))
plt.plot(x, sig, '-', color = 'b', label='Signal')
plt.plot(x, u_1, '-', color = 'g', label='Dekompozicija', alpha=0.5)
plt.plot(x, u, '-', color = 'r', label = 'Dekompozicija + filter')
if filename!="signal0.dat":
    plt.plot(x, sig_ref, '-', color = 'black', label='Signal0')

#plt.plot(x, ucos, '-', color = 'b', label = 'filtriran signal + cosine')
#plt.plot(x, uo, '-', color = 'g', label = 'filtriran signal + Hamming')
#plt.plot(x, ux, '-', color = 'k', label = 'filtriran signal + cut')
plt.grid()
plt.legend()

plt.figure(2)
plt.axvline(x= meja, color = 'k', ls = '--')
plt.plot(x[:257], S, label = 'privzet signal')
plt.plot(x[:257], F, label = 'filter')
plt.plot(x[:257], M, label = 'privzet šum')

plt.plot(x[:257], np.abs(C)**2, label = r'$|C|^2$')
plt.legend()
plt.yscale('log')

plt.xlabel(r'$\nu$')
plt.ylabel('PSD')

plt.show()
