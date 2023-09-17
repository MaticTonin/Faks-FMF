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
ime = "C:\\Users\\Blaž Levpušček\\Documents\\FMF\\Mag\MA\\110\\druga\\signal2.dat"
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
def okno(i):
    if i < 100:
        return 0
    if i > 350: 
        return 0
    else:
        return 1

def r(t,len):
    if t<len/2:
        return (1/(2*16) *np.exp(-np.abs(t)/16))
    else:
        return (1/(2*16) *np.exp(-np.abs(len-t)/16))
def r(t,len):
    if t<len/2:
        return (1/(2*16) *np.exp(-np.abs(t)/16))
    else:
        return (1/(2*16) *np.exp(-np.abs(len+1-t)/16))

def konst(x, a):
    return a

def exp(x,k, c):
    return k*x + c


def izdelava(N,filename):
    filename_ref="signal0ref.txt"
    f=open(THIS_FOLDER +"\\"+filename,"r")
    g=open(THIS_FOLDER +"\\"+filename_ref,"r")

    lines=f.readlines()
    lines_ref=g.readlines()
    sig=[]
    hamm = windows.hamming(N)
    cos = windows.cosine(N)        
    t=0
    for y in lines:
        if t >= N:
            break
        sig = np.append(sig, float(y.split('\n')[0]))
        t = t + 1  
    f.close()
    x = np.linspace(0,len(sig), len(sig))
    C = np.fft.rfft(sig)
    Co = np.fft.rfft(sig*hamm)
    Ccos = np.fft.rfft(sig*cos)
    Cx = np.fft.rfft([sig[i]*okno(i) for i in range(len(sig))] )
    R = np.fft.rfft(np.array([r(t,len(sig)) for t in range(len(sig))]))
    if filename == "signal0.dat":
        meja = 66 #sig 0
    if filename == "signal1.dat":
        meja = 50 #sig 1
    if filename == "signal2.dat":
        meja = 20 #sig 2
    if filename == "signal3.dat":
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
    U = F*C/R
    Uo = Fo*Co/R
    Ucos = Fcos*Ccos/R
    Ux = Fx*Cx/R

    u_1=np.fft.irfft(C/R)
    u = np.fft.irfft(U)
    uo = np.fft.irfft(Uo)
    ucos = np.fft.irfft(Ucos)
    ux = np.fft.irfft(Ux)
    return x,sig, u_1,u,uo,ucos,ux,S,F,M,C,meja

filenames=["signal0.dat","signal1.dat","signal2.dat","signal3.dat"]
filenames=["signal3.dat"]
deli=1
N = int(512/deli)
index=0
plt.title("Prikaz spektralne moči za vse signale")
for i in filenames:
    x, sig, u_1,u,uo,ucos,ux,S,F,M,C,meja=izdelava(N,i)
    #plt.axvline(x= meja, color = 'k', ls = '--')
    plt.plot(x[:257], np.abs(C)**2, alpha=0.001, label="Signal: "+str(i))
    plt.plot(x[:257], np.abs(C)**2, label = r"$|C_{%i}|^2$" %(index),color = plt.cm.Blues((index+3)/(len(filenames)+3)))
    plt.plot(x[:257], S, label = r"$|S_{%i}|^2$" %(index), color=plt.cm.Greens((index+3)/(len(filenames)+3)))
    #plt.plot(x[:257], F, label = r"$|F_{%i}|^2$" %(index))
    plt.plot(x[:257], M, label = r"$|M_{%i}|^2=%.6f$" %(index,M[0]),color=plt.cm.Reds((index+3)/(len(filenames)+3)))
    index+=1
if len(filenames)>=2:
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend()
plt.grid()
plt.yscale('log')

plt.xlabel(r'$\nu$')
plt.ylabel('PSD')
plt.show()