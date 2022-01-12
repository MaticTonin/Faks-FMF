import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import time
import os

def Podaljšanje(f):  #število podaljšanj
    f=np.asarray(f,dtype=float)
    N=len(f)
    h=np.zeros(N*2)
    for i in range(N):
            h[i]=f[i]
    return h

def Autocorelation1(Inverse): #naredi autokorelacijo
    N=len(Inverse)
    Auto=[]
    F=Inverse
    F=np.fft.ifftshift(Inverse)
    for i in range(N):
        Auto.append((1/(N-i))*F[i])
    return Auto

samples=44100
#mapa="Velika uharica 1"
#mapa="Velika uharica 2"
#mapa="Cricek pri potoku"
mapa="Cricek" 
#mapa="Deroča reka 2"
#mapa="Deroča reka"
#mapa="Bach.44100"
#mapa="Rick.wav"
#mapa="Tell me why"
barva="crimson"
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(THIS_FOLDER, str(mapa) + ".txt")
f1=np.loadtxt(fname, dtype = int)
H1=np.fft.fft(f1)  #da naredi
H_21=np.fft.ifftshift((np.absolute(H1)**2))
N1=len(H1)
delta=N1/samples
t1=np.linspace(0, delta ,N1)
fc1 =samples/2.
freq1 = np.linspace(-fc1, fc1, N1)
F_inverse1=np.fft.ifft(H_21)
#N=20000
#delta=50*np.pi
#t=np.linspace(0, delta- (delta/N),N)
#f=np.sin(t)
f=Podaljšanje(f1) #podaljša podatke, da so dolžine 2N 
H=np.fft.fft(f)  #da naredi
H_2=np.absolute(H)**2
start1 = time.time()
F_inverse=np.fft.ifft(H_2)
Auto=Autocorelation1(F_inverse)
Spekt=np.absolute(np.fft.ifftshift(np.fft.fft(Auto)))**2
end1 = time.time()

N=len(Auto)
delta=N/samples #časovni intervali
t=np.linspace(0, delta ,N)
fc =samples/2.
freq = np.linspace(-fc, fc, N)

print(end1-start1)

plt.plot(t1,f1/max(f1),'-',color=barva)
#plt.plot(t,Auto/max(Auto),'-',color="blue")
plt.title("Podatki za posnetek "+ str(mapa) + ", 41100 Hz")
plt.xlabel("t")
plt.ylabel("f(x)")
plt.legend()
plt.show()


plt.plot(abs(freq1),H_21/max(H_21),'-',color=barva)
plt.title('Spektrograf za '+ str(mapa))
plt.xlabel("$\omega$")
plt.ylabel("Re[FT($\omega$)]")
plt.axis([-10,2000, -0.03,1.05])
plt.legend()
plt.show()

#Autokorelacija
plt.plot(t,f/max(f),'-',color=barva, label="Podaljšan original")
plt.plot(t,Auto/max(Auto),'-',color="blue", label="Autokorelacija")
plt.title("Podatki za posnetek "+ str(mapa) + ", z autokorelacijo")
plt.xlabel("t")
plt.ylabel("f(x)")
plt.legend()
plt.show()


plt.plot(abs(freq1),H_21/max(H_21),'-',color="gray", label="Brez autokorelacije")
plt.plot(abs(freq),Spekt/max(Spekt),'-',color=barva, label="Z autokorelacijo")
plt.title('Spektrograf za '+ str(mapa) + ", z autokorelacijo")
plt.xlabel("$\omega$")
plt.ylabel("Re[FT($\omega$)]")
plt.axis([-10,2000, -0.03,1.05])
plt.legend()
plt.show()