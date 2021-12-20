import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import time
import os
def Autocorelation1(Inverse): #naredi autokorelacijo
    N=len(Inverse)
    Auto=[]
    F=np.fft.ifftshift(Inverse)
    for i in range(N):
        Auto.append((1/N-i)*F[i])
    Auto=Auto[:-1]
    return Auto

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
    #F=np.fft.ifftshift(Inverse)
    for i in range(N):
        Auto.append(1/N-i*F[i])
    return Auto

samples1=44100
samples2=44100
mapa1="Velika uharica 1"
#mapa1="Velika uharica 2"
#mapa2="Cricek pri potoku"
#mapa2="Cricek" 
#mapa2="Deroča reka 2"
mapa2="Deroča reka"
#mapa1="Bach.44100"
#mapa2="Bach.882"
barva="crimson"
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
fname1 = os.path.join(THIS_FOLDER, str(mapa1) + ".txt")
fname2 = os.path.join(THIS_FOLDER, str(mapa2) + ".txt")
f1=np.loadtxt(fname1, dtype = int)
f2=np.loadtxt(fname2, dtype = int)
f1=Podaljšanje(f1)
f2=Podaljšanje(f2) #podaljša podatke, da so dolžine 2N 
H1=np.fft.fft(f1) 
H2=np.fft.fft(f2) #da naredi
H_12=np.absolute(H1)**2
H_22=np.absolute(H2)**2
start1 = time.time()
F_inverse1=np.fft.ifft(H_12)
F_inverse2=np.fft.ifft(H_22)
Auto1=Autocorelation1(F_inverse1)
Auto2=Autocorelation1(F_inverse2)
Spekt1=np.absolute(np.fft.ifftshift(np.fft.fft(Auto1)))**2
Spekt2=np.absolute(np.fft.ifftshift(np.fft.fft(Auto2)))**2
end1 = time.time()

N1=len(Auto1)
delta1=N1/samples1 #časovni intervali
t1=np.linspace(0, delta1 ,N1)
fc1 =samples1/2.
freq1 = np.linspace(-fc1, fc1, N1)
N2=len(Auto2)
delta2=N2/samples2 #časovni intervali
t2=np.linspace(0, delta2 ,N2)
fc2 =samples2/2.
freq2 = np.linspace(-fc2, fc2, N2)
print(end1-start1)



#Autokorelacija
plt.plot(t1,Auto1/max(Auto1),'--',color="blue", label="Autokorelacija 1" + str(mapa1))
plt.plot(t2,Auto2/max(Auto2),'--',color="red", label="Autokorelacija 2" + str(mapa2))
plt.title("Podatki za posnetek "+ str(mapa1) + ", z autokorelacijo")
plt.xlabel("t")
plt.ylabel("f(x)")
plt.legend()
plt.show()


"""plt.plot(abs(freq1),H_12/max(H_12),'-',color="gray", label="Brez autokorelacije")
plt.plot(abs(freq1),Spekt1/max(Spekt1),'-', color="lime",  label="Autokorelacija za "+ str(mapa1))"""
plt.plot(abs(freq2),H_22/max(H_22),'-',color="gray", label="Brez autokorelacije")
plt.plot(abs(freq2),Spekt2/max(Spekt2),'-',color="lime", label="Autokorelacija za " + str(mapa2))
plt.axis([-10,700, -0.03,1.05])
plt.legend()
plt.title('Spektrograf za '+ str(mapa1) + " in "+ str(mapa2))
plt.xlabel("$\omega$")
plt.ylabel("Re[FT($\omega$)]")
plt.show()