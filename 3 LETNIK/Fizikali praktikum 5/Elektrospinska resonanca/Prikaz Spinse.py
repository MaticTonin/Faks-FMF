import os
from sklearn import preprocessing
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt



THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
nu=80
I, U = np.loadtxt(THIS_FOLDER + "\\"+ str(nu) +" MHz.txt",delimiter = "\t", unpack ="True")
max=0
min=0
I_max=0
I_min=0
for i in range(len(U)):
    if max<U[i]:
        print(U[i])
        I_max=I[i]
        max=U[i]
    if min>U[i]:
        I_min=I[i]
        min=U[i]
plt.plot(I, U, label=r"$\nu=%.2f, I_{max}=%.2f, I_{mi}=%.2f$" %(nu, I_max, I_min))

nu=83
I, U = np.loadtxt(THIS_FOLDER + "\\"+ str(nu) +" MHz.txt",delimiter = "\t", unpack ="True")
max=0
min=0
I_max=0
I_min=0
for i in range(len(U)):
    if max<U[i]:
        print(U[i])
        I_max=I[i]
        max=U[i]
    if min>U[i]:
        I_min=I[i]
        min=U[i]
plt.plot(I, U, label=r"$\nu=%.2f, I_{max}=%.2f, I_{mi}=%.2f$" %(nu, I_max, I_min))

nu=85
I, U = np.loadtxt(THIS_FOLDER + "\\"+ str(nu) +" MHz.txt",delimiter = "\t", unpack ="True")
max=0
min=0
I_max=0
I_min=0
for i in range(len(U)):
    if max<U[i]:
        print(U[i])
        I_max=I[i]
        max=U[i]
    if min>U[i]:
        I_min=I[i]
        min=U[i]
plt.plot(I, U, label=r"$\nu=%.2f, I_{max}=%.2f, I_{mi}=%.2f$" %(nu, I_max, I_min))

nu=90
I, U = np.loadtxt(THIS_FOLDER + "\\"+ str(nu) +" MHz.txt",delimiter = "\t", unpack ="True")
max=0
min=0
I_max=0
I_min=0
for i in range(len(U)):
    if max<U[i]:
        print(U[i])
        I_max=I[i]
        max=U[i]
    if min>U[i]:
        I_min=I[i]
        min=U[i]
plt.plot(I, U, label=r"$\nu=%.2f, I_{max}=%.2f, I_{mi}=%.2f$" %(nu, I_max, I_min))

plt.xlabel("I[mA]")
plt.ylabel("U[mV]")
plt.legend()
plt.title(r"Spreminjanje napetosti v odvisnosti od toka")
plt.show()

nu=[80,83,85,90]
max=[270,282,288,303]
min=[282,295,300,317]
max=np.array(max)
min=np.array(min)
plt.subplot(2, 1, 1)
plt.title(r"Spreminjanje vrhov v odvisnosti od toka")
plt.plot(nu, max, label="Spreminjanje max")
plt.plot(nu, min, label="Spreminjanje min")
plt.xlabel(r"$\nu$[MHz]")
plt.ylabel("I[mA]")
plt.subplot(2, 1, 2)
plt.plot(nu, min-max, label="Spreminjanje njune razlike")

plt.xlabel(r"$\nu$[MHz]")
plt.ylabel("I[mA]")
plt.legend()
plt.show()
frekvence=[80,83,85,90]
for nu in frekvence:
    I, U = np.loadtxt(THIS_FOLDER + "\\"+ str(nu) +" MHz.txt",delimiter = "\t", unpack ="True")
    max=0
    min=0
    I_max=0
    I_min=0
    zero=0
    index=0
    for i in range(len(U)):
        if max<U[i]:
            I_max=I[i]
            max=U[i]
        if min>U[i]:
            I_min=I[i]
            min=U[i]
        if U[i]<0 and index==0:
            index=1
            zero=(I[i]+I[i-1])/2

    plt.plot(I, U, label=r"$\nu=%.2f, I_{max}=%.2f, I_{mi}=%.2f$" %(nu, I_max, I_min))
    print(r"$\nu=%.2f, I_0=%.2f, I_{max}=%.2f, I_{mi}=%.2f$" %(nu, zero, I_max, I_min))
    Nmu_d=0.010657177
    B_0=Nmu_d*zero*10**(-3)
    delta_B=Nmu_d*(I_max-I_min)*10**(-3)
    B_0_nu=B_0/(nu*10**(6))
    g=6.626*10**(-34)*nu*10**(6)/(B_0*9.27*10**(-24))
    print(r"$B_0="+str(B_0)+", B_0/\nu="+str(B_0_nu)+", g=$"+str(g)+ "\\n")
    print(delta_B)


plt.xlabel("I[mA]")
plt.ylabel("U[mV]")
plt.legend()
plt.title(r"Spreminjanje napetosti v odvisnosti od toka, zgolj uporabne")
plt.show()