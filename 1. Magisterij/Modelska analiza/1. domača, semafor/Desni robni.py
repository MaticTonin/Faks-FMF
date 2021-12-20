import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
cm = plt.cm.jet
cm1 = plt.cm.winter
cm2 = plt.cm.autumn
import os
import emcee
from scipy.stats import norm
import corner
from scipy.optimize import curve_fit
import os
from sklearn import preprocessing
from mpl_toolkits import mplot3d
N=400
t=np.linspace(0, 1, N)
v=np.linspace(5,0,N)
v00=np.linspace(0,0,N)


def velocity(t, v0, tc):
    a = (1 - v0 * tc * np.tanh(1/tc)) / (1 - tc * np.tanh(1/tc))
    return (v0 - a) * np.cosh(t/tc) - (1 - a) / tc * np.sinh(t/tc) + a

def acceleration(t, v0, tc):
    a = (1 - v0 * tc * np.tanh(1/tc)) / (1 - tc * np.tanh(1/tc))
    return (v0 - a) / tc * np.sinh(t/tc) - (1 - a) / tc**2 * np.cosh(t/tc)
tc=np.linspace(0.05,1,N)


index=0
index_red=0
index_blue=0
mejna=0
a_0=4
t=np.linspace(0, 7, N)
for i in v:
    v_1=[]
    nope=0
    for j in t:
        v11=3*(1-a_0/2-i)*j**2+a_0*j+i
        v_1.append(v11)
        if v11<0:
            if i<mejna:
                mejna=i
            nope=1
    if nope==0:
        index_blue+=1
        plt.plot(t, v_1, "-", color=cm1(index_blue/(N/3)))
    if nope==1:
        index_red+=1
        plt.plot(t, v_1, "-", color=cm2(index_red/(N/2)))
v_meja=[]
for j in t:
    v11=-3/2*(1-mejna)*j**2+3*(1-mejna)*j+mejna
    v_meja.append(v11)
plt.plot(t, v_meja, "-", color="grey", label=r"Zadnja dovoljena krivulja v= %.2f" %(mejna))
plt.xlabel("t")
plt.ylabel("v")


plt.legend()
plt.title("Spreminjanje hitrosti za perioden pogoj, $a_0=%.2f$" %(a_0))
plt.show()

index=0
index_red=0
index_blue=0
mejna=10
t=np.linspace(0, 2*np.pi, N)
for i in v:
    v_1=[]
    nope=0
    for j in t:
        v11=(i+1/(2*np.pi))*np.cos(j)+1/(2*np.pi)*np.sin(j)-1/(2*np.pi)
        v_1.append(v11)
        if v11<0:
            if i<mejna:
                mejna=i
            nope=1
    if nope==0:
        index_blue+=1
        plt.plot(t, v_1, "-", color=cm(index_blue/(N/3)))
    if nope==1:
        index_red+=1
        plt.plot(t, v_1, "-", color=cm(index_red/(N)))
    if index==0:
        plt.plot(t, v_1, "-", color=cm(index/N),label=r"Začetni $v_0=%.2f$" %(i))
    if index==len(v)-1:
        plt.plot(t, v_1, "-", color=cm(index/N),label=r"Končni $v_0=%.2f$" %(i))
    index+=1
v_meja=[]
for j in t:
    v11=-3/2*(1-mejna)*j**2+3*(1-mejna)*j+mejna
    v_meja.append(v11)
plt.xlabel("t")
plt.ylabel("v")


plt.legend()
plt.title("Spreminjanje hitrosti za perioden pogoj, $a_0=%.2f$" %(a_0))
plt.show()

v_0=6
A, B = np.meshgrid(t, tc)
v=velocity(A,v_0,B)
fig4 = plt.figure("Spreminjanje hitrosti s kvadratnim členov, brez desnega robnega pogoja v prostoru pri $v=%.2f$" %(v_0))
ax = fig4.add_subplot(111, projection = '3d')
im = ax.contour3D(A, B, v, 950, cmap=cm)
fig4.colorbar(im, label="Hitrost v")
ax.set_xlabel("t")
ax.set_ylabel("$t_c$")
ax.set_title("Spreminjanje hitrosti s kvadratnim členov, brez desnega robnega pogoja v prostoru pri $v=%.2f$" %(v_0))
ax.set_zticks([])

plt.legend(loc = "lower left")
plt.show()

index=0
index_red=0
index_blue=0
v_0=6
for i in tc:
    v_1=[]
    for j in t:
        v11=velocity(j,v_0,i)
        v_1.append(v11)
    if index==0:
        plt.plot(t, v_1, "-", color=cm(index/N),label=r"Začetni $t_0=%.2f$" %(i))
    if index==len(tc)-1:
        plt.plot(t, v_1, "-", color=cm(index/N),label=r"Končni $t_0=%.2f$" %(i))
    else: 
        plt.plot(t, v_1, "-", color=cm(index/N))
    index+=1
plt.xlabel("t")
plt.ylabel("v")


plt.legend()
plt.title("Spreminjanje hitrosti s kvadratnim čelnom, brez desnega robnega $v_0=%.2f$" %(v_0))
plt.show()

t=np.linspace(0, 1, N)
v=np.linspace(5,0,N)
index=0
index_red=0
index_blue=0
v_0=6
mejna=10
for i in tc:
    v_1=[]
    nope=0
    for j in t:
        v11=velocity(j,v_0,i)
        v_1.append(v11)
    if v_1[-1]<=0:
        if j>mejna:
            mejna=j
        nope=1
    if nope==0:
        index_blue+=1
        plt.plot(t, v_1, "-", color=cm1(index_blue/(N/5)))
        if index==0:
            plt.plot(t, v_1, "-", color=cm1(index_blue/(N/5)),label=r"Začetni $t_0=%.2f$" %(i))
        if index==len(tc)-1:
            plt.plot(t, v_1, "-", color=cm1(index_blue/(N/5)),label=r"Končni $t_0=%.2f$" %(i))
        else: 
            plt.plot(t, v_1, "-", color=cm1(index_blue/(N/5)))
    if nope==1:
        index_red+=1
        if index==0:
            plt.plot(t, v_1, "-", color=cm2(index_red/(N/3)),label=r"Začetni $t_0=%.2f$" %(i))
        if index==len(tc)-1:
            plt.plot(t, v_1, "-", color=cm2(index_red/(N/3)),label=r"Končni $t_0=%.2f$" %(i))
        else: 
            plt.plot(t, v_1, "-", color=cm2(index_red/(N/3)))
    index+=1
v_meja=[]
for j in t:
    v11=velocity(j,v_0,mejna)
    v_meja.append(v11)
plt.plot(t, v00, "-", color="black", label="Spodnja meja")
plt.plot(t, v_meja, "-", color="grey", label=r"Zadnja dovoljena krivulja t_0= %.2f" %(mejna))
plt.xlabel("t")
plt.ylabel("v")


plt.legend()
plt.title("Spreminjanje hitrosti s kvadratnim čelnom, brez desnega robnega $v_0=%.2f$" %(v_0))
plt.show()

index=0
index_red=0
index_blue=0
v_0=6
sum_meja=0
for i in v:
    v_1=[]
    nope=0
    mejna=0.001
    for j in tc:
        for k in t:
            v11=velocity(k,i,j)
            v_1.append(v11)
        if abs(v_1[-1])<0.01:
            if j>mejna:
                mejna=j
    print(mejna)
    v_meja=[]
    for z in t:
        v11=velocity(z,i,mejna)
        v_meja.append(v11)
    if index==0 and mejna!=0.001:
        plt.plot(t, v_meja, "-", color=cm2(index/N),alpha=0.3,label=r"Začetni $t_0=%.2f, v_0=%.2f$" %(tc[0], v[0]))
    if index==len(v)-1 and mejna!=0.001:
        plt.plot(t, v_meja, "-", color=cm2(index/N),alpha=0.3,label=r"Končni $t_0=%.2f, v_0=%.2f$" %(tc[len(tc)-1], v[len(v)-1]))
    if index%15==0 and mejna!=0.001:
        plt.plot(t, v_meja, "-", color=cm1((int(index/15))/7),label=r"$t_0=%.2f, v_0=%.2f$" %(tc[index-1], v[index-1]))
    elif  mejna!=0.001: 
        plt.plot(t, v_meja, "-",alpha=0.3, color=cm2(index/N))
    index+=1
    print(index)
    mejna=10

plt.plot(t, v00, "-", color="black", label="Spodnja meja")
plt.xlabel("t")
plt.ylabel("v")


plt.legend()
plt.title( "Spreminjanje hitrosti s kvadratnim čelnom, ugotavljaje meje $t_c$ v odvisnosti od roba,")
plt.show()

index=0
index_red=0
index_blue=0
t_c=0.5
for i in v:
    v_1=[]
    for j in t:
        v11=velocity(j,i,t_c)
        v_1.append(v11)
    plt.plot(t, v_1, "-", color=cm(index/N))
    index+=1
plt.xlabel("t")
plt.ylabel("v")


plt.legend()
plt.title("Spreminjanje hitrosti s kvadratnim čelnom, brez desnega robnega $t_c=%.2f$" %(t_c))
plt.show()

index=0
index_red=0
index_blue=0
t_c=0.5
mejna=10
for i in v:
    v_1=[]
    nope=0
    for j in t:
        v11=velocity(j,i,t_c)
        v_1.append(v11)
        if v11<0:
            if i<mejna:
                mejna=i
            nope=1
    if nope==0:
        index_blue+=1
        plt.plot(t, v_1, "-", color=cm1(index_blue/(N/3)))
    if nope==1:
        index_red+=1
        plt.plot(t, v_1, "-", color=cm2(index_red/(N/2)))
v_meja=[]
for j in t:
    v11=velocity(j,i,t_c)
    v_meja.append(v11)
plt.plot(t, v00, "-", color="black", label="Spodnja meja")
plt.plot(t, v_meja, "-", color="grey", label=r"Zadnja dovoljena krivulja v= %.2f" %(mejna))
plt.xlabel("t")
plt.ylabel("v")


plt.legend()
plt.title("Spreminjanje hitrosti s kvadratnim čelnom, brez desnega robnega $t_c=%.2f$" %(t_c))
plt.show()

index=0
index_red=0
index_blue=0
for i in v:
    v_1=[]
    for j in t:
        v11=-3/2*(1-i)*j**2+3*(1-i)*j+i
        v_1.append(v11)
    plt.plot(t, v_1, "-", color=cm(index/N))
    index+=1
plt.plot(t, v00, "-", color="black", label="Spodnja meja")
plt.xlabel("t")
plt.ylabel("v")


plt.legend()
plt.title("Spreminjanje hitrosti s kvadratno mero, brez desnega robnega")
plt.show()

index=0
index_red=0
index_blue=0
mejna=0
for i in v:
    v_1=[]
    nope=0
    for j in t:
        v11=-3/2*(1-i)*j**2+3*(1-i)*j+i
        v_1.append(v11)
        if v11<0:
            if i<mejna:
                mejna=i
            nope=1
    if nope==0:
        index_blue+=1
        plt.plot(t, v_1, "-", color=cm1(index_blue/(N/3)))
    if nope==1:
        index_red+=1
        plt.plot(t, v_1, "-", color=cm2(index_red/(N/2)))
v_meja=[]
for j in t:
    v11=-3/2*(1-mejna)*j**2+3*(1-mejna)*j+mejna
    v_meja.append(v11)
plt.plot(t, v_meja, "-", color="grey", label=r"Zadnja dovoljena krivulja v= %.2f" %(mejna))
plt.xlabel("t")
plt.ylabel("v")


plt.legend()
plt.title("Spreminjanje hitrosti s kvadratno mero, brez desnega robnega")
plt.show()

index=0
index_red=0
index_blue=0
mejna=10
t=np.linspace(0, 2*np.pi, N)
for i in v:
    v_1=[]
    nope=0
    for j in t:
        v11=(i+1/(2*np.pi))*np.cos(j)+1/(2*np.pi)*np.sin(j)-1/(2*np.pi)
        v_1.append(v11)
        if v11<0:
            if i<mejna:
                mejna=i
            nope=1
    if nope==0:
        index_blue+=1
        plt.plot(t, v_1, "-", color=cm(index_blue/(N/3)))
    if nope==1:
        index_red+=1
        plt.plot(t, v_1, "-", color=cm(index_red/(N)))
v_meja=[]
for j in t:
    v11=-3/2*(1-mejna)*j**2+3*(1-mejna)*j+mejna
    v_meja.append(v11)
plt.xlabel("t")
plt.ylabel("v")


plt.legend()
plt.title("Spreminjanje hitrosti s kvadratno mero hitrosti, brez desnega robnega")
plt.show()
N=10
index=0
index_red=0
index_blue=0
mejna=10
n=np.linspace(1, 10, N)
v_0=0.5
for i in n:
    v_1=[]
    nope=0
    for j in t:
        prvi=(i**(i/(i-1)))
        drugi=(i-1)**(2/(i-2))
        tretji=(1-v_0)**(1/(i-2))
        lamba=-prvi*drugi/tretji
        v11=-(i-1)/lamba*(-lamba*j/i)**(i/(i-1))+v_0
        v_1.append(v11)
        if v11<0:
            if i<mejna:
                mejna=i
            nope=1
    if nope==0:
        index_blue+=1
        plt.plot(t, v_1, "-", color=cm(index_blue/(N)), label=r"n= %.2f" %(i))
    if nope==1:
        index_red+=1
        plt.plot(t, v_1, "-", color=cm2(index_red/(N/2)))
v_meja=[]
for j in t:
    v11=-3/2*(1-mejna)*j**2+3*(1-mejna)*j+mejna
    v_meja.append(v11)
plt.xlabel("t")
plt.ylabel("v")


plt.legend()
plt.title(r"Spreminjanje hitrosti s kvadratno mero in višjimi potencami, brez desnega robnega pogoja")
plt.show()
A, B = np.meshgrid(t, n)
prvi=(B**(B/(B-1)))
drugi=(B-1)**(2/(B-2))
tretji=(1-v_0)**(1/(B-2))
lamba=-prvi*drugi/tretji
v=-(B-1)/lamba*(-lamba*A/B)**(B/(B-1))+v_0
fig4 = plt.figure("Spreminjanje hitrosti s kvadratno mero in višjimi potencami, brez desnega robnega pogoja v prostoru")
ax = fig4.add_subplot(111, projection = '3d')
im = ax.contour3D(A, B, v, 950, cmap=cm)
fig4.colorbar(im, label="Hitrost v")
ax.set_xlabel("t")
ax.set_ylabel("n")
ax.set_title("Spreminjanje hitrosti s kvadratno mero in višjimi potencami, brez desnega robnega pogoja v prostoru ")
ax.set_zticks([])

plt.legend(loc = "lower left")
plt.show()