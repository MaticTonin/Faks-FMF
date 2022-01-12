import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.core.numeric import identity
import scipy
from scipy import special

def matrikaHdod(N):
    H_0=np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i==j:
                H_0[i,j]=1/2+i
    q_4=np.zeros((N,N))
    for i in range(N):
         for j in range(N):
             if i==j+4:
                 q_4[i,j]=0.25 * np.sqrt((j+1)*(j+2)*(j+3)*(j+4))
             if i==j+2:
                 q_4[i,j]=0.25 * (4*j + 6) * np.sqrt((j+1) * (j+2))
             if i==j:
                 q_4[i,j]=0.25 * (6*j**2 + 6*j + 3)
             if i==j-2:
                 q_4[i,j]=0.25 * (4*j - 2) * np.sqrt(j * (j-1))
             if i==j-4:
                 q_4[i,j]=0.25 * np.sqrt(j*(j-1)*(j-2)*(j-3))
    q_2=np.zeros((N,N))
    for i in range(N):
         for j in range(N):
             if i==j-2:
                 q_2[i,j]=q_2[i,j]+1/2* np.sqrt(j*(j-1))
             if i==j:
                 q_2[i,j]=q_2[i,j]+1/2*(2*j+1)
             if i==j+2:
                  q_2[i,j]=q_2[i,j]+ 1/2*np.sqrt((j+1)*(j+2))
     
    H=H_0-5/2*q_2+1/10*q_4
    return H

def izdelava_w(H,N):
    stolpec=np.zeros(N) #VZAMEMO SAMO PRVI STOLPEC, NA KATEREGA IZVAJAMO PROJEKCIJO
    for i in range(N):
            stolpec[i]=H[i,0]
    norm=np.linalg.norm(stolpec) #2. NORMA STOLPCA
    w=np.zeros(N)
    for i in range(N):
        if i==0:
            w[i]=stolpec[i]+np.sign(stolpec[0])*np.linalg.norm(stolpec)
        else:
            w[i]=stolpec[i]
    return w

def Householder1(H,N):
    if N==10000000:
        return H
    else:
        w=izdelava_w(H,N)
        P=np.identity(N)-(2/np.dot(w, w))* np.outer(w,w) #MATRIKA ZRCALJENJA PREKO HIPERAVNINE
        H_diag=np.zeros((N,N))
        for i in range(N):
            H_diag[i]=np.matmul(P,H[i])   #MNOŽENJE TE MATRIKE Z VSEMI VEKTORJI V H
        return H_diag, P


def iteracija_na_stolpcu(H, N, eps, K):
    I=identity(N)
    i=0
    D=np.zeros((N,N))
    while i<K:
        for j in range(N):
         for k in range(N):
                if j!=k and abs(H[j,k])>eps:
                    P=Householder1(H,N)[1]
                    B=np.matmul(P,H)
                    H=np.matmul(B,P)
                    i+=1
        #if      N-i==0:
            #B.append(R[N-2,N-2]
    vrednost=H[0,0]
    return H, vrednost

def postopek(N, eps, K):
    a=1
    R=matrikaHdod(N)
    i=0
    H=np.zeros((N,N))
    vrednost=[]
    if N==2 or N==1: 
        for i in range(N):
            for j in range(N):
                if i==j:
                    vrednost.append(R[i,j])
        H=R
    else: 
        while i<N-2:
            R=iteracija_na_stolpcu(R, N-i, eps, K)[0]
            vrednost.append(iteracija_na_stolpcu(R,N-i, eps, K)[1])
            for j in range(len(R)):
             for k in range(len(R)):
                 H[j+i,k+i]=R[j,k]
            #B.append(R[0])
            R=R[1:N,1:N]
            #print(R)
            i+=1
        vrednost.append(H[N-2,N-2])
        vrednost.append(H[N-1,N-1])
    return H, vrednost
n=50
N=np.linspace(0,n,n)
dejanskeHdod=np.sort(np.linalg.eig(matrikaHdod(n))[0])
householder1=sorted(postopek(n, 10**(-18), 100)[1],reverse=False)
print(householder1)

plt.plot(N,dejanskeHdod,'o', label= "Dejanske Lastne vrednost" )
plt.plot(N,householder1,'-', label= "Lastne z Householderjem" )
plt.title('Odvisnost lastnih energij v potencialu $H_0-5/2*q_2+1/10*q_4$')
plt.xlabel("$N$")
plt.ylabel("$E_(n)$")
plt.legend()
plt.show()

absolute1=[]
for i in range(n):
    print(i)
    absolute1.append(np.log(abs(householder1[i]-dejanskeHdod[i])))
plt.plot(N,absolute1,'-', label= "Dostopanje za $5/2*q_2+1/10*q_4$" )
plt.title('Odvisnost odstopanja za našo matriko H v log skali')
plt.xlabel("$N$")
plt.ylabel("$\log(E_(n))$")
plt.legend()
plt.show()


x=np.linspace(-5,5,100)
def lastne_za_x(N, x, n, H):
    funkcija=0
    for i in range(N):
        funkcija= funkcija+ (2. ** i * np.math.factorial(i) * np.pi ** 0.5) ** (-0.5) *np.exp(-x**(2)/2) * special.hermite(i,0)(x) * (H[i,n]) 
        #print(funkcija)
    return funkcija

lastne_0=[]
lastne_1=[]
lastne_2=[]
lastne_3=[]
lastne_4=[]
lastne_5=[]
lastne_6=[]
lastne_7=[]
lastne_8=[]
lastne_9=[]
crta0=[]
crta1=[]
crta2=[]
crta3=[]
crta4=[]
crta5=[]
crta6=[]
crta7=[]
crta8=[]
crta9=[]

meja=[]
meja1=[]
lambd=1
q=np.linspace(-5,5, 100)
H=np.linalg.eigh(matrikaHdod(100))[1]
vrednost=np.linalg.eigh(matrikaHdod(100))[0]
for i in range(10):
    print(special.hermite(i,0)(1))
for i in range(len(x)):
    print(i)
    meja1.append(q[i]**2)
    meja.append(-2*q[i]**2+1/10*(q[i]**4))
    crta0.append(vrednost[0])
    crta1.append(vrednost[1])
    crta2.append(vrednost[2])
    crta3.append(vrednost[3])
    crta4.append(vrednost[4])
    crta5.append(vrednost[5])
    crta6.append(vrednost[6])
    crta7.append(vrednost[7])
    crta8.append(vrednost[8])
    crta9.append(vrednost[9])
    lastne_0.append(lastne_za_x(100,x[i],0,H)+(vrednost[0]))
    lastne_1.append(lastne_za_x(100,x[i],1,H)+(vrednost[1]))
    lastne_2.append(lastne_za_x(100,x[i],2,H)+vrednost[2])
    lastne_3.append(lastne_za_x(100,x[i],3,H)+vrednost[3])
    lastne_4.append(lastne_za_x(100,x[i],4,H)+vrednost[4])
    lastne_5.append(lastne_za_x(100,x[i],5,H)+vrednost[5])
    lastne_6.append(lastne_za_x(100,x[i],6,H)+vrednost[6])
    lastne_7.append(lastne_za_x(100,x[i],7,H)+vrednost[7])
    lastne_8.append(lastne_za_x(100,x[i],8,H)+vrednost[8])
    lastne_9.append(lastne_za_x(100,x[i],9,H)+vrednost[9])
    #lastne_7.append(lastne_za_x(8,x[i],7,np.linalg.eig(matrika_H2(8,0.2))[1], np.linalg.eig(matrika_H2(8,0.2))[0])+8+(np.sort(np.linalg.eig(matrika_H2(8,0.2))[0]))[7])
    #lastne_8.append(lastne_za_x(9,x[i],8,np.linalg.eig(matrika_H2(9,0.2))[1], np.linalg.eig(matrika_H2(9,0.2))[0])+9+(np.sort(np.linalg.eig(matrika_H2(9,0.2))[0]))[8])
    #lastne_9.append(lastne_za_x(10,x[i],9,np.linalg.eig(matrika_H2(10,0.2))[1], np.linalg.eig(matrika_H2(10,0.2))[0])+10+(np.sort(np.linalg.eig(matrika_H2(10,0.2))[0]))[9])
    #lastne_1.append(lastne_za_x(10,x[i],2,0,postopek(10, 1, 10**(-16), 20)[0]))
sprememba0=[]
sprememba1=[]
sprememba2=[]
sprememba3=[]
sprememba4=[]
for i in range(len(x)):
        print(i)
        sprememba0.append(np.log(abs(lastne_za_x(100,x[i],4,H)-lastne_za_x(5,x[i],4,H))))
        sprememba1.append(np.log(abs(lastne_za_x(100,x[i],4,H)-lastne_za_x(10,x[i],4,H))))
        sprememba2.append(np.log(abs(lastne_za_x(100,x[i],4,H)-lastne_za_x(20,x[i],4,H))))
        sprememba3.append(np.log(abs(lastne_za_x(100,x[i],4,H)-lastne_za_x(50,x[i],4,H))))
        sprememba4.append(np.log(abs(lastne_za_x(100,x[i],4,H)-lastne_za_x(70,x[i],4,H))))


plt.subplot(221)
plt.plot(x,sprememba0,'-', label= "Velikost matrike n=5" )
plt.title('Odvisnost spreminjanja odstopanj lastnih funkcij n=5')
plt.xlabel("$x$")
plt.ylabel("$log(\delta E_(n))$")
plt.legend()
#plt.savefig('Lastne vrednosti od lambde (50) in q^2')

plt.subplot(222)
plt.plot(x,sprememba1,'-', label= "Velikost matrike n=10" )
plt.title('Odvisnost spreminjanja odstopanj lastnih funkcij n=10')
plt.xlabel("$x$")
plt.ylabel("$log(\delta E_(n))$")
plt.legend()
#plt.savefig('Lastne vrednosti od lambde (50) in q^2')

plt.subplot(223)
plt.plot(x,sprememba2,'-', label= "Velikost matrike n=20" )
plt.title('Odvisnost spreminjanja odstopanj lastnih funkcij n=20')
plt.xlabel("$x$")
plt.ylabel("$log(\delta E_(n))$")
plt.legend()
#plt.savefig('Lastne vrednosti od lambde (50) in q^2')

plt.subplot(224)
plt.plot(x,sprememba3,'-', label= "Velikost matrike n=50" )
plt.title('Odvisnost spreminjanja odstopanj lastnih funkcij n=50')
#plt.axis([-5,6,-1,28])
plt.xlabel("$x$")
plt.ylabel("$log(\delta E_(n))$")
plt.legend()
#plt.savefig('Lastne vrednosti od lambde (50) in q^2')
plt.show()


plt.plot(x,lastne_0,'-', label= "n=0" )
plt.plot(x,crta0,"--", color="silver")
plt.plot(x,crta1,"--", color="silver")
plt.plot(x,crta2,"--", color="silver")
plt.plot(x,crta3,"--", color="silver")
plt.plot(x,crta4,"--", color="silver")
plt.plot(x,crta5,"--", color="silver")
plt.plot(x,crta6,"--", color="silver")
plt.plot(x,crta7,"--", color="silver")
plt.plot(x,crta8,"--", color="silver")
plt.plot(x,crta9,"--", color="silver")
plt.text(5.1, vrednost[1], "$E_{0,1}$=$%.2f $" %vrednost[1])
plt.text(5.1, vrednost[3], "$E_{2,3}$=$%.2f $" %vrednost[3])
plt.text(5.1, vrednost[5], "$E_{4,5}$=$%.2f $" %vrednost[5])
plt.text(5.1, vrednost[7], "$E_{6,7}$=$%.2f $" %vrednost[7])
plt.text(5.1, vrednost[8], "$E_8$=$%.2f $" %vrednost[8])
plt.text(5.1, vrednost[9], "$E_9$=$%.2f $" %vrednost[9])
plt.plot(x,lastne_1,'-', label= "n=1" )
plt.plot(x,lastne_2,'-', label= "n=2" )
plt.plot(x,lastne_3,'-', label= "n=3" )
plt.plot(x,lastne_4,'-', label= "n=4" )
plt.plot(x,lastne_5,'-', label= "n=5" )
plt.plot(x,lastne_6,'-', label= "n=6" )
plt.plot(x,lastne_7,'-', label= "n=7" )
plt.plot(x,lastne_8,'-', label= "n=8" )
plt.plot(x,lastne_9,'-', label= "n=9" )
plt.plot(q,meja,'-',color="black", label= "$5/2*q_2+1/10*q_4$" )
plt.title('Lastne funkcije pri $5/2*q_2+1/10*q_4$')
plt.axis([-5,6,-11,3.5])
plt.xlabel("$x$")
plt.ylabel("$E_(n)$")
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#plt.savefig('Lastne vrednosti od lambde (50) in q^2')
plt.show()
