import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.core.numeric import identity
import scipy
from scipy import special

#ZGRADBA OSNOVNE BAZE

#Zgradba normalne matrike 
def matrika_H1(N, lambd):
    H_0=np.zeros((N, N)) #definiramo matriko H_0
    #Izdelava H_0
    for i in range(N):
        for j in range(N):
            if i==j:
                H_0[i,j]=1/2+i
    #Izdelava matrike \lambda q^4 
    q=np.zeros((N,N))
    for i in range (N):
        for j in range(N):
            if abs(i-j)==1:
                #q[i,j]=2
                q[i,j]= 1/2*(i+j+1)**(1/2)
    #Izdelava potenciala q^4      
    q_2=np.matmul(q,q)
    q_3=np.matmul(q_2,q)
    q_4=lambd * np.matmul(q_3,q)
    #Skupna izdelava matrike H
    H=np.zeros((N,N))
    H=H_0+q_4
    return H
# izdelava matrike H z q^2
def matrika_H2(N, lambd):
     H_0=np.zeros((N, N)) #definiramo matriko H_0
    #Izdelava H_0
     for i in range(N):
         for j in range(N):
            if i==j:
                H_0[i,j]=1/2+i

     q_2=np.zeros((N,N))
     for i in range(N):
         for j in range(N):
             if i==j-2:
                 q_2[i,j]=q_2[i,j]+1/2* np.sqrt(j*(j-1))
             if i==j:
                 q_2[i,j]=q_2[i,j]+1/2*(2*j+1)
             if i==j+2:
                  q_2[i,j]=q_2[i,j]+ 1/2*np.sqrt((j+1)*(j+2))
     q_4=lambd* np.matmul(q_2,q_2)
     H=np.zeros((N,N))
     H=H_0+q_4
     return H

def factorial(n):
    if n ==1 or n==0:
        return 1
    else: return n * factorial(n-1)

def matrika_H4(N, lambd):
     H_0=np.zeros((N, N)) #definiramo matriko H_0
    #Izdelava H_0
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

     H=H_0+lambd*q_4
     return H

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
#print(matrika_H1(5, 1))
#print(matrika_H2(5, 1))
#print(matrika_H4(5,1))
    
#IZDELAVA OMEGE ZA HOUSEHOLDERJEVO ZRCALJENJE
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

def postopek(N, lambd, eps, K):
    a=1
    R=matrika_H4(N,lambd)
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



#Grafi prikazov v odvisnosti od lambda
n=30
N=np.linspace(0,n,n)
print(np.sort(np.linalg.eig(matrika_H1(10,1))[0]))
dejanskeH1=np.sort(np.linalg.eig(matrika_H1(n,1))[0])
dejanskeH2=np.sort(np.linalg.eig(matrika_H2(n,1))[0])
dejanskeH4=np.sort(np.linalg.eig(matrika_H4(n,1))[0])
#householder0=sorted(postopek(50,0.09, 10**(-16), 100)[1],reverse=False)
print(np.sort(np.linalg.eig(matrika_H1(n,0.09))[0]))
#print(householder0)
householder1=sorted(postopek(n,0.1, 10**(-16), 10)[1],reverse=False)
print(householder1)
householder2=sorted(postopek(n,0.2, 10**(-16), 10)[1],reverse=False)
print(householder2)
#householder3=sorted(postopek(50,0.3, 10**(-16), 100)[1],reverse=False)
#print(householder3)
householder4=sorted(postopek(n,0.4, 10**(-16), 10)[1],reverse=False)
#householder5=sorted(postopek(50,0.5, 10**(-16), 100)[1],reverse=False)
print(householder4)
householder6=sorted(postopek(n,0.6, 10**(-16), 10)[1],reverse=False)
#print(householder5)
#householder7=sorted(postopek(50,0.7, 10**(-16), 100)[1],reverse=False)
print(householder6)
householder8=sorted(postopek(n,0.8, 10**(-16), 10)[1],reverse=False)
#print(householder7)
#householder9=sorted(postopek(50,0.9, 10**(-16), 100)[1],reverse=False)
print(householder8)
householder10=sorted(postopek(n,1, 10**(-16), 10)[1],reverse=False)
#print(householder9)

plt.plot(N,dejanskeH4,'o', label= "Dejanske Lastne  $\lambda$=1" )
#plt.plot(N,householder0,'-', label= "Lastne  $\lambda$=0.09" )
plt.plot(N,householder1,'-', label= "Lastne  $\lambda$=0.1" )
plt.plot(N,householder2,'-', label= "Lastne  $\lambda$=0.2" )
#plt.plot(N,householder3,'-', label= "Lastne  $\lambda$=0.3" )
plt.plot(N,householder4,'-', label= "Lastne  $\lambda$=0.4" )
#plt.plot(N,householder5,'-', label= "Lastne  $\lambda$=0.5" )
plt.plot(N,householder6,'-', label= "Lastne $\lambda$=0.6" )
#plt.plot(N,householder7,'-', label= "Lastne  $\lambda$=0.7" )
plt.plot(N,householder8,'-', label= "Lastne  $\lambda$=0.8")
#plt.plot(N,householder9,'-', label= "Lastne  $\lambda$=0.9" )
plt.plot(N,householder10,'-', label= "Lastne  $\lambda$=1" )

plt.title('Odvisnost lastnih energij od lambde pri N=30 in q^4=q^4')
plt.xlabel("$\lambda$")
plt.ylabel("$E_(n)$")
plt.legend()
#plt.savefig('Lastne vrednosti od lambde (50) in q^2')
plt.show()

#Grafi prikazov posameznih metod:
def postopek1(N, lambd, eps, K):
    a=1
    R=matrika_H1(N,lambd)
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

def postopek2(N, lambd, eps, K):
    a=1
    R=matrika_H2(N,lambd)
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
def postopek4(N, lambd, eps, K):
    a=1
    R=matrika_H4(N,lambd)
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
n=80
N=np.linspace(0,n,n)
householder1=sorted(postopek1(n,0.5, 10**(-18), 40)[1],reverse=False)
print(householder1)
householder2=sorted(postopek2(n,0.5, 10**(-18), 40)[1],reverse=False)
print(householder2)
householder4=sorted(postopek4(n,0.5, 10**(-18), 40)[1],reverse=False)

dejanskeH1=np.sort(np.linalg.eig(matrika_H1(n,0.5))[0])
dejanskeH2=np.sort(np.linalg.eig(matrika_H2(n,0.5))[0])
dejanskeH4=np.sort(np.linalg.eig(matrika_H4(n,0.5))[0])
plt.plot(N,householder1,'-', label= "Lastne - metoda q" )
plt.plot(N,householder2,'-', label= "Lastne - metoda q^2" )
plt.plot(N,householder4,'-', label= "Lastne - metoda q^4" )
plt.title('Odvisnost lastnih energij od metode pri $\lambda$=0.5')
plt.xlabel("$N$")
plt.ylabel("$E_(n)$")
plt.legend()
plt.show()

plt.plot(N,dejanskeH1,'-', label= "Lastne denajske- metoda q" )
plt.plot(N,dejanskeH2,'-', label= "Lastne dejanske- metoda q^2" )
plt.plot(N,dejanskeH4,'-', label= "Lastne dejanske- metoda q^4" )
plt.title('Odvisnost lastnih energij od metode pri $\lambda$=0.5, izračunane')
plt.xlabel("$N$")
plt.ylabel("$E_(n)$")
plt.legend()
plt.show()

absolute1=[]
absolute2=[]
absolute4=[]
for i in range(n):
    print(i)
    absolute1.append(np.log(abs(householder1[i]-dejanskeH1[i])))
    absolute2.append(np.log(abs(householder2[i]-dejanskeH2[i])))
    absolute4.append(np.log(abs(householder4[i]-dejanskeH4[i])))
plt.plot(N,absolute1,'-', label= "Dostopanje za q" )
plt.plot(N,absolute2,'-', label= "Dostopanje za q^2" )
plt.plot(N,absolute4,'-', label= "Dostopanje za q^4" )
plt.title('Odvisnost odstopanja  za vse tri metode v log skali')
plt.xlabel("$N$")
plt.ylabel("$\log(E_(n))$")
plt.legend()
plt.show()