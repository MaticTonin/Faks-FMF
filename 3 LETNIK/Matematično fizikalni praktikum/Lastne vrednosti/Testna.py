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
                 q_4[i,j]=q_4[i,j]+(1/2)**4*np.sqrt(16*(i-3)*(i-2)*(i-1)*i)
             if i==j+2:
                 q_4[i,j]=q_4[i,j]+(1/2)**4*np.sqrt(4*i*(i-1))*(4*(2*j+3))
             if i==j:
                 q_4[i,j]=q_4[i,j]+(1/2)**4*np.sqrt((2**i * factorial(i))/(2**j * factorial(j)))*(12*(2*j**2+2*j+1))
             if i==j-2:
                 q_4[i,j]=q_4[i,j]+(1/2)**4*np.sqrt((2**i * factorial(i))/(2**j * factorial(j)))*(16*(2*j**2-3*j+1))
             if i==j-4:
                 q_4[i,j]=q_4[i,j]+(1/2)**4*np.sqrt((2**i * factorial(i))/(2**j * factorial(j)))*(16*j*(j**3-6*j**2+11*j-6))

     H=H_0+lambd*q_4
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
    V=identity(N)
    while i<K:
        for j in range(N):
         for k in range(N):
                if j!=k and abs(H[j,k])>eps:
                    P=Householder1(H,N)[1]
                    B=np.matmul(P,H)
                    H=np.matmul(B,P)
                    V=np.matmul(P,V)
                    V=np.matmul(V,P)
                    i+=1
        #if      N-i==0:
            #B.append(R[N-2,N-2]
    vrednost=H[0,0]
    return H, vrednost, V

def postopek(N, lambd, eps, K):
    a=1
    R=matrika_H2(N,lambd)
    i=0
    H=np.zeros((N,N))
    vrednost=[]
    lastni1=np.zeros((N,N))
    lastni=np.zeros((N,N))
    if N==2 or N==1: 
        for i in range(N):
            for j in range(N):
                if i==j:
                    vrednost.append(R[i,j])
                    lastni[i,j]=R[i,j]
        H=R
    else: 
        while i<N-2:
            R=iteracija_na_stolpcu(R, N-i, eps, K)[0]
            vrednost.append(iteracija_na_stolpcu(R,N-i, eps, K)[1])
            lastni1=iteracija_na_stolpcu(R,N-i, eps, K)[2]
            for j in range(len(R)):
             for k in range(len(R)):
                 H[j+i,k+i]=R[j,k]
                 lastni[j+i,k+i]=lastni1[j,k]
            R=R[1:N,1:N]
            i+=1
        vrednost.append(H[N-2,N-2])
        vrednost.append(H[N-1,N-1])
    return H, vrednost, lastni

x=np.linspace(-3,3,1000)
def lastne_za_x(N, x, n, lastni):
    funkcija=0
    for i in range(N):
        funkcija= funkcija+ (2. ** i * np.math.factorial(i) * np.pi ** 0.5) ** (-0.5) *np.exp(-x**(2)/2) * special.hermite(i,0)(x) * (lastni[i,n]) 
        #print(funkcija)
    return funkcija