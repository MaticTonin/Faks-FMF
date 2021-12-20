import numpy as np
from numpy.lib.function_base import meshgrid
import matplotlib.pyplot as plt
import time
from mpl_toolkits import mplot3d
from scipy import linalg
from scipy import integrate
from scipy.sparse import block_diag
from scipy.sparse.linalg import spsolve
from matplotlib.collections import LineCollection
from scipy.special import beta as B
np.set_printoptions(linewidth=np.inf)


def trial_func(m, n, r, phi):
    return r**(2*m + 1) * (1-r)**n * np.sin((2*m + 1) * phi)

def Aij(m, nmax):
    Aij_temp = np.zeros((nmax, nmax))
    for i in range(nmax):
        for j in range(i+1):
            Aij_temp[i][j] = -(np.pi/2.)*(i+1.)*(j+1.)*(3.+4.*m)*B(i+j+1.,3.+4.*m)/(2.+4.*m+i+j+2.)
            if(j != i): Aij_temp[j][i] = Aij_temp[i][j]
    return Aij_temp

def podmatrixA(m,n):
    pod_A=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            pod_A[i,j]= -(np.pi/2.)*(i+1.)*(j+1.)*(3.+4.*m)*B(i+j+1.,3.+4.*m)/(2.+4.*m+i+j+2.)
    return pod_A

"""print(podmatrixA(10,10))
print(Aij(10,10))
print(podmatrixA(10,10)-Aij(10,10))"""

def podvektorB(m,n):
    pod_b=np.zeros(n)
    for i in range(n):
        nindex= i+1
        pod_b[i]=-2.*B(2*m+3., nindex+1.)/(2.*m +1.)
    return pod_b

def bj(m, nmax):
    bj_temp = np.zeros(nmax)
    for i in range(nmax):
        n = i+1
        bj_temp[i] = -2.*B(2*m+3., n+1.)/(2.*m +1.)
    return bj_temp
"""print(bj(4,4))
print(podvektorB(4,4))
print(podvektorB(4,4)-bj(4,4))"""

def Garlekin(maxm, maxn):
    a=np.zeros((maxm+1)*maxn)
    b=np.zeros((maxm+1)*maxn)
    A_niz=[]
    C=0
    for i in range(maxm+1):
        b[i*(maxn):(i+1)*(maxn)]=podvektorB(i,maxn)
        A_niz.append(podmatrixA(i,maxn))
    A=block_diag(A_niz,format='csr')
    a=spsolve(A, b)
    C=-(32./np.pi)*b.dot(a)
    return C, a

def Garlekin_test(maxm, maxn):
    a=np.zeros((maxm+1)*maxn)
    b=np.zeros((maxm+1)*maxn)
    A_niz=[]
    C=0
    for i in range(maxm+1):
        b[i*(maxn):(i+1)*(maxn)]=podvektorB(i,maxn)
        a[i*(maxn):(i+1)*(maxn)]=spsolve(podmatrixA(i,maxn), podvektorB(i,maxn))
    C=-(32./np.pi)*b.dot(a)
    return C, a
print(Garlekin(3,3)[0])
print(Garlekin_test(3,3)[0])
N=100
r = np.linspace(0, 1, N)
phi = np.linspace(0, np.pi, N)
R, PHI = np.meshgrid(r, phi) #za plt.countourf
X = R * np.cos(PHI)
Y = R * np.sin(PHI)
Z = np.zeros((N, N))

#Plotanje rešitve priporočam m=3, n=3
m=3
n=3
C, a = Garlekin(m, n)
for i in range(m+1):
    for j in range(n):
        print(i, j)
        for k in range(N):
            for l in range(N):
                Z[k][l] += a[i*(n)+j]*trial_func(i, j+1, R[k][l], PHI[k][l])
ax = plt.axes(projection='3d')
ax.contour3D(X, Y , Z, N, cmap='jet')
ax.set_title(r"Prikaz pretoka skozi polovično cev")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel(r"$\Phi$")
plt.show()



m=3
n=3
C, a = Garlekin(m, n)
for i in range(m+1):
    for j in range(n):
        print(i, j)
        for k in range(N):
            for l in range(N):
                Z[k][l] += a[i*(n)+j]*trial_func(i, j+1, R[k][l], PHI[k][l])

plt.contourf(X, Y, Z, np.linspace(0, 0.105, 500), cmap='jet')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Prikaz pretoka skozi polovično cev')
plt.show()

#Časovna zahtevnost v odvisnosti od izbire 
"""mcas=0
ncas=50
delcki=[]
skupno=[]
MCAS=[]
while mcas<50:
    start1=time.time()
    Garlekin_test(mcas,ncas)
    end1 = time.time()
    delcki.append(end1-start1)
    
    start2=time.time()
    Garlekin(mcas,ncas)
    end2 = time.time()
    skupno.append(end2-start2)
    MCAS.append(mcas)
    mcas+=1
plt.plot(MCAS,skupno, '-',label="Reševanje sistema")
plt.plot(MCAS,delcki, '-',label="Reševanje podsistemov")
plt.xlabel("M")
plt.ylabel("t")
plt.title(r"Prikaz časovne odvisnosti od izbire reševanja sistema pri $n=50$")
plt.legend()
plt.show()

ncas=0
mcas=50
delcki=[]
skupno=[]
MCAS=[]
while ncas<50:
    start1=time.time()
    Garlekin_test(mcas,ncas)
    end1 = time.time()
    delcki.append(end1-start1)
    
    start2=time.time()
    Garlekin(mcas,ncas)
    end2 = time.time()
    skupno.append(end2-start2)
    MCAS.append(ncas)
    ncas+=1
    print(ncas)
plt.plot(MCAS,skupno, '-',label="Reševanje sistema")
plt.plot(MCAS,delcki, '-',label="Reševanje podsistemov")
plt.xlabel("N")
plt.ylabel("t")
plt.title(r"Prikaz časovne odvisnosti od izbire reševanja sistema pri $m=50$")
plt.legend()
plt.show()"""
#Prikaz odvisnotsi napake od m
m=[0,5,10,20,35,50,70,100, 100]
n=10
#barve = cm.jet([256//10*i for i in range(10)])
for i in range(len(m)):
    C_niz=[]
    N_niz=[]
    for j in range(n):
        N_niz.append(j)
        C_niz.append((np.log(np.abs(C-Garlekin(i,j)[0]))))
    plt.plot(N_niz,C_niz, '-',label="m="+str(m[i]))
    print(i)
plt.xlabel("N")
plt.ylabel(r"$\log(C_{ref}-C_m)$")
plt.title(r"Prikaz odstopanj glede na izbiro m in n")
plt.legend()
plt.show()


m1=70
n1=70
d=n1
C_ref, a_ref = Garlekin(10, 10)
print(C_ref)
N1 = np.linspace(0, n1, d)
M1 = np.linspace(0, m1, d)
N,M=meshgrid(N1,M1)
E = np.zeros((d, d))
for i in range(len(N1)):
    for j in range(len(M1)):
        E[i,j]=(Garlekin(i+1,j+1)[0])
    print(i)
print(E)
ax = plt.axes(projection='3d')
ax.contour3D(N, M, E, m1, cmap='jet')
ax.set_title(r"Prikaz vrednosti C v odvisnosti od m in n")
ax.set_xlabel('N')
ax.set_ylabel('M')
ax.set_zlabel(r"$C$")
plt.show()

m1=20
n1=20
C_ref, a_ref = Garlekin(100, 100)
print(C_ref)
N = np.linspace(0, n1, n1)
M = np.linspace(0, m1, m1)
N,M=meshgrid(N,M)
E = np.zeros((m1, n1))
for i in range(len(N)):
    for j in range(len(M)):
        E[j,i]=(np.log(np.abs(C_ref-Garlekin(i,j)[0])))
    print(i)

ax = plt.axes(projection='3d')
ax.contour3D(M, N , E, m1, cmap='jet')
ax.set_title(r"Prikaz napake v odvisnosti od izbire m in n")
ax.set_xlabel('M')
ax.set_ylabel('N')
ax.set_zlabel(r"$\log(C_{ref}-C_m)$")
plt.show()
#Večji N
m1=20
n1=20
C_ref, a_ref = Garlekin(300, 300)
print(C_ref)
N = np.linspace(100,140, m1)
M = np.linspace(100,140, n1)
N1,M1=meshgrid(N,M)
E = np.zeros((m1, n1))
for i in range(len(N)):
    for j in range(len(M)):
        E[j,i]=(np.log(np.abs(C_ref-Garlekin(int(N[i]),int(M[j]))[0])))
    print(i)

ax = plt.axes(projection='3d')
ax.contour3D(M1, N1 , E, m1, cmap='jet')
ax.set_title(r"Prikaz napake v odvisnosti od izbire m in n")
ax.set_xlabel('M')
ax.set_ylabel('N')
ax.set_zlabel(r"$\log(C_{ref}-C_m)$")
plt.show()
mmax = [0,1,2,5,10,20,30,50,75,100]
nmax = 50
C_ref, a_ref = Garlekin(200, 200)

plt.figure()
for i in mmax:
    error = []
    for j in range(1, nmax):
        C = Garlekin(i, j+1)[0]
        print(C)
        error.append(np.abs(C - C_ref))
    #plt.plot(range(1, nmax), error, label='m_{max} = %d' % i)
    plt.semilogy(range(1, nmax), error, label='m_{max} = %d' % i)
plt.legend(loc=(0.75,0.7))
plt.xlabel('$n_{max}$')
plt.ylabel('$|C_{100, 100} - C_{m_{max},n_{max}}|$')
plt.show()
"""
#Prikaz testnih funkcij:

N = 1000
mmax = 3
nmax = 3

r = np.linspace(0, 1, N)
phi = np.linspace(0, np.pi, N)
R, PHI = np.meshgrid(r, phi) #za plt.countourf
X = R * np.cos(PHI)
Y = R * np.sin(PHI)

plt.figure()
for i in range(mmax+1):
    for j in range(1, nmax+1):
        plt.subplot(mmax+1, nmax, i*(nmax)+j)
        print(i*(nmax)+j)
        Z = np.zeros((N, N))
        for k in range(N):
            for l in range(N):
                Z[k][l] = trial_func(i, j, R[k][l], PHI[k][l])
        plt.contourf(X, Y, Z, cmap='jet')
        plt.colorbar()
        #plt.xlabel('x')
        #plt.ylabel('y')
        plt.title('m = ' + str(i) + ', n = ' + str(j))
plt.show()

    """
            


