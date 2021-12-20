import numpy as np
from numpy.lib.function_base import meshgrid
import scipy.special as spec
import matplotlib.pyplot as plt
import time
import numpy as np
from mpl_toolkits import mplot3d
from numpy import fft
from scipy import linalg
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
D = 1e-4
sigma = 0.2
a=1
N=200
t_max=1000
E=1
def Gauss(D,a,sigma,x):
    return D*np.exp(-(x - a/2)**2. / sigma**2.)

N_X=700
N_T=700
x=np.linspace(0, a, N_X)
t=np.linspace(0, t_max, N_T)
h_x=x[1]-x[0]
h_t=t[1]-t[0]
#Periodični
gauss=Gauss(E,a,sigma,x)
f_k=np.fft.fftfreq(N_T,h_x)
T0K=np.fft.fft(gauss)
T = np.empty((N_T, N_X), dtype=complex)
# Dirichletovi robni pogoji
"""gauss = Gauss(C, a, sigma, x)
gauss[0], gauss[-1] = [0.,0.]
gauss = np.append(gauss, -gauss[:0:-1])
print(gauss)
T = np.empty((N_T, 2 * N_X - 1), dtype=complex)
f_k=np.fft.fftfreq(len(T[0, :]),h_x)
T0K=np.fft.fft(gauss)"""

def Fourierova(t,T,f_k):
    N_T=len(T[0,:])
    N_X=len(T[:,0])
    for i in range(N_T):
        T[:,i]=np.exp(-4*D*np.pi**2*f_k[i]**2*t)*T0K[i]
        """h_t=10**(-8)
        T[:,i]=T0K[i]
        T0K[i]*=( 1 - 4 * h_t * D * np.pi**2 * (i/a)**2)*t
        #T[:,i]=T[:,i-1]+h_t*D*(-4*np.pi**2*f_k[i]**2)*T[:,i-1]
        T[:,i]=np.array((1-4*h_t*D*np.pi**2*f_k[i]**2))*T0K[i]
        T0K[i]*=( 1 - 4 * h_t * D * np.pi**2 * (i/a)**2)"""
    for j in range(N_X):
        T[j,:]=np.fft.ifft(T[j,:])
    return T
TF=Fourierova(t,T,f_k)

x_gra = np.array([x for i in range(N_T)])
fig, ax = plt.subplots()
segments = [np.column_stack([x1, y]) for x1, y in zip(x_gra, TF[:, :N_X].real)]
lc = LineCollection(segments, cmap='gnuplot')
lc.set_array(np.asarray(t))
ax.add_collection(lc)
ax.autoscale()
ax.set_title(r"Prikaz periodične difucijske enačbe za Fourierovo metodo pri a="+str(a))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$T(x)$')
axcb = fig.colorbar(lc)
axcb.set_label(r'$t$')
plt.savefig('Fourier_barviti_graf_0.pdf')
plt.show()
plt.close()

print("{:.2f}".format(t[10]))
plt.plot(x, TF[0,:N_T].real, '-',color="Navy", label="t={:.2f}".format(t[0]))
"""plt.plot(x, TF[10,:N_T].real, '-',color="Darkblue",label="t={:.2f}".format(t[10]))
plt.plot(x, TF[20,:N_T].real, '-',color="MediumBlue",label="t={:.2f}".format(t[20]))
plt.plot(x, TF[50,:N_T].real, '-',color="Blue",label="t={:.2f}".format(t[50]))
plt.plot(x, TF[100,:N_T].real, '-',color="SkyBlue",label="t={:.2f}".format(t[100]))
plt.plot(x, TF[200,:N_T].real, '-',color="LightSalmon",label="t={:.2f}".format(t[200]))
plt.plot(x, TF[500,:N_T].real, '-',color="Salmon",label="t={:.2f}".format(t[500]))"""
#plt.plot(x, TF[700,:N_T].real, '-',color="Crimson",label="t={:.2f}".format(t[700]))
#plt.plot(x, TF[N_X-1,:N_T].real, '-',color="Red",label="t={:.2f}".format(t[N_X-1]))
plt.xlabel("x")
plt.ylabel("T[K]")
plt.title(r"Prikaz periodične difucijske enačbe za Fourierovo metodo pri a="+str(a))
plt.legend()
plt.show()



def matrix_AB(h_x,N):
    A=np.zeros((N,N))
    B=np.zeros((N,N))
    for i in range(N):
        B[i,i]=-2
        B[i,i-1]=1
        B[i-1,i]=1
        B[N-1,0]=0
        B[0,N-1]=0
        A[i,i]=4
        A[i,i-1]=1
        A[i-1,i]=1
        A[N-1,0]=0
        A[0,N-1]=0
    B=6*D/(h_x**2)*B
    return A,B

A,B=matrix_AB(h_x,N_X)


def zlepki_B(x,k,h_x,x_0):
    if x_0<=x[k-2]:
        return 0
    if x[k-2]<=x_0<=x[k-1]:
        return 1/(h_x**3)*(x_0-x[k-2])**3
    if x[k-1]<=x_0<=x[k]:
        return 1/(h_x**3)*(x_0-x[k-2])**3-4/(h_x**3)*(x_0-x[k-1])**3
    if x[k]<=x_0<=x[k+1]:
        return 1/(h_x**3)*(x[k+2]-x_0)**3-4/(h_x**3)*(x[k+1]-x_0)**3
    if x[k+1]<=x_0<=x[k+2]:
        return 1/(h_x**3)*(x[k+2]-x_0)**3
    if x_0>=x[k+2]:
        return 0

def kubicni_B_zlepki(delta, k, x0):
    # k = -1,...,n+1
    if(x0 <= delta*(k-2)):
        return 0.
    elif(x0 <= delta*(k-1)):
        return (x0 - (k-2)*delta)**3. / (6. * delta**3.)
    elif(x0 <= delta*k):
        return 1 / 6. + (x0 - delta*(k-1)) / (2. * delta) + (x0 - delta*(k-1))**2. / (2. * delta**2.) -\
               (x0 - delta*(k-1))**3. / (2. * delta**3.)
    elif(x0 <= delta*(k+1)):
        return 1 / 6. - (x0 - delta*(k+1)) / (2. * delta) + (x0 - delta*(k+1))**2. / (2. * delta**2.) +\
               (x0 - delta*(k+1))**3. / (2. * delta**3.)
    elif(x0 <= delta*(k+2)):
        return -(x0 - delta*(k+2))**3. / (6. * delta**3.)
    else:
        return 0.

def koef_c(f_0,A,B,h_t):
    N=len(A[0,:])
    C=[]
    vsota=A+h_t/2*B
    minus=A-h_t/2*B
    c=np.linalg.solve(A,f_0)
    for i in range(N):
        D=np.dot(vsota,c)
        c=np.linalg.solve(minus,D)
        C.append(c)
    return np.asarray(C)

plt.plot(x, koef_c(gauss, A, B, h_t)[0,:], '-',color="Blue",label="t={:.2f}".format(t[N_X-1]))
plt.xlabel("x")
plt.ylabel("T[K]")
plt.title(r" a="+str(a))
plt.legend()
plt.show()

def Kolokacija(C,x,t,h_x):
    N_X=len(x)
    N_T=len(t)
    T=np.zeros((N_X,N_T))
    for i in range(N_X):
        for j in range(N_T):
            vsoa=0
            for k in range(2,N_X+2):
                vsoa+=C[i,k-2]*kubicni_B_zlepki(h_x,k,x[j])
            T[i,j]=vsoa
    return T
A,B=matrix_AB(h_x,N_X)
C=koef_c(gauss, A, B, h_t)
TK=Kolokacija(C,x,t,h_x)
print(TK)


x=np.linspace(0, a, N_X)
t=np.linspace(0, 100, N_T)
h_x=x[1]-x[0]
h_t=t[1]-t[0]
#Periodični
gauss=Gauss(E,a,sigma,x)
f_k=np.fft.fftfreq(N_T,h_x)
T0K=np.fft.fft(gauss)
T = np.empty((N_T, N_X), dtype=complex)
TA=Fourierova(t,T,f_k)
X,Y=meshgrid(x,t)
Z=np.zeros((N_T,N_X))
for i in range(N_T):
    for j in range(N_X):
        Z[i,j]=TF[i,j]
ax = plt.axes(projection='3d')
ax.contour3D(X, Y , Z, N_X, cmap='hot')
ax.set_title(r"Prikaz periodične difucijske enačbe za Fourierovo metodo pri a="+str(a))
ax.set_xlabel('X')
ax.set_ylabel('t')
ax.set_zlabel('T')
plt.show()

X,Y=meshgrid(x,t)
Z=np.zeros((N_T,N_X))
for i in range(N_T):
    for j in range(N_X):
        Z[i,j]=TK[i,j]
ax = plt.axes(projection='3d')
ax.contour3D(X, Y , Z, N_X, cmap='hot')
ax.set_title(r"Prikaz periodične difucijske enačbe za Kolokacijkso metodo pri a="+str(a))
ax.set_xlabel('X')
ax.set_ylabel('t')
ax.set_zlabel('T')
plt.show()

# Dirichletovi robni pogoji
gauss = Gauss(E, a, sigma, x)
gauss[0], gauss[-1] = [0.,0.]
gauss = np.append(gauss, -gauss[:0:-1])
T = np.empty((N_T, 2 * N_X - 1), dtype=complex)
f_k=np.fft.fftfreq(len(T[0, :]),h_x)
T0K=np.fft.fft(gauss)
