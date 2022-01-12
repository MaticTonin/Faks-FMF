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
D = 1e-3
sigma = 0.2
a=1
N=200
t_max=100
E=1
def Gauss(D,a,sigma,x):
    return D*np.sin(x*np.pi/a)*np.exp(-(x - a/2)**2. / sigma**2.)+ 4*D*np.cos(x*np.pi/a)*np.exp(-(x - a/6)**2. / sigma**2.)

N_X=800
N_T=800
x=np.linspace(0, a, N_X)
t=np.linspace(0, t_max, N_T)
h_x=x[1]-x[0]
h_t=t[1]-t[0]
#Periodični
gauss=Gauss(E,a,sigma,x)
f_k=np.fft.fftfreq(N_T,h_x)
T0K=np.fft.fft(gauss)
T = np.empty((N_T, N_X), dtype=complex)

def analit_res(L, D, sigma, A, x, t, p):
    T = len(x) * [0.]
    func = lambda x_i, n_i, L_i, sigma_i, A_i: np.sin(n_i * np.pi * x_i / L_i) *\
                                          A_i * np.exp(-(x_i - L_i/2.)**2. / sigma_i**2.)
    for n in range(200):
        B_integral = integrate.quad(func, 0., 1., args=(n, L, sigma, A))
        B_n = 2. / L * B_integral[0]
        # print B_integral[0]
        s = B_n * np.sin(n * np.pi * x / L) * np.exp(-D * (n * np.pi / L)**2. * t)
        T += s
        if(np.amax(s) < p and n > 1):
            break
    return T
    
T_analit = np.zeros((N_T, N_X))
p = 1e-25
for i in range(N_T):
    T_analit[i, :] = analit_res(a, D, sigma, E, x, t[i], p)
print(T_analit)

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
ax.set_title(r"Prikaz različnih enačb za Fourierovo metodo pri D="+str(D))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$T(x)$')
axcb = fig.colorbar(lc)
axcb.set_label(r'$t$')
#plt.savefig('Fourier_barviti_graf_0.pdf')
plt.show()
plt.close()

x_gra = np.array([x for i in range(N_T)])
fig, ax = plt.subplots()
segments = [np.column_stack([x1, y]) for x1, y in zip(x_gra[0:], abs(TF[0:, :N_X].real)-T_analit[0:, :N_X])]
lc = LineCollection(segments, cmap='gnuplot')
lc.set_array(np.asarray(t))
ax.add_collection(lc)
ax.autoscale()
ax.set_title(r"Napake periodične za Fourierovo metodo pri D="+str(D))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$T(x)$')
axcb = fig.colorbar(lc)
axcb.set_label(r'$t$')
#plt.savefig('Fourier_barviti_graf_0.pdf')
plt.show()
plt.close()

print("{:.2f}".format(t[10]))
plt.plot(x, TF[0,:N_T].real, '-',color="Navy", label="t={:.2f}".format(t[0]))
plt.plot(x, TF[10,:N_T].real, '-',color="Darkblue",label="t={:.2f}".format(t[10]))
plt.plot(x, TF[20,:N_T].real, '-',color="MediumBlue",label="t={:.2f}".format(t[20]))
plt.plot(x, TF[50,:N_T].real, '-',color="Blue",label="t={:.2f}".format(t[50]))
plt.plot(x, TF[100,:N_T].real, '-',color="SkyBlue",label="t={:.2f}".format(t[100]))
plt.plot(x, TF[200,:N_T].real, '-',color="LightSalmon",label="t={:.2f}".format(t[200]))
plt.plot(x, TF[500,:N_T].real, '-',color="Salmon",label="t={:.2f}".format(t[500]))
#plt.plot(x, TF[700,:N_T].real, '-',color="Crimson",label="t={:.2f}".format(t[700]))
plt.plot(x, TF[N_X-1,:N_T].real, '-',color="Red",label="t={:.2f}".format(t[N_X-1]))
plt.xlabel("x")
plt.ylabel("T[K]")
plt.title(r"Prikaz različnih enačb za Fourierovo metodo pri D="+str(D))
plt.legend()
plt.show()


X,Y=meshgrid(x,t)
Z=np.zeros((N_T,N_X))
for i in range(N_T):
    for j in range(N_X):
        Z[i,j]=TF[i,j]
ax = plt.axes(projection='3d')
ax.contour3D(X, Y , Z, N_X, cmap='gnuplot')
ax.set_title(r"Prikaz različnih enačb za Fourierovo metodo pri D="+str(D))
ax.set_xlabel('X')
ax.set_ylabel('t')
ax.set_zlabel('T')
plt.show()


# Dirichletovi robni pogoji
gauss = Gauss(E, a, sigma, x)
gauss[0], gauss[-1] = [0.,0.]
gauss = np.append(gauss, -gauss[:0:-1])
print(gauss)
T = np.empty((N_T, 2 * N_X - 1), dtype=complex)
f_k=np.fft.fftfreq(len(T[0, :]),h_x)
T0K=np.fft.fft(gauss)
xa=np.append(x,-x[:0:-1] )
plt.plot(xa, gauss, '-',color="Navy", label="t={:.2f}".format(t[0]))
plt.xlabel("x")
plt.ylabel("T[K]")
plt.title(r"Gauss D="+str(D))
plt.legend()
plt.show()
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
ax.set_title(r"Prikaz različnih enačb enačbe za Fourierovo metodo pri D="+str(D))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$T(x)$')
axcb = fig.colorbar(lc)
axcb.set_label(r'$t$')
plt.savefig('Fourier_barviti_graf_0.pdf')
plt.show()
plt.close()

x_gra = np.array([x for i in range(N_T)])
fig, ax = plt.subplots()
segments = [np.column_stack([x1, y]) for x1, y in zip(x_gra[2:], abs(TF[2:, :N_X].real)-T_analit[2:, :N_X])]
lc = LineCollection(segments, cmap='gnuplot')
lc.set_array(np.asarray(t))
ax.add_collection(lc)
ax.autoscale()
ax.set_title(r"Napake različnih enačb enačbe za Fourierovo metodo pri D="+str(D))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$T(x)$')
axcb = fig.colorbar(lc)
axcb.set_label(r'$t$')
#plt.savefig('Fourier_barviti_graf_0.pdf')
plt.show()
plt.close()

print("{:.2f}".format(t[10]))
plt.plot(x, TF[0,:N_T].real, '-',color="Navy", label="t={:.2f}".format(t[0]))
plt.plot(x, TF[10,:N_T].real, '-',color="Darkblue",label="t={:.2f}".format(t[10]))
plt.plot(x, TF[20,:N_T].real, '-',color="MediumBlue",label="t={:.2f}".format(t[20]))
plt.plot(x, TF[50,:N_T].real, '-',color="Blue",label="t={:.2f}".format(t[50]))
plt.plot(x, TF[100,:N_T].real, '-',color="SkyBlue",label="t={:.2f}".format(t[100]))
plt.plot(x, TF[200,:N_T].real, '-',color="LightSalmon",label="t={:.2f}".format(t[200]))
plt.plot(x, TF[500,:N_T].real, '-',color="Salmon",label="t={:.2f}".format(t[500]))
#plt.plot(x, TF[700,:N_T].real, '-',color="Crimson",label="t={:.2f}".format(t[700]))
#plt.plot(x, TF[N_X-1,:N_T].real, '-',color="Red",label="t={:.2f}".format(t[N_X-1]))
plt.xlabel("x")
plt.ylabel("T[K]")
plt.title(r"Prikaz različnih enačb za Fourierovo metodo pri D="+str(D))
plt.legend()
plt.show()


X,Y=meshgrid(x,t)
Z=np.zeros((N_T,N_X))
for i in range(N_T):
    for j in range(N_X):
        Z[i,j]=TF[i,j]
ax = plt.axes(projection='3d')
ax.contour3D(X, Y , Z, N_X, cmap='gnuplot')
ax.set_title(r"Prikaz različnih enačb za Fourierovo metodo pri D="+str(D))
ax.set_xlabel('X')
ax.set_ylabel('t')
ax.set_zlabel('T')
plt.show()