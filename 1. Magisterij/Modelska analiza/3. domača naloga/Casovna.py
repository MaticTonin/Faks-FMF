import numpy as np
import numpy.linalg as lin
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull  
from mpl_toolkits.mplot3d import Axes3D
from random import seed
import time
from random import random
# seed random number generator
seed(1)



def sphere_coord(fi, th):
    xy = np.sin(th)
    x = xy * np.cos(fi)
    y = xy * np.sin(fi)
    z = np.cos(th)

    return np.array([x, y, z])

def dipole_moment(x):
    e1 = np.array([0, 0, 1])
    e2 = sphere_coord(0, x[0])
    return e1 + e2 + np.sum([sphere_coord(x[i], x[i+1]) for i in range(1, len(x)-1, 2)], axis=0)

def quad_moment(x):
    e1 = np.array([0, 0, 1])
    e2 = sphere_coord(0, x[0])
    e = [sphere_coord(x[i], x[i+1]) for i in range(1, len(x)-1, 2)]

    qp = np.outer(e1, e1) - 1/3 * np.eye(3)
    qp += np.outer(e2, e2) - 1/3 * np.eye(3)
    for i in range(len(e)):
        qp += np.outer(e[i], e[i]) - 1/3 * np.eye(3)

    return qp


def plot_sphere(N, ax):
    u = np.linspace(0, 2*np.pi, N)
    v = np.linspace(0, np.pi, N)
    u, v = np.meshgrid(u, v)
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    return x, y, z
    



def izdelava(N,method):
    alpha=0.5
    naboj=[]
    for i in range(N):
        naboj.append(1)
        #naboj.append(random())
    def energy12(fi1, th1, fi2, th2, naboj1, naboj2):
        return naboj1*naboj2 / (2 - 2*np.cos(fi1 - fi2)*np.sin(th1)*np.sin(th2) - 2*np.cos(th1)*np.cos(th2))**(alpha)

    def energy(x):
        #first charge
        fi0 = 0
        th0 = 0
        #second charge
        fi1 = 0
        th1 = x[0]
        #other charges
        fi = x[1::2]
        th = x[2::2]
        #number of free charges
        M = len(fi)
        M = len(fi)
        #energy of first two charges
        E = energy12(fi0, th0, fi1, th1,naboj[0],naboj[1]) + np.sum([energy12(fi0, th0, fi[i], th[i], naboj[0], naboj[i]) for i in range(M)])
        E += np.sum([energy12(fi1, th1, fi[i], th[i],naboj[1],naboj[i]) for i in range(M)])
        #energy of other charges
        E += np.sum([np.sum([energy12(fi[i], th[i], fi[j], th[j],naboj[i],naboj[j]) for i in range(j)]) for j in range(M)])
        return E

    def grad_fi12(fi1, th1, fi2, th2,naboj1,naboj2):
        if fi1 == fi2 and th1 == th2: return 0
        return - np.sin(fi1 - fi2) * np.sin(th1)*np.sin(th2) * energy12(fi1, th1, fi2, th2,naboj1,naboj2)**(1 + 1/alpha)

    def grad_th12(fi1, th1, fi2, th2,naboj1,naboj2):
        if fi1 == fi2 and th1 == th2: return 0
        return  (np.cos(fi1 - fi2) * np.cos(th1)*np.sin(th2) - np.sin(th1) * np.cos(th2)) * energy12(fi1, th1, fi2, th2,naboj1,naboj2)**(1 + 1/alpha)

    def grad(x):
        #first charge
        fi0 = 0
        th0 = 0
        #second charge
        fi1 = 0
        th1 = x[0]
        #other charges
        fi = x[1::2]
        th = x[2::2]
        #number of free charges
        M = len(fi)
        E = energy(x)
        J = np.empty(len(x))
        #grad of second charge
        J[0] = (grad_th12(fi1, th1, fi0, th0,naboj[0],naboj[1]) + np.sum([grad_th12(fi1, th1, fi[i], th[i],naboj[0],naboj[i]) for i in range(M)]))
        #first charge effect
        J[2::2] = np.array([grad_th12(fi[i], th[i], fi0, th0, naboj[i],naboj[0])  for i in range(M)])
        J[1::2] = np.array([grad_fi12(fi[i], th[i], fi0, th0, naboj[i],naboj[0])  for i in range(M)])
        #second charge effect
        J[2::2] += np.array([grad_th12(fi[i], th[i], fi1, th1, naboj[i],naboj[1]) for i in range(M)])
        J[1::2] += np.array([grad_fi12(fi[i], th[i], fi1, th1, naboj[i],naboj[1]) for i in range(M)])
        #other charges effect
        J[2::2] += np.array([np.sum([grad_th12(fi[j], th[j], fi[i], th[i],naboj[j],naboj[i]) for i in range(M)]) for j in range(M)])
        J[1::2] += np.array([np.sum([grad_fi12(fi[j], th[j], fi[i], th[i],naboj[j],naboj[i]) for i in range(M)]) for j in range(M)])

        return J 

    def inital_x(N):
        x0 = np.empty(2*N + 1)
        x0[::2] = np.linspace(0, np.pi, N+2, endpoint=False)[1:]
        x0[1::2] = np.linspace(0, 2*np.pi, N+1, endpoint=False)[1:]

        return x0

    def bounds(N):
        l = np.zeros(2*N + 1)
        u = np.empty(2*N + 1)

        u[::2] = np.array([np.pi for i in range(N+1)])
        u[1::2] = np.array([2*np.pi for i in range(N)])

        return opt.Bounds(l, u)


    #print(inital_x(N))
    #print(grad(inital_x(N)))
    #quit()
    sol = opt.minimize(energy, inital_x(N), bounds=bounds(N), method=method) 
    """,jac=grad"""

    dp = dipole_moment(sol.x)
    energija=energy(sol.x)

    Q = quad_moment(sol.x)
    eigQ, vecQ = lin.eig(Q)

    """ax.quiver(0,0,0,*dp, color="red")
    for i in range(len(eigQ)):
        ax.quiver(0,0,0,*(eigQ[i]*vecQ[:,i]), color="green")"""
    
    r_s = np.concatenate(([0, 0, 0], sol.x))
    r_c = np.array([sphere_coord(r_s[i], r_s[i+1]) for i in range(0, len(r_s), 2)])
    return naboj, r_c, energija

import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data_file = THIS_FOLDER + "\Wiki.txt"
data = np.loadtxt(data_file, encoding="utf8", dtype=str, delimiter="\t")
metode=["BFGS", "Powell","Nelder-Mead","CG","SLSQP"]
metode=["BFGS", "Powell","Nelder-Mead","SLSQP"]
#metode=["BFGS","CG","SLSQP"]
thompson=data.T[1]
"""for i in metode:
    cas=[]
    razlika=[]
    x=[]
    print(i)
    for j in range(2,40):
        x.append(j)
        start = time.time()
        naboj,r_c, energija=izdelava(j, i)
        end = time.time()
        cas.append(end-start)
        razlika.append(np.log(abs(float(thompson.T[j])-energija)))
        print(j)
    plt.plot(x,cas,"x-", label= "Metoda "+str(i))
    plt.title("Prikaz časovne zahtevnosti alguritma brez gradienta")
    plt.xlabel("N")
    plt.ylabel("t[s]")
    plt.legend()
plt.show()"""

for i in metode:
    cas=[]
    razlika=[]
    x=[]
    print(i)
    for j in range(2,40):
        x.append(j)
        start = time.time()
        naboj,r_c, energija=izdelava(j, i)
        end = time.time()
        cas.append(end-start)
        razlika.append(np.log(abs(float(thompson.T[j])-energija)))
        print(j)
    plt.plot(x,razlika,"x-",label= "Metoda "+str(i))
    plt.title("Prikaz odstopanja alguritma brez gradienta")
    plt.xlabel("N")
    plt.ylabel("log(E-E_ref)")
    plt.legend()
    print(j)
plt.show()
