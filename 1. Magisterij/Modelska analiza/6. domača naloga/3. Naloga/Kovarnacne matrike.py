
import fractions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
from tqdm import tqdm
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(THIS_FOLDER + "\\thtg-xfp-thfp.dat", dtype=str,delimiter=" ")
empty,tg,xfp,tfp=data.T
tg=np.array(tg,dtype=float)
xfp=np.array(xfp,dtype=float)
tfp=np.array(tfp,dtype=float)
tg=tg*np.pi/180
tfp=tfp*np.pi/180

def Ai(x,t,p,q): 
    A=[]
    for i in range(p+1):
        for j in range(q+1):
                A.append(x**j*t**i)
    return A

p=7
q=5
A1=Ai(xfp[0], tfp[0],p,q)

A = np.zeros((len(xfp),len(A1)))
for i in range(0, len(xfp)):
    for j in range(0, len(A1)):
        A[i][j] = Ai(xfp[i], tfp[i],p,q)[j]

u, s, vh = np.linalg.svd(A, full_matrices=False)
a =  np.zeros(shape=(len(A1),1))
a = 0
for i in range(0, len(A1)):
    a = a + (np.dot(np.transpose(u[:,i]),tg))/s[i]*vh[i]

chi = 0
for i in range(0, len(tg)):
    zi = np.dot(Ai(xfp[i], tfp[i],p,q),a)
    chi = chi + ((zi - tg[i])/0.003)**2

def var(j,vh,s):
    v = 0
    for i in range(0, len(vh[0])):
        v = v + vh[i][j]**2/s[i]**2
    return v

def cov(j, k,vh,s):
    co = 0
    for i in range(0, len(vh[0])):
        co = co + vh[i][j]*vh[i][k]/s[i]**2
    return co

def kor(j,k,vh,s):
    return cov(j,k,vh,s)/np.sqrt(var(j,vh,s)*var(k,vh,s))

def creating_labels(p,q):
    xticklabels=[]
    for i in range(p+1):
        for j in range(q+1):
            if i==0 and j==0:
                xticklabels.append(r"1")
            elif i!=0 and j==0:
                xticklabels.append(r"$x^{%i}$" %(i))
            elif i==0 and j!=0:
                xticklabels.append(r"$\theta^{%i}$" %(j))
            elif i!=0 and j!=0:
                xticklabels.append(r"$x^{%i}\theta^{%i}$" %(i,j))
    return xticklabels

K = np.zeros(shape=(len(vh[0]),len(vh[0])))
for j in range(0, len(vh[0])):
    for k in range(0, len(vh[0])):
        K[j][k] = kor(j,k,vh,s)

xticklabels=creating_labels(p,q)
sns.heatmap(K, xticklabels=xticklabels,yticklabels=xticklabels,cmap="Greens")
plt.show()
