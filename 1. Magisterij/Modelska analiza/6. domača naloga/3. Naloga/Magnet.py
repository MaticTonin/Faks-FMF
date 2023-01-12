

import fractions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(THIS_FOLDER + "\\thtg-xfp-thfp.dat", dtype=str,delimiter=" ")
empty,tg,xfp,tfp=data.T
tg=np.array(tg,dtype=float)
xfp=np.array(xfp,dtype=float)
tfp=np.array(tfp,dtype=float)
tg=tg*np.pi/180
tfp=tfp*np.pi/180

def Ai(x,t): 
    return [1, x, x**2, x**3, t, t*x, t*x**2, t*x**3, t**2, t**2*x, t**2*x**2, t**2*x**3, t**3, t**3*x, t**3*x**2, t**3*x**3]

A1 = Ai(xfp[0], tfp[0])

A = np.zeros(shape=(len(xfp),len(A1)))



for i in range(0, len(xfp)):
    for j in range(0, len(A1)):
        A[i][j] = Ai(xfp[i], tfp[i])[j]

#ax = plt.axes(projection='3d')
#ax.scatter3D(xfp, tfp, tg, color='r')
#ax.set_xlabel('x_fp')
#ax.set_ylabel('theta_fp')
#ax.set_zlabel('theta_tg')
#
#plt.show()
#

#
#plt.errorbar(tfp, tg, yerr = 0.001, xerr = 0.001,  fmt ='.r', capsize=3, label='meritev')

plt.xlabel('theta_{fp} [rad]')
plt.ylabel('theta_{tg} [rad]')


u, s, vh = np.linalg.svd(A, full_matrices=False)


#print(np.shape(u))
#print(np.shape(s))
#print(np.shape(vh))

a =  np.zeros(shape=(len(A1),1))

a = 0
for i in range(0, len(A1)):
    a = a + (np.dot(np.transpose(u[:,i]),tg))/s[i]*vh[i]
print(a)

#-------------------------------------------------------

#for i in range(1, len(tg)):
#    zi = np.dot(Ai(xfp[i], tfp[i]),a)
#    plt.scatter(tfp[i], zi, color = 'b')
#plt.scatter(tfp[0], np.dot(Ai(xfp[0], tfp[0]),a), color = 'b', label = 'model')
#
#plt.xlabel('theta_{fp} [rad]')
#plt.ylabel('theta_{tg} [rad]')
#plt.legend()
#plt.show()

#-------------------------------------------------------

#for i in range(1, len(tg)):
#    zi = np.dot(Ai(xfp[i], tfp[i]),a)
#    plt.scatter(xfp[i], zi, color = 'b')
#plt.scatter(xfp[0], np.dot(Ai(xfp[0], tfp[0]),a), color = 'b', label = 'model')
#
#plt.errorbar(xfp, tg, yerr = 0.001, xerr = 1,  fmt ='.r', capsize=3, label='meritev')
#
#plt.xlabel('x_{fp} [mm]')
#plt.ylabel('theta_{tg} [rad]')
#plt.legend()
#plt.show()

#------------------------------------------------------
#for j in range(1, 101):
#    l = j/10000
#    stevec = 0
#    for i in range(0, len(tg)):
#        zi = np.dot(Ai(xfp[i], tfp[i]),a)
#        if (np.abs(zi - tg[i]) <= l):
#            stevec = stevec + 1
#    plt.scatter(l, stevec/len(tg), color = 'r')
#plt.hlines(0.666, 0, 0.01)
##    plt.errorbar(i, zi-tg[i],yerr = 0.001, fmt ='.k', capsize=3)
##plt.hlines(0, 0, 11600)
#plt.xlabel('odstopanja')
#plt.ylabel('% meritev znotraj odstopanja')
##plt.legend()
#plt.show()

#print(stevec)
#------------------------------------------------------

chi = 0
for i in range(0, len(tg)):
    zi = np.dot(Ai(xfp[i], tfp[i]),a)
    chi = chi + ((zi - tg[i])/0.003)**2

print(chi)

#----------------------------------------------------

def var(j):
    v = 0
    for i in range(0, len(vh[0])):
        v = v + vh[i][j]**2/s[i]**2
    return v

def cov(j, k):
    co = 0
    for i in range(0, len(vh[0])):
        co = co + vh[i][j]*vh[i][k]/s[i]**2
    return co

def kor(j,k):
    return cov(j,k)/np.sqrt(var(j)*var(k))




K = np.zeros(shape=(len(vh[0]),len(vh[0])))
for j in range(0, len(vh[0])):
    for k in range(0, len(vh[0])):
        K[j][k] = kor(j,k)

for j in range(0, len(K[0])):
    sum = -1
    for i in range(0, len(K[0])):
        sum = sum + K[j][i]
    print(j, sum)


sns.heatmap(K, 
        xticklabels=['a0', 'a1','a2','a3', 'a4','a5','a6', 'a7','a8','a9', 'a10','a11','a12', 'a13','a14','a15'],
        yticklabels=['a0', 'a1','a2','a3', 'a4','a5','a6', 'a7','a8','a9', 'a10','a11','a12', 'a13','a14','a15'], cmap="bwr", annot=True)
plt.show()




x = np.linspace(-100, 100, 1000)
t = np.linspace(-3.5, 1000, 10000)