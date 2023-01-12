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

p=1
q=1
A1=Ai(xfp[0], tfp[0],p,q)

A = np.zeros((len(xfp),len(A1)))
for i in range(0, len(xfp)):
    for j in range(0, len(A1)):
        A[i][j] = Ai(xfp[i], tfp[i],p,q)[j]
print(A)
plt.title(r"Prikaz podatkov magnetnega spektrometra v ravnini ($x_{fp}$, $\theta_{tg}$)")
plt.scatter(tfp,xfp,c=tg, cmap='rainbow')
plt.colorbar(label=r"$\theta_{tg}$")
plt.ylabel(r"$x_{fp}$")
plt.xlabel(r"$\theta_{fg}$")
plt.legend()
plt.grid()
plt.show()

plt.title(r"Prikaz podatkov magnetnega spektrometra v ravnini ($\theta_{fp}$, $\theta_{tg}$)")
plt.scatter(tfp,tg,c=xfp, cmap='rainbow')
plt.colorbar(label=r"$x_{fp}$")
plt.ylabel(r"$\theta_{tg}$")
plt.xlabel(r"$\theta_{fp}$")
plt.legend()
plt.grid()
plt.show()


print(xfp)
ax = plt.axes(projection='3d')
ax.set_title(r"Prikaz podatkov magnetnega spektrometra v 3D prostor")
ax.scatter3D(xfp, tfp, tg, c=tg, cmap='rainbow')
ax.set_xlabel(r"$x_{fp}$")
ax.set_ylabel(r"$\theta_{fp}$")
ax.set_zlabel(r"$\theta_{tg}$")
plt.show()


u, s, vh = np.linalg.svd(A, full_matrices=False)
a =  np.zeros(shape=(len(A1),1))
a = 0
for i in range(0, len(A1)):
    a = a + (np.dot(np.transpose(u[:,i]),tg))/s[i]*vh[i]

chi = 0
for i in range(0, len(tg)):
    zi = np.dot(Ai(xfp[i], tfp[i],p,q),a)
    chi = chi + ((zi - tg[i])/0.003)**2
print(chi)

fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
zi_line=[]
for i in range(len(tg)):
    zi = np.dot(Ai(xfp[i], tfp[i],p,q),a)
    zi_line.append(zi)
    #ax.scatter(tfp[i], zi, color = 'b')
    #ax1.scatter(xfp[i], zi, color = 'b')

ax.scatter(tfp[0], np.dot(A1,a), color = 'b', label = 'Model za $p=%.2f,q=%.2f$' %(p,q))
ax1.scatter(xfp[0], np.dot(A1,a), color = 'b', label = 'Model za $p=%.2f,q=%.2f$' %(p,q))
ax.set_title(r"Primerjava modela in podatkov v ravnini ($\theta_{fp}$, $\theta_{tg}$)")
ax1.set_title(r"Primerjava modela in podatkov v ravnini ($x_{fp}$, $\theta_{tg}$)")
ax.errorbar(tfp, tg, yerr = 0.001, xerr = 0.001,  fmt ='.r', capsize=3, label='Meritve')
ax1.errorbar(xfp, tg, yerr = 0.001, xerr = 1,  fmt ='.r', capsize=3, label='Meritve')
ax.set_xlabel(r'$\theta_{fp}$')
ax.set_ylabel(r'$\theta_{tg}$')
ax1.set_xlabel(r'$\theta_{fp}$')
ax1.set_ylabel(r'$\theta_{tg}$')
ax1.legend()
ax.legend()
plt.show()

ax = plt.axes(projection='3d')
ax.set_title(r"Prikaz podatkov in meritev magnetnega spektrometra v 3D prostor")
ax.scatter3D(xfp, tfp, zi_line, c=tg, cmap='summer', label="Model")
ax.scatter3D(xfp, tfp, tg, c=tg, cmap='cool', label="Podatki")
ax.legend()
ax.set_xlabel(r"$x_{fp}$")
ax.set_ylabel(r"$\theta_{fp}$")
ax.set_zlabel(r"$\theta_{tg}$")
plt.show()
"""
XFP,TFP=np.meshgrid(xfp, tfp)
z_matrix=[]
for i in range(len(tg)):
    z_matrix.append(abs(tg-zi_line))
z_matrix=np.array(z_matrix)
ax = plt.axes(projection='3d')
ax.set_title(r"Odstopanje magnetnega spektrometra v 3D prostor")
ax.plot_surface(XFP, TFP, z_matrix, cmap='cool')
ax.set_xlabel(r"$x_{fp}$")
ax.set_ylabel(r"$\theta_{fp}$")
ax.set_zlabel(r"$\theta_{tg}$")
plt.show()"""

p_line=[3,4,5,6,7,8,9,10]
q_line=[3,4,5,6,7,8,9,10]

p_line=[10,9,8,7,6,5,4,3,2]
q_line=[10,9,8,7,6,5,4,3,2]
p_line=[7,6,5,4,3,2]
q_line=[7,6,5,4,3,2]
p_line=[2,3,4,5,6,7]
q_line=[2,3,4,5,6,7]
p_line=[2,3,4,5,6,7,8,9,10,11,12,13,14,15]
q_line=[2,3,4,5,6,7,8,9,10,11,12,13,14,15]
import time
start_time = time.time()
timer=np.zeros((len(p_line),len(q_line)))
chi_line=np.zeros((len(p_line),len(q_line)))
error_line=np.zeros((len(p_line),len(q_line)))
index=0
jndex=0
t_tot=0
for i in tqdm(p_line):
    jndex=0
    for j in tqdm(q_line):
        start = time.time()
        chi=0
        A1=Ai(xfp[0], tfp[0],i,j)
        A = np.zeros((len(xfp),len(A1)))
        k=0
        l=0
        for k in range(0, len(xfp)):
            for l in range(0, len(A1)):
                A[k][l] = Ai(xfp[k], tfp[k],i,j)[l]
        u, s, vh = np.linalg.svd(A, full_matrices=False)
        a =  np.zeros(shape=(len(A1),1))
        a = 0
        l=0
        for l in range(0, len(A1)):
            a = a + (np.dot(np.transpose(u[:,l]),tg))/s[l]*vh[l]
        zi=0
        error=0
        chi=0
        k=0
        for k in range(0, len(tg)):
            zi = np.dot(Ai(xfp[k], tfp[k],i,j),a)
            chi = chi + ((zi - tg[k])/0.003)**2
            error+=abs(tg[k]-zi)
        end= time.time()
        error_line[index,jndex]=error
        chi_line[index,jndex]=chi
        timer[index,jndex]=end-start
        t_tot+=end-start
        jndex+=1
    index+=1
index=0
p_line, q_line = np.meshgrid(p_line, q_line)
ax = plt.axes(projection='3d')
ax.set_title(r"Prikaz časovne zahtevnosti od izbire $p$, $q$, $t_{tot}=%.2f s$" %(t_tot))
surf=ax.plot_surface(p_line, q_line, timer, cmap='rainbow')
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$q$")
ax.set_zlabel(r"$t[s]$")
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.legend()
plt.show()


sns.heatmap(np.log10(chi_line),cmap="Greens",cbar_kws={'label': r'$log(\chi^2)$'}, annot=True)
plt.show()

ax = plt.axes(projection='3d')
ax.set_title(r"Prikaz $\chi^2$ od izbire $p$, $q$")
ax.plot_surface(p_line, q_line, chi_line, cmap='rainbow')
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$q$")
ax.set_zlabel(r"$\chi^2$")
ax.legend()
plt.show()


ax = plt.axes(projection='3d')
ax.set_title(r"Prikaz vsote odstopanj odstopanj od izbire $p$, $q$")
ax.plot_surface(p_line, q_line, error_line, cmap='rainbow')
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$q$")
ax.set_zlabel(r"error")
ax.legend()
plt.show()


ax = plt.axes(projection='3d')
ax.set_title(r"Prikaz $\chi^2$ od izbire $p$, $q$ v log skali")
ax.plot_surface(p_line, q_line, chi_line, cmap='rainbow')
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$q$")
ax.set_zlabel(r"$\chi^2$")
ax.set_zscale("log")
ax.legend()
plt.show()

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


#----------------------------------------------------

def var(j,vh):
    v = 0
    for i in range(0, len(vh[0])):
        v = v + vh[i][j]**2/s[i]**2
    return v

def cov(j, k,vh):
    co = 0
    for i in range(0, len(vh[0])):
        co = co + vh[i][j]*vh[i][k]/s[i]**2
    return co

def kor(j,k,vh):
    return cov(j,k,vh)/np.sqrt(var(j)*var(k))




K = np.zeros(shape=(len(vh[0]),len(vh[0])))
for j in range(0, len(vh[0])):
    for k in range(0, len(vh[0])):
        K[j][k] = kor(j,k,vh)

for j in range(0, len(K[0])):
    sum = -1
    for i in range(0, len(K[0])):
        sum = sum + K[j][i]
    print(j, sum)


sns.heatmap(K, cmap="bwr", annot=True)
plt.show()




x = np.linspace(-100, 100, 1000)
t = np.linspace(-3.5, 1000, 10000)
