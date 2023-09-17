from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import scipy.signal as sig
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import scipy.optimize as opt
import os
from tqdm import tqdm
import numpy.linalg as lin
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data = THIS_FOLDER+r"\kalman_relative_data.dat"
kontrola = THIS_FOLDER+r"\kalman_cartesian_kontrola.dat"


t = np.genfromtxt(data, usecols = 0)
meritve = np.genfromtxt(data, usecols = (1,2,3,4))
kontr = np.genfromtxt(kontrola, usecols = (1,2,3,4))
c = np.genfromtxt(data, usecols = (1,2,3,4))

sig_xy = 25
sig_a = 0.05
sig_v = 0.01

r = [sig_xy**2, sig_xy**2, max(sig_v*np.abs(meritve[0][2])**2, 1/3.6), max(sig_v*np.abs(meritve[0][3]), 1/3.6)**2 ]

for i in c:
    i[0] = 0
    i[1] = 0

for i in meritve:
    meritve[3] = 0
    meritve[4] = 0

B = np.eye(4)

x_nap = np.empty((len(meritve), 4))
x_nap[0] = meritve[0] 

P_nap = np.empty((len(meritve), 4, 4))
P_nap[0] = np.diag([sig_xy**2, sig_xy**2, max(sig_v*np.sqrt(meritve[0][2]**2 + meritve[0][3]**2), 1/3.6)**2, max(sig_v*np.sqrt(meritve[0][2]**2 + meritve[0][3]**2), 1/3.6)**2 ])

x_pos = np.empty((len(meritve), 4))
x_pos[0] = meritve[0]

P_pos = np.empty((len(meritve), 4, 4))
P_pos[0] = np.diag([sig_xy**2, sig_xy**2, max(sig_v*np.sqrt(meritve[0][2]**2 + meritve[0][3]**2), 1/3.6)**2, max(sig_v*np.sqrt(meritve[0][2]**2 + meritve[0][3]**2), 1/3.6)**2 ])

dt = 1.783
F = np.eye(4)
F[0, 2] = 1*dt
F[1, 3] = 1*dt

H = np.diag([1,1,0,0])

def Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, number):

    x_nap = np.empty((len(meritve), 4))
    x_nap[0] = meritve[0] 

    P_nap = np.empty((len(meritve), 4, 4))
    P_nap[0] = np.diag([sig_xy**2, sig_xy**2, max(sig_v*np.sqrt(meritve[0][2]**2 + meritve[0][3]**2), 1/3.6)**2, max(sig_v*np.sqrt(meritve[0][2]**2 + meritve[0][3]**2), 1/3.6)**2 ])

    x_pos = np.empty((len(meritve), 4))
    x_pos[0] = meritve[0]

    P_pos = np.empty((len(meritve), 4, 4))
    P_pos[0] = np.diag([sig_xy**2, sig_xy**2, max(sig_v*np.sqrt(meritve[0][2]**2 + meritve[0][3]**2), 1/3.6)**2, max(sig_v*np.sqrt(meritve[0][2]**2 + meritve[0][3]**2), 1/3.6)**2 ])

    dt = 1.783
    F = np.eye(4)
    F[0, 2] = 1*dt
    F[1, 3] = 1*dt

    Q_n = np.diag([0,0,sig_a**2*dt**2,sig_a**2*dt**2])
    Q_vv = np.diag([sig_a**2,sig_a**2])

    P_vv = np.eye(2)

    for i in range(1, len(meritve)):
        #napoved glede na fiziko + šum
        B[2][3] = -x_pos[i-1][3]/np.sqrt(x_pos[i-1][2]**2 + x_pos[i-1][3]**2)
        B[2][2] = x_pos[i-1][2]/np.sqrt(x_pos[i-1][2]**2 + x_pos[i-1][3]**2)
        B[3][2] = x_pos[i-1][3]/np.sqrt(x_pos[i-1][2]**2 + x_pos[i-1][3]**2)
        B[3][3] = x_pos[i-1][2]/np.sqrt(x_pos[i-1][2]**2 + x_pos[i-1][3]**2)

        P_vv = P_pos[i-1][2:, 2:]
        B_vv = B[2:, 2:]

        #Q_vv = (Q_vv + lin.multi_dot([x_pos[i-1][2:].T, P_vv, x_pos[i-1][2:].T])/(x_pos[i-1][2]**2+x_pos[i-1][3]**2)**2 * np.outer(np.dot(B_vv, c[i-1][2:]), np.dot(B_vv, c[i-1][2:])))*dt**2
        #Q_n[2:,2:] = Q_vv

        x_nap[i] = np.dot(F, x_pos[i-1]) + np.dot(B,c[i])*dt

        P_nap[i] = lin.multi_dot([F, P_pos[i-1], F.T]) + Q_n

        H = np.diag([0,0,0,0])
        if i % number == 0:
            H[2][2] = 1
            H[3][3] = 1
        if i % number == 0:
            H[0][0] = 1
            H[1][1] = 1


        #kalman aka nove meritve
        R = np.diag([sig_xy**2, sig_xy**2, max(sig_v*np.sqrt(meritve[i][2]**2 + meritve[i][3]**2), 1/3.6)**2, max(sig_v*np.sqrt(meritve[i][2]**2 + meritve[i][3]**2), 1/3.6)**2 ])
        r = [sig_xy**2, sig_xy**2, max(sig_v*np.sqrt(meritve[i][2]**2 + meritve[i][3]**2), 1/3.6)**2, max(sig_v*np.sqrt(meritve[i][2]**2 + meritve[i][3]**2), 1/3.6)**2 ]

        K = lin.multi_dot([P_nap[i], H.T, lin.inv(lin.multi_dot([H, P_nap[i], H.T]) + R)])

        x_pos[i] = x_nap[i] + np.dot(K, (np.dot(H, meritve[i])  - np.dot(H, x_nap[i]))) #Pazi!!! meritve brez šuma??

        P_pos[i] = np.dot(np.eye(4) - np.dot(K,H), P_nap[i])
    return x_pos, P_pos
number = 1
x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, number)
plt.figure(1)
plt.plot(x_pos.T[0], x_pos.T[1],ls = '--', color = 'r', label = 'filter, relativen pospešek')
plt.plot(kontr.T[0], kontr.T[1],ls = '--', color = 'k', label = 'kontrola')
plt.scatter(meritve.T[0], meritve.T[1], s=2, color = 'g', label = 'meritve')
plt.legend()
plt.xlabel('x [m]')
plt.ylabel('y [m]')


plt.figure(2)

plt.plot(t, x_pos.T[2],ls = '--', color = 'r', label = 'filter, rel')
plt.plot(t, kontr.T[2],ls = '--', color = 'k', label = 'kontrola')
plt.xlabel('t [s]')
plt.ylabel(r'$v_x$ [$\frac{m}{s}$]')
plt.legend()



plt.figure(3)
plt.plot(t, x_pos.T[3],ls = '--', color = 'r', label = 'filter, rel')
plt.plot(t, kontr.T[3],ls = '--', color = 'k', label = 'kontrola')
plt.xlabel('t [s]')
plt.ylabel(r'$v_y$ [$\frac{m}{s}$]')
plt.legend()

#
plt.figure(4)
plt.plot(t, x_pos.T[0]-kontr.T[0], color = 'r', label = 'filter, rel')
plt.plot(t, np.sqrt([i[0, 0] for i in P_pos]),ls = '-', color = 'k', label = r'$\sigma_x$, rel')
plt.plot(t, -np.sqrt([i[0, 0] for i in P_pos]),ls = '-', color = 'k')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel(r'x-$x_k$ [m]')
pod = 0
for i in range(len(meritve)):
    if np.abs(x_pos[i][0] - kontr[i][0]) <= np.sqrt(P_pos[i][0][0]):
        pod = pod + 1
plt.title(str(np.round(pod/len(meritve)*100, 2))+r'% napovedi je med $\pm \sigma$')


plt.figure(5)
plt.plot(t, x_pos.T[1]-kontr.T[1], color = 'r', label = 'filter, rel')
plt.plot(t, np.sqrt([i[1, 1] for i in P_pos]),ls = '-', color = 'k', label = r'$\sigma_y$, rel')
plt.plot(t, -np.sqrt([i[1, 1] for i in P_pos]),ls = '-', color = 'k')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel(r'y-$y_k$ [m]')
pod = 0
for i in range(len(meritve)):
    if np.abs(x_pos[i][1] - kontr[i][1]) <= np.sqrt(P_pos[i][1][1]):
        pod = pod + 1
plt.title(str(np.round(pod/len(meritve)*100, 2))+r'% napovedi je med $\pm \sigma$')


plt.figure(6)
plt.plot(t, x_pos.T[2]-kontr.T[2], color = 'r', label = 'filter, rel')
plt.plot(t, np.sqrt([i[2, 2] for i in P_pos]),ls = '-', color = 'k', label = r'$\sigma_{v_x}$, rel')
plt.plot(t, -np.sqrt([i[2, 2] for i in P_pos]),ls = '-', color = 'k')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel(r'$v_x$-$v_{x,k}$ [$\frac{m}{s}$]')
pod = 0
for i in range(len(meritve)):
    if np.abs(x_pos[i][2] - kontr[i][2]) <= np.sqrt(P_pos[i][2][2]):
        pod = pod + 1
plt.title(str(np.round(pod/len(meritve)*100, 2))+r'% napovedi je med $\pm \sigma$')



plt.figure(7)
plt.plot(t, x_pos.T[3]-kontr.T[3], color = 'r', label = 'filter, rel')
plt.plot(t, np.sqrt([i[3, 3] for i in P_pos]),ls = '-', color = 'k', label = r'$\sigma_{v_y}$, rel')
plt.plot(t, -np.sqrt([i[3, 3] for i in P_pos]),ls = '-', color = 'k')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel(r'$v_y$-$v_{y,k}$ [$\frac{m}{s}$]')
pod = 0
for i in range(len(meritve)):
    if np.abs(x_pos[i][3] - kontr[i][3]) <= np.sqrt(P_pos[i][3][3]):
        pod = pod + 1
plt.title(str(np.round(pod/len(meritve)*100, 2))+r'% napovedi je med $\pm \sigma$')


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data = THIS_FOLDER+r"\kalman_cartesian_data.dat"
kontrola = THIS_FOLDER+r"\kalman_cartesian_kontrola.dat"


t = np.genfromtxt(data, usecols = 0)
meritve = np.genfromtxt(data, usecols = (1,2,3,4))
kontr = np.genfromtxt(kontrola, usecols = (1,2,3,4))
c = np.genfromtxt(data, usecols = (3,4,5,6))

sig_xy = 25
sig_a = 0.05
sig_v = 0.01

r = [sig_xy**2, sig_xy**2, max(sig_v*np.abs(meritve[0][2])**2, 1/3.6), max(sig_v*np.abs(meritve[0][3]), 1/3.6)**2 ]

for i in c:
    i[0] = 0
    i[1] = 0


x_nap = np.empty((len(meritve), 4))
x_nap[0] = meritve[0] 

P_nap = np.empty((len(meritve), 4, 4))
P_nap[0] = np.diag([sig_xy**2, sig_xy**2, max(sig_v*np.sqrt(meritve[0][2]**2 + meritve[0][3]**2), 1/3.6)**2, max(sig_v*np.sqrt(meritve[0][2]**2 + meritve[0][3]**2), 1/3.6)**2 ])

x_pos = np.empty((len(meritve), 4))
x_pos[0] = meritve[0]

P_pos = np.empty((len(meritve), 4, 4))
P_pos[0] = np.diag([sig_xy**2, sig_xy**2, max(sig_v*np.sqrt(meritve[0][2]**2 + meritve[0][3]**2), 1/3.6)**2, max(sig_v*np.sqrt(meritve[0][2]**2 + meritve[0][3]**2), 1/3.6)**2 ])

dt = 1.783
F = np.eye(4)
F[0, 2] = 1*dt
F[1, 3] = 1*dt





Q_n = np.diag([0,0,sig_a**2*dt**2,sig_a**2*dt**2])


for i in range(1, len(meritve)):
    #napoved glede na fiziko + šum
    x_nap[i] = np.dot(F, x_pos[i-1]) + c[i]*dt

    P_nap[i] = lin.multi_dot([F, P_pos[i-1], F.T]) + Q_n
    
    H = np.diag([1,1,1,1])
    


    #kalman aka nove meritve
    R = np.diag([sig_xy**2, sig_xy**2, max(sig_v*np.sqrt(meritve[i][2]**2 + meritve[i][3]**2), 1/3.6)**2, max(sig_v*np.sqrt(meritve[i][2]**2 + meritve[i][3]**2), 1/3.6)**2 ])
    r = [sig_xy**2, sig_xy**2, max(sig_v*np.sqrt(meritve[i][2]**2 + meritve[i][3]**2), 1/3.6)**2, max(sig_v*np.sqrt(meritve[i][2]**2 + meritve[i][3]**2), 1/3.6)**2 ]
    
    K = lin.multi_dot([P_nap[i], H.T, lin.inv(lin.multi_dot([H, P_nap[i], H.T]) + R)])

    x_pos[i] = x_nap[i] + np.dot(K, (np.dot(H, meritve[i])  - np.dot(H, x_nap[i]))) #Pazi!!! meritve brez šuma??

    P_pos[i] = np.dot(np.eye(4) - np.dot(K,H), P_nap[i])

plt.figure(1)
plt.plot(x_pos.T[0], x_pos.T[1],ls = '--', color = 'b', label = 'filter, aboluten pospešek')
plt.legend()
plt.xlabel('x [m]')
plt.ylabel('y [m]')


plt.figure(2)
plt.plot(t, x_pos.T[2],ls = '--', color = 'b', label = 'filter, abs')

plt.xlabel('t [s]')
plt.ylabel(r'$v_x$ [$\frac{m}{s}$]')
plt.legend()


plt.figure(3)
plt.plot(t, x_pos.T[3],ls = '--', color = 'b', label = 'filter, abs')

plt.xlabel('t [s]')
plt.ylabel(r'$v_y$ [$\frac{m}{s}$]')
plt.legend()

#


plt.figure(4)

plt.plot(t, x_pos.T[0]-kontr.T[0], color = 'b', label = 'filter, abs')
plt.plot(t, np.sqrt([i[0, 0] for i in P_pos]),ls = '-', color = 'g', label = r'$\sigma_x$, abs')
plt.plot(t, -np.sqrt([i[0, 0] for i in P_pos]),ls = '-', color = 'g')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel(r'x-$x_k$ [m]')


plt.figure(5)

plt.plot(t, x_pos.T[1]-kontr.T[1], color = 'b', label = 'filter, abs')
plt.plot(t, np.sqrt([i[1, 1] for i in P_pos]),ls = '-', color = 'g', label = r'$\sigma_y$, abs')
plt.plot(t, -np.sqrt([i[1, 1] for i in P_pos]),ls = '-', color = 'g')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel(r'y-$y_k$ [m]')



plt.figure(6)
plt.plot(t, x_pos.T[2]-kontr.T[2], color = 'b', label = 'filter, abs')
plt.plot(t, np.sqrt([i[2, 2] for i in P_pos]),ls = '-', color = 'g', label = r'$\sigma_{v_x}$, abs')
plt.plot(t, -np.sqrt([i[2, 2] for i in P_pos]),ls = '-', color = 'g')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel(r'$v_x$-$v_{x,k}$ [$\frac{m}{s}$]')



plt.figure(7)
plt.plot(t, x_pos.T[3]-kontr.T[3], color = 'b', label = 'filter, abs')
plt.plot(t, np.sqrt([i[3, 3] for i in P_pos]),ls = '-', color = 'g', label = r'$\sigma_{v_y}$, abs')
plt.plot(t, -np.sqrt([i[3, 3] for i in P_pos]),ls = '-', color = 'g')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel(r'$v_y$-$v_{y,k}$ [$\frac{m}{s}$]')



#plt.plot(t, [i[0, 0] for i in P_pos],ls = '-', color = 'r', label = r'$\sigma_x^2$')
#plt.plot(t, [i[1, 1] for i in P_pos],ls = '-', color = 'b', label = r'$\sigma_y^2$')
#plt.xlabel('t [s]')
#plt.ylabel(r'$\sigma^2$ [$m^2$]')
#plt.legend()
#
#
#plt.plot(t, [i[2, 2] for i in P_pos],ls = '-', color = 'r', label = r'$\sigma_{v_x}^2$')
#plt.plot(t, [i[3, 3] for i in P_pos],ls = '-', color = 'b', label = r'$\sigma_{v_y}^2$')
#plt.xlabel('t [s]')
#plt.ylabel(r'$\sigma_v^2$ [$\frac{m^2}{s^2}$]')
#plt.legend()
#
plt.show()
#meritve[vrstica = kdaj smo izmerili][stolpec = kaj smo izmerili (x, y, vx, vy)]

x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, number)
plt.subplot(2,1,1)
plt.title("Prikaz gibanja GPS signala po x,y prostoru, dodan pospešek")
plt.plot(x_pos.T[0], x_pos.T[1],ls = '--', color = 'r', label = 'Kalman')
#plt.scatter(x_pos[348][0], x_pos[348][1], s=100, color = 'b')
#plt.scatter(x_pos[1289][0], x_pos[1289][1], s=100, color = 'b')
plt.plot(kontr.T[0], kontr.T[1],ls = '--', color = 'g', label = 'Kontrolna')
plt.scatter(meritve.T[0], meritve.T[1], s=2, color = 'b', label = 'Meritve')
plt.legend()
plt.grid()

plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.subplot(2,2,3)
plt.title("Hitrost v x smeri")
plt.plot(t, x_pos.T[2],ls = '--', color = 'r', label = 'Kalman')
plt.plot(t, kontr.T[2],ls = '--', color = 'g', label = 'Kontrola')
plt.scatter(t, meritve.T[2], s = 2, color = 'b', label = 'Meritve')
plt.xlabel('t [s]')
plt.ylabel(r'$v_x$ [$\frac{m}{s}$]')
plt.grid()
plt.legend()
plt.subplot(2,2,4)
plt.title("Hitrost v y smeri")
plt.plot(t, x_pos.T[3],ls = '--', color = 'r', label = 'Kalman')
plt.plot(t, kontr.T[3],ls = '--', color = 'g', label = 'Kontrola')
plt.scatter(t, meritve.T[3], s = 2, color = 'b', label = 'Meritve')
plt.xlabel('t [s]')
plt.ylabel(r'$v_y$ [$\frac{m}{s}$]')
plt.legend()
plt.grid()
plt.show()



number = 1
plt.subplot(2,1,1)
plt.title("Prikaz gibanja GPS signala po x,y prostoru, dodan pospešek")
x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, 1)
plt.plot(x_pos.T[0], x_pos.T[1],ls = '-.', color = 'r', label = 'Kalman: vsaka meritev')
x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, 5)
plt.plot(x_pos.T[0], x_pos.T[1],ls = '--', color = 'g', label = 'Kalman: 5')
x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, 100)
plt.plot(x_pos.T[0], x_pos.T[1], '-x', color = 'b', label = 'Kalman: 100')
x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, 1000)
plt.plot(x_pos.T[0], x_pos.T[1],'-', color = 'black', label = 'Kalman: 1000')
plt.legend()
plt.grid()
plt.subplot(2,2,4)
plt.title("Hitrost v y smeri")
x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, 1)
plt.plot(t, x_pos.T[3],ls = '-.', color = 'r', label = 'Kalman: vsaka meritev')
x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, 5)
plt.plot(t, x_pos.T[3],ls = '--', color = 'g', label = 'Kalman: 5')
x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, 100)
plt.plot(t, x_pos.T[3],'-x', color = 'b', label = 'Kalman: 100')
x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, 1000)
plt.plot(t, x_pos.T[3],'-', color = 'black', label = 'Kalman: 1000')
plt.xlabel('t [s]')
plt.ylabel(r'$v_y$ [$\frac{m}{s}$]')
plt.legend()
plt.grid()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.subplot(2,2,3)
plt.title("Hitrost v x smeri")
x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, 1)
plt.plot(t, x_pos.T[2],ls = '-.', color = 'r', label = 'Kalman : vsaka meritev')
x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, 5)
plt.plot(t, x_pos.T[2],ls = '--', color = 'g', label = 'Kalman : 5')
x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, 100)
plt.plot(t, x_pos.T[2], '-x', color = 'b', label = 'Kalman : 100')
x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, 1000)
plt.plot(t, x_pos.T[2],'-', color = 'black', label = 'Kalman : 1000')
plt.xlabel('t [s]')
plt.ylabel(r'$v_x$ [$\frac{m}{s}$]')
plt.grid()
plt.legend()
plt.show()

x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, number)
plt.subplot(2,1,1)
plt.title("Prikaz gibanja GPS signala po x,y prostoru, dodan pospešek")
plt.plot(x_pos.T[0], x_pos.T[1],ls = '--', color = 'r', label = 'Kalman')
#plt.scatter(x_pos[348][0], x_pos[348][1], s=100, color = 'b')
#plt.scatter(x_pos[1289][0], x_pos[1289][1], s=100, color = 'b')
plt.plot(kontr.T[0], kontr.T[1],ls = '--', color = 'g', label = 'Kontrolna')
plt.scatter(meritve.T[0], meritve.T[1], s=2, color = 'b', label = 'Meritve')
plt.legend()
plt.grid()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.subplot(2,2,3)
plt.title("Hitrost v x smeri")
plt.plot(t, x_pos.T[2],ls = '--', color = 'r', label = 'Kalman')
plt.plot(t, kontr.T[2],ls = '--', color = 'g', label = 'Kontrola')
plt.scatter(t, meritve.T[2], s = 2, color = 'b', label = 'Meritve')
plt.xlabel('t [s]')
plt.ylabel(r'$v_x$ [$\frac{m}{s}$]')
plt.grid()
plt.legend()
plt.subplot(2,2,4)
plt.title("Hitrost v y smeri")
plt.plot(t, x_pos.T[3],ls = '--', color = 'r', label = 'Kalman')
plt.plot(t, kontr.T[3],ls = '--', color = 'g', label = 'Kontrola')
plt.scatter(t, meritve.T[3], s = 2, color = 'b', label = 'Meritve')
plt.xlabel('t [s]')
plt.ylabel(r'$v_y$ [$\frac{m}{s}$]')
plt.legend()
plt.grid()
plt.show()

number = 1
draw_all="True"
H_matrix = [np.diag([1,1,1,1]),  np.diag([1,1,0,0]),  np.diag([0,0,1,1]), np.diag([0,0,0,0])]
if draw_all=="True":

    title=["Hitrosti in pozicije", "Pozicija", "Hitrosti", "Nič"]
    index=0
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    fig1, axs1 = plt.subplots(2, 2, constrained_layout=True)
    fig2, axs2 = plt.subplots(2, 2, constrained_layout=True)
    fig3, axs3 = plt.subplots(2, 2, constrained_layout=True)
    fig4, axs4 = plt.subplots(2, 2, constrained_layout=True)
    for i in range(len(H_matrix)):
        j=0    
        H=H_matrix[i]
        if i==2:
            j=1
            i=0
        if i==3:
            j=1
            i=1
        axs[j,i].set_title(title[index])
        fig.suptitle("Prikaz ocen gibanja GPS pri različnih merjenih vrednostih, dodan pospešek")
        x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, number)
        axs[j,i].plot(x_pos.T[0], x_pos.T[1],ls = '--', color = 'r', label = 'Kalman')
        axs[j,i].plot(kontr.T[0], kontr.T[1],ls = '--', color = 'g', label = 'Kontrolna')
        axs[j,i].scatter(meritve.T[0], meritve.T[1], s=2, color = 'b', label = 'Meritve')
        axs[j,i].legend()
        axs[j,i].grid()
        axs[j,i].set_xlabel('x [m]')
        axs[j,i].set_ylabel('y [m]')

        axs1[j,i].set_title(title[index])
        fig1.suptitle("Prikaz odstopanj $x$ GPS pri različnih merjenih vrednosti, dodan pospešekh")
        axs1[j,i].plot(t, x_pos.T[0]-kontr.T[0],alpha = 0.7, color = 'r', label="Kalman")
        axs1[j,i].plot(t,meritve.T[0]-kontr.T[0], alpha=0.3,color = 'b', label="Meritev")
        axs1[j,i].plot(t, np.sqrt([i[0, 0] for i in P_pos]),ls = '--', color = 'black', label = r'$\sigma_x$')
        axs1[j,i].plot(t, -np.sqrt([i[0, 0] for i in P_pos]),ls = '--', color = 'black', label = r'$\sigma_x$')
        axs1[j,i].legend()
        axs1[j,i].grid()
        axs1[j,i].set_xlabel(r't [s]')
        axs1[j,i].set_ylabel(r'$\Delta x$ [m]')

        axs2[j,i].set_title(title[index])
        fig2.suptitle("Prikaz odstopanj $v_x$ GPS pri različnih merjenih vrednostih, dodan pospešek")
        axs2[j,i].plot(t, x_pos.T[2]-kontr.T[2],alpha = 0.7, color = 'r', label = "Kalman")
        axs2[j,i].plot(t, meritve.T[2]-kontr.T[2], alpha=0.3, color = 'b', label = "Meritev")
        axs2[j,i].plot(t, np.sqrt([i[2, 2] for i in P_pos]),ls = '--', color = 'black', label = r'$\sigma_{v_x}$')
        axs2[j,i].plot(t, -np.sqrt([i[2, 2] for i in P_pos]),ls = '--', color = 'black')
        axs2[j,i].legend()
        axs2[j,i].grid()
        axs2[j,i].set_xlabel(r'$t$ [s]')
        axs2[j,i].set_ylabel(r'$\Delta v_x$ [m/s]')

        axs3[j,i].set_title(title[index])
        fig3.suptitle("Prikaz odstopanj $y$ GPS pri različnih merjenih vrednostih, dodan pospešek")
        axs3[j,i].plot(t, x_pos.T[1]-kontr.T[1],alpha = 0.7, color = 'r', label="Kalman")
        axs3[j,i].plot(t,meritve.T[1]-kontr.T[1], alpha=0.3,color = 'b', label="Meritev")
        axs3[j,i].plot(t, np.sqrt([i[1, 1] for i in P_pos]),ls = '--', color = 'black', label = r'$\sigma_y$')
        axs3[j,i].plot(t, -np.sqrt([i[1, 1] for i in P_pos]),ls = '--', color = "black")
        axs3[j,i].legend()
        axs3[j,i].grid()
        axs3[j,i].set_xlabel(r't [s]')
        axs3[j,i].set_ylabel(r'$\Delta y$ [m]')

        axs4[j,i].set_title(title[index])
        fig4.suptitle("Prikaz odstopanj $v_x$ GPS pri različnih merjenih vrednostih, dodan pospešek")
        axs4[j,i].plot(t, x_pos.T[3]-kontr.T[3], alpha = 0.7, color = 'r', label = "Kalman")
        axs4[j,i].plot(t, meritve.T[3]-kontr.T[3],alpha=0.3, color = 'b', label = "Meritev")
        axs4[j,i].plot(t, np.sqrt([i[3, 3] for i in P_pos]),ls = '--', color = 'black', label = r'$\sigma_{v_y}$ ')
        axs4[j,i].plot(t, -np.sqrt([i[3, 3] for i in P_pos]),ls = '--', color = 'black')
        axs4[j,i].legend()
        axs4[j,i].grid()
        axs4[j,i].set_xlabel(r'$t$ [s]')
        axs4[j,i].set_ylabel(r'$\Delta v_y$ [m/s]')
        index+=1
    plt.show()


number = 1
draw_all="True"
number_list = [1,10,100,500]
if draw_all=="True":
    title=["Vsaka 1", "Vsaka 5", "Vsaka 10", "Vsaka 1000"]
    index=0
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    fig1, axs1 = plt.subplots(2, 2, constrained_layout=True)
    fig2, axs2 = plt.subplots(2, 2, constrained_layout=True)
    fig3, axs3 = plt.subplots(2, 2, constrained_layout=True)
    fig4, axs4 = plt.subplots(2, 2, constrained_layout=True)
    for i in range(len(number_list)):
        j=0
        if i==2:
            j=1
            i=0
        if i==3:
            j=1
            i=1
        axs[j,i].set_title(title[index])
        print(number_list[index])
        fig.suptitle("Prikaz ocen gibanja GPS pri različnih merjenih vrednostih, dodan pospešek")
        x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, number_list[index])
        axs[j,i].plot(x_pos.T[0], x_pos.T[1],ls = '--', color = 'r', label = 'Kalman')
        axs[j,i].plot(kontr.T[0], kontr.T[1],ls = '--', color = 'g', label = 'Kontrolna')
        axs[j,i].scatter(meritve.T[0], meritve.T[1], s=2, color = 'b', label = 'Meritve')
        axs[j,i].legend()
        axs[j,i].grid()
        axs[j,i].set_xlabel('x [m]')
        axs[j,i].set_ylabel('y [m]')

        axs1[j,i].set_title(title[index])
        fig1.suptitle("Prikaz odstopanj $x$ GPS pri različnih merjenih vrednostih, dodan pospešek")
        axs1[j,i].plot(t, x_pos.T[0]-kontr.T[0],alpha = 0.7, color = 'r', label="Kalman")
        axs1[j,i].plot(t,meritve.T[0]-kontr.T[0], alpha=0.3,color = 'b', label="Meritev")
        axs1[j,i].plot(t, np.sqrt([i[0, 0] for i in P_pos]),ls = '--', color = 'black', label = r'$\sigma_x$')
        axs1[j,i].plot(t, -np.sqrt([i[0, 0] for i in P_pos]),ls = '--', color = 'black', label = r'$\sigma_x$')
        axs1[j,i].legend()
        axs1[j,i].grid()
        axs1[j,i].set_xlabel(r't [s]')
        axs1[j,i].set_ylabel(r'$\Delta x$ [m]')

        axs2[j,i].set_title(title[index])
        fig2.suptitle("Prikaz odstopanj $v_x$ GPS pri različnih merjenih vrednostih, dodan pospešek")
        axs2[j,i].plot(t, x_pos.T[2]-kontr.T[2],alpha = 0.7, color = 'r', label = "Kalman")
        axs2[j,i].plot(t, meritve.T[2]-kontr.T[2], alpha=0.3, color = 'b', label = "Meritev")
        axs2[j,i].plot(t, np.sqrt([i[2, 2] for i in P_pos]),ls = '--', color = 'black', label = r'$\sigma_{v_x}$')
        axs2[j,i].plot(t, -np.sqrt([i[2, 2] for i in P_pos]),ls = '--', color = 'black')
        axs2[j,i].legend()
        axs2[j,i].grid()
        axs2[j,i].set_xlabel(r'$t$ [s]')
        axs2[j,i].set_ylabel(r'$\Delta v_x$ [m/s]')

        axs3[j,i].set_title(title[index])
        fig3.suptitle("Prikaz odstopanj $y$ GPS pri različnih merjenih vrednostih, dodan pospešek")
        axs3[j,i].plot(t, x_pos.T[1]-kontr.T[1],alpha = 0.7, color = 'r', label="Kalman")
        axs3[j,i].plot(t,meritve.T[1]-kontr.T[1], alpha=0.3,color = 'b', label="Meritev")
        axs3[j,i].plot(t, np.sqrt([i[1, 1] for i in P_pos]),ls = '--', color = 'black', label = r'$\sigma_y$')
        axs3[j,i].plot(t, -np.sqrt([i[1, 1] for i in P_pos]),ls = '--', color = "black")
        axs3[j,i].legend()
        axs3[j,i].grid()
        axs3[j,i].set_xlabel(r't [s]')
        axs3[j,i].set_ylabel(r'$\Delta y$ [m]')

        axs4[j,i].set_title(title[index])
        fig4.suptitle("Prikaz odstopanj $v_y$ GPS pri različnih merjenih vrednostih, dodan pospešek")
        axs4[j,i].plot(t, x_pos.T[3]-kontr.T[3], alpha = 0.7, color = 'r', label = "Kalman")
        axs4[j,i].plot(t, meritve.T[3]-kontr.T[3],alpha=0.3, color = 'b', label = "Meritev")
        axs4[j,i].plot(t, np.sqrt([i[3, 3] for i in P_pos]),ls = '--', color = 'black', label = r'$\sigma_{v_y}$ ')
        axs4[j,i].plot(t, -np.sqrt([i[3, 3] for i in P_pos]),ls = '--', color = 'black')
        axs4[j,i].legend()
        axs4[j,i].grid()
        axs4[j,i].set_xlabel(r'$t$ [s]')
        axs4[j,i].set_ylabel(r'$\Delta v_y$ [m/s]')
        index+=1
    plt.show()

y_label_list=[r'$\Delta x$ [m]', r'$\Delta y$ [m]', r'$\Delta v_x$ [m/s]', r'$\Delta v_y$ [m/s]']
y_sigma_label_list=[r'$\sigma_x$ [m]', r'$\sigma_y$ [m]', r'$\sigma_{v_{x}}$ [m/s]', r'$\sigma_{v_{y}}$ [m/s]']
H_matrix = [np.diag([1,1,1,1]),  np.diag([1,1,0,0]),  np.diag([0,0,1,1])]
label_list=["Hitrosti in pozicije", "Pozicija", "Hitrosti"]
fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle("Prikaz odstopanj vrednosti GPS pri izbiri različnih merjenih vrednostih, dodan pospešek")
fig1, axs1 = plt.subplots(2, 2, constrained_layout=True)
fig1.suptitle("Prikaz $\sigma$ signala GPS pri izbiri različnih merjenih vrednostih, dodan pospešek")
index=0
jndex=0
for i in range(4):
    if i==2:
        jndex=1
        index=0
    if i==3:
        jndex=1
        index=1
    j=0
    for j in range(len(H_matrix)):
            H=H_matrix[j]
            axs[jndex,index].set_title(y_label_list[i])
            x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, number)
            axs[jndex,index].plot(t, x_pos.T[i]-kontr.T[i],alpha = 0.7, color = plt.cm.brg(j/(len(H_matrix)-1)), label=label_list[j])
            axs[jndex,index].legend()
            axs[jndex,index].grid()
            axs[jndex,index].set_xlabel("$t$ [s]")
            axs[jndex,index].set_ylabel(y_label_list[i])

            axs1[jndex,index].set_title(y_sigma_label_list[i])
            x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, number)
            axs1[jndex,index].plot(t, np.sqrt([p[i, i] for p in P_pos]),alpha = 0.7, color = plt.cm.brg(j/(len(H_matrix)-1)), label=label_list[j])
            axs1[jndex,index].legend()
            axs1[jndex,index].grid()
            axs1[jndex,index].set_xlabel("$t$ [s]")
            axs1[jndex,index].set_ylabel(y_sigma_label_list[i])
    index+=1
plt.show()


y_label_list=[r'$\Delta x$ [m]', r'$\Delta y$ [m]', r'$\Delta v_x$ [m/s]', r'$\Delta v_y$ [m/s]']
y_sigma_label_list=[r'$\sigma_x$ [m]', r'$\sigma_y$ [m]', r'$\sigma_{v_{x}}$ [m/s]', r'$\sigma_{v_{y}}$ [m/s]']
number_list = [1,10,100,500]
label_list=["Vsaka 1", "Vsaka 5", "Vsaka 10", "Vsaka 1000"]
fig, axs = plt.subplots(2, 2, constrained_layout=True)
fig.suptitle("Prikaz odstopanj vrednosti GPS pri izbiri različnih merjenih vrednostih, dodan pospešek")
fig1, axs1 = plt.subplots(2, 2, constrained_layout=True)
fig1.suptitle("Prikaz $\sigma$ signala GPS pri izbiri različnih merjenih vrednostih, dodan pospešek")
index=0
jndex=0
for i in range(4):
    if i==2:
        jndex=1
        index=0
    if i==3:
        jndex=1
        index=1
    j=0
    for j in range(len(number_list)):
            axs[jndex,index].set_title(y_label_list[i])
            x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, number_list[j])
            axs[jndex,index].plot(t, x_pos.T[i]-kontr.T[i],alpha = 0.7, color = plt.cm.brg(j/(len(number_list)-1)), label=label_list[j])
            axs[jndex,index].legend()
            axs[jndex,index].grid()
            axs[jndex,index].set_xlabel("$t$ [s]")
            axs[jndex,index].set_ylabel(y_label_list[i])

            axs1[jndex,index].set_title(y_sigma_label_list[i])
            x_pos,P_pos=Kalmanov(meritve, kontr, c, r, sig_xy,sig_a,sig_v, H, number_list[j])
            axs1[jndex,index].plot(t, np.sqrt([p[i, i] for p in P_pos]),alpha = 0.7, color = plt.cm.brg(j/(len(number_list)-1)), label=label_list[j])
            axs1[jndex,index].legend()
            axs1[jndex,index].grid()
            axs1[jndex,index].set_xlabel("$t$ [s]")
            axs1[jndex,index].set_ylabel(y_sigma_label_list[i])
    index+=1
plt.show()
