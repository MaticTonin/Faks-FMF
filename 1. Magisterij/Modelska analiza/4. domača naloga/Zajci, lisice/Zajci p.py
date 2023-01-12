from cgi import print_exception
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# function that returns dz/dt

n = 40000 #stevilo korakov integracije
l = 50 #čas do kamor simuliramo
N=10

def model(z,t,p):
    Z = z[0]
    L = z[1]
    dZdt = p*Z*(1-L)
    dLdt = L*(Z-1)/p
    dzdt = [dZdt,dLdt,p]
    return dzdt

Z0 =5
L0 = 5
p0 = 1
z0 = [Z0,L0,p0]
t = np.linspace(0,l,n)
p_values = np.linspace(1,2.6,N)
X, Y = np.meshgrid(t, p_values)


z=[]
for i in range(N):
    z.append(0)

c=0
for i in p_values:
    print(i)
    z[c] = odeint(model,z0,t, args=(i,))
    #plt.plot(t,z[:,0],'b',label='Zajci,  p = %.3f' %(i))
    c = c +1
z = np.array(z)  
print(z[:,:,0]) 
print(z.ndim)
"""

fig, (ax1,ax2) = plt.subplots(2)
fig.suptitle(r"Prikaz obnašanja gozda pri parametru $p$ in začetni populaciji Zajci: $%.2f$, Lisice: $%.2f$" %(Z0,L0))

im1=ax1.imshow(z[:,:,0], extent=[min(t),max(t),0,20],cmap=cm.coolwarm)
axins = inset_axes(ax1,
                    width="5%",  
                    height="100%",
                    loc='center right',
                    borderpad=-5
                   )
plt.colorbar(im1, cax=axins)
y_label_list = np.linspace(1,5,5)
ax1.set_title("Zajci")
ax1.set_xlabel('t')
ax1.set_ylabel('p')
ax1.set_yticks([0,5,10, 15,20])
ax1.set_yticklabels(y_label_list)

im2 = ax2.imshow(z[:,:,1], extent=[min(t),max(t),0,20],cmap=cm.coolwarm)
axins = inset_axes(ax2,
                    width="5%",  
                    height="100%",
                    loc="center right",
                    borderpad=-5
                   )
plt.colorbar(im2, cax=axins)
y_label_list = np.linspace(1,5,5)
ax2.set_title("Lisice")
ax2.set_xlabel('t')
ax2.set_ylabel('p')
ax2.set_yticks([0,5,10, 15,20])
ax2.set_yticklabels(y_label_list)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.axes.set_zlim3d(bottom=0, top=20)
ax.set_xlabel("t")
ax.set_ylabel('p')
ax.set_zlabel('Zajci')
surf = ax.plot_surface(X,Y,z[:,:,0], cmap=cm.coolwarm)
fig.colorbar(surf, shrink=0.5, aspect=5)


# Add a color bar which maps values to colors.

#print(z[0])
#print(z)
#plt.plot(t,z[:,0],'b',label='Zajci,  p = %.3f' %(p[0]))
#plt.plot(t,z[:,1],'r',label='Lisice, L_0 = %.3f' %(L0))
#plt.ylabel('brezdimenzijsko število živali')
#plt.xlabel('čas')
#plt.legend(loc='best')

plt.show()

im = plt.imshow(z[:,:,0], extent=[min(t),max(t),0,10],cmap=cm.coolwarm)
plt.colorbar(im)
x_label_list = [min(p_values), max(p_values)/4, 2*max(p_values)/4, 3*max(p_values)/4, max(p_values)]
ax.set_xlabel('t')
ax.set_ylabel('p')
ax.set_xticks([0,2.5,5, 7.5,10])
ax.set_xticklabels(x_label_list)
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('t')
ax.set_ylabel('p')
ax.set_zlabel('Lisice')
ax.plot_surface(X,Y,z[:,:,1], cmap=cm.coolwarm)

plt.show()


im = plt.imshow(z[:,:,1], extent=[min(t),max(t),0,10],cmap=cm.coolwarm)
plt.colorbar(im)
x_label_list = [min(p_values), max(p_values)/4, 2*max(p_values)/4, 3*max(p_values)/4, max(p_values)]
ax.set_xlabel('t')
ax.set_ylabel('p')
ax.set_xticks([0,2.5,5, 7.5,10])
ax.set_xticklabels(x_label_list)
plt.show()
"""
from cgi import print_exception
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# function that returns dz/dt

n = 40000 #stevilo korakov integracije
l = 50 #čas do kamor simuliramo
N=10

def model(z,t,p):
    Z = z[0]
    L = z[1]
    dZdt = p*Z*(1-L)
    dLdt = L*(Z-1)/p
    dzdt = [dZdt,dLdt,p]
    return dzdt

Z0 =5
L0 = 5
p0 = 0.5
z0 = [Z0,L0,p0]
t = np.linspace(0,l,n)
X, Y = np.meshgrid(t, p_values)
integration=np.linspace(1,10000,1000)

z=[]
for i in range(1000):
    z.append(0)

c=0
for i in integration:
    t = np.linspace(0,l,n)
    z[c] = odeint(model,z0,t, args=(p0,))
    c=c+1
z = np.array(z)  
print(z[:,:,0]) 
print(z.ndim)


fig, (ax1,ax2) = plt.subplots(2)
fig.suptitle(r"Prikaz obnašanja gozda pri parametru $p$ in začetni populaciji Zajci: $%.2f$, Lisice: $%.2f$" %(Z0,L0))

im1=ax1.imshow(z[:,:,0], extent=[min(t),max(t),0,20],cmap=cm.coolwarm)
axins = inset_axes(ax1,
                    width="5%",  
                    height="100%",
                    loc='center right',
                    borderpad=-5
                   )
plt.colorbar(im1, cax=axins)
y_label_list = np.linspace(10,5000,5)
ax1.set_title("Zajci")
ax1.set_xlabel('t')
ax1.set_ylabel('n')
ax1.set_yticks([0,5,10, 15,20])
ax1.set_yticklabels(y_label_list)
im2 = ax2.imshow(z[:,:,1], extent=[min(t),max(t),0,20],cmap=cm.coolwarm)
axins = inset_axes(ax2,
                    width="5%",  
                    height="100%",
                    loc="center right",
                    borderpad=-5
                   )
plt.colorbar(im2, cax=axins)
y_label_list = np.linspace(10,5000,5)
ax2.set_title("Lisice")
ax2.set_xlabel('t')
ax2.set_ylabel('n')
ax2.set_yticks([0,5,10, 15,20])
ax2.set_yticklabels(y_label_list)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.axes.set_zlim3d(bottom=0, top=20)
ax.set_xlabel("t")
ax.set_ylabel('p')
ax.set_zlabel('Zajci')
surf = ax.plot_surface(X,Y,z[:,:,0], cmap=cm.coolwarm)
fig.colorbar(surf, shrink=0.5, aspect=5)


# Add a color bar which maps values to colors.

#print(z[0])
#print(z)
#plt.plot(t,z[:,0],'b',label='Zajci,  p = %.3f' %(p[0]))
#plt.plot(t,z[:,1],'r',label='Lisice, L_0 = %.3f' %(L0))
#plt.ylabel('brezdimenzijsko število živali')
#plt.xlabel('čas')
#plt.legend(loc='best')

plt.show()

im = plt.imshow(z[:,:,0], extent=[min(t),max(t),0,10],cmap=cm.coolwarm)
plt.colorbar(im)
x_label_list = [min(p_values), max(p_values)/4, 2*max(p_values)/4, 3*max(p_values)/4, max(p_values)]
ax.set_xlabel('t')
ax.set_ylabel('p')
ax.set_xticks([0,2.5,5, 7.5,10])
ax.set_xticklabels(x_label_list)
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('t')
ax.set_ylabel('p')
ax.set_zlabel('Lisice')
ax.plot_surface(X,Y,z[:,:,1], cmap=cm.coolwarm)

plt.show()


im = plt.imshow(z[:,:,1], extent=[min(t),max(t),0,10],cmap=cm.coolwarm)
plt.colorbar(im)
x_label_list = [min(p_values), max(p_values)/4, 2*max(p_values)/4, 3*max(p_values)/4, max(p_values)]
ax.set_xlabel('t')
ax.set_ylabel('p')
ax.set_xticks([0,2.5,5, 7.5,10])
ax.set_xticklabels(x_label_list)
plt.show()

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.integrate as intg
from mpl_toolkits import mplot3d


P=0.1
p=[P]
init_B = 1
y0 = np.array([50, 50])

def Zajci(state, P, t):
    z,l = state
    #alpha=p**2*gamma
    #l=beta/alpha*L
    #z=delta/alpha*Z
    #tau=t*np.sqrt(alpha*gamma)
    dz=P*z*(1-l)
    dl=l/P*(z-1)
    return [dz, dl]

t_max=50.0
N_1=400
t = np.linspace(0.0, t_max, N_1)
#Z,L = odeint(Zajci, y0, t, args=(beta, gamma, delta)).T
plt.title(r"Prikaz obnašanja gozda pri parametrih $p=%.6f$" %(P))
#Z,L= intg.solve_ivp(lambda t, x: Zajci(x, P, t), t_span=(0,t_max),y0=y0,t_eval=t).y
#plt.plot(t,Z, label="Zajci")
p#lt.plot(t,L, label="Lisice")
plt.xlabel("t")
plt.ylabel("Delež")
plt.legend()
plt.show()
Parameter=np.linspace(5,1,400)
N=len(Parameter)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N))
index=0
fig, (ax1,ax2) = plt.subplots(2)
plt.title(r"Prikaz obnašanja gozda pri parametrih $p=%.6f$" %(P))
fig1, (ax4) = plt.subplots(1)
fig1.suptitle(r"Prikaz obnašanja gozda pri parametrih $p=%.6f$" %(P))

for P in Parameter:
    Z,L= intg.solve_ivp(lambda t, x: Zajci(x, P, t), t_span=(0,t_max),y0=y0,t_eval=t).y
    print(index)
    if index%5==0:
        ax4.plot(t,Z,"-", label=r"Zajci pri $p=%.4f$" %(P),color=cm_d[index])
        #ax4.plot(t_line,B,"-.", label=r"Bolni pri $\eta=%.7f$" %(eta),color=cm_b[index])
        ax4.plot(t,L,"-", label=r"Lisice pri $p=%.4f$" %(P), color=cm_i[index])
    if index%4==0:
        ax1.plot(t,Z, label="$p=%.4f$" %(P), color=cm_b[index])
        ax1.set_title("Zajci")
        ax2.plot(t,L, color=cm_b[index])
        ax2.set_title("Lisice")
    index+=1
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center right',bbox_to_anchor=(1, 0.5))
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for ax in fig.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(t, P, Z, 'gray')
plt.show()
alpha=1
BETA=[1,0.5,0.3]
N=len(BETA)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N))
index=0
fig, (ax1,ax2) = plt.subplots(2)
fig.suptitle(r"Prikaz obnašanja gozda pri parametrih $\alpha=%.2f$, $\beta>0$, $\gamma=%.2f$, $\delta=%.2f$" %(alpha,gamma, delta))
fig1, (ax4) = plt.subplots(1)
fig1.suptitle(r"Prikaz obnašanja gozda pri parametrih $\alpha=%.2f$, $\beta>0$, $\gamma=%.2f$, $\delta=%.2f$" %(alpha,gamma, delta))
for beta in BETA:
    Z,L= intg.solve_ivp(lambda t, x: Zajci(x, alpha, beta, gamma, delta, t), t_span=(0,t_max),y0=y0,t_eval=t).y
    if index%N==0:
        ax4.plot(t,Z,"-", label=r"Zajci pri $\beta=%.4f$" %(beta),color=cm_d[index])
        #ax4.plot(t_line,B,"-.", label=r"Bolni pri $\eta=%.7f$" %(eta),color=cm_b[index])
        ax4.plot(t,L,"-", label=r"Lisice pri $\beta=%.4f$" %(beta), color=cm_i[index])
    ax1.plot(t,Z, label=r"$\beta=%.3f$" %(beta), color=cm_b[index])
    ax1.set_title("Zajci")
    ax2.plot(t,L, color=cm_b[index])
    ax2.set_title("Lisice")
    index+=1
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center right',bbox_to_anchor=(1, 0.5))
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for ax in fig.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
plt.show()


alpha=1
beta=1
GAMMA=[1,0.5,0.3]
N=len(GAMMA)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N))
index=0
fig, (ax1,ax2) = plt.subplots(2)
fig.suptitle(r"Prikaz obnašanja gozda pri parametrih $\alpha=%.2f$, $\beta=%.2f$, $\gamma>0$, $\delta=%.2f$" %(alpha,beta, delta))
fig1, (ax4) = plt.subplots(1)
fig1.suptitle(r"Prikaz obnašanja gozda pri parametrih $\alpha=%.2f$, $\beta=%.2f$, $\gamma>0$, $\delta=%.2f$" %(alpha,beta, delta))
for gamma in GAMMA:
    Z,L= intg.solve_ivp(lambda t, x: Zajci(x, alpha, beta, gamma, delta, t), t_span=(0,t_max),y0=y0,t_eval=t).y
    if index%N==0:
        ax4.plot(t,Z,"-", label=r"Zajci pri $\gamma=%.4f$" %(gamma),color=cm_d[index])
        #ax4.plot(t_line,B,"-.", label=r"Bolni pri $\eta=%.7f$" %(eta),color=cm_b[index])
        ax4.plot(t,L,"-", label=r"Lisice pri $\gamma=%.4f$" %(gamma), color=cm_i[index])
    ax1.plot(t,Z, label=r"$\gamma=%.3f$" %(gamma), color=cm_b[index])
    ax1.set_title("Zajci")
    ax2.plot(t,L, color=cm_b[index])
    ax2.set_title("Lisice")
    index+=1
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center right',bbox_to_anchor=(1, 0.5))
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for ax in fig.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
plt.show()


alpha=1
beta=1
gamma=1
DELTA=[1,0.5,0.3]
N=len(DELTA)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N))
index=0
fig, (ax1,ax2) = plt.subplots(2)
fig.suptitle(r"Prikaz obnašanja gozda pri parametrih $\alpha=%.2f$, $\beta=%.2f$, $\gamma=%.2f$, $\delta<=$" %(alpha,beta, gamma))
fig1, (ax4) = plt.subplots(1)
fig1.suptitle(r"Prikaz obnašanja gozda pri parametrih $\alpha=%.2f$, $\beta=%.2f$, $\gamma=%.2f$, $\delta>0$" %(alpha,beta, gamma))
for delta in DELTA:
    Z,L= intg.solve_ivp(lambda t, x: Zajci(x, alpha, beta, gamma, delta, t), t_span=(0,t_max),y0=y0,t_eval=t).y
    if index%N==0:
        ax4.plot(t,Z,"-", label=r"Zajci pri $\delta=%.4f$" %(delta),color=cm_d[index])
        #ax4.plot(t_line,B,"-.", label=r"Bolni pri $\eta=%.7f$" %(eta),color=cm_b[index])
        ax4.plot(t,L,"-", label=r"Lisice pri $\delta=%.4f$" %(delta), color=cm_i[index])
    ax1.plot(t,Z, label=r"$\delta=%.3f$" %(delta), color=cm_b[index])
    ax1.set_title("Zajci")
    ax2.plot(t,L, color=cm_b[index])
    ax2.set_title("Lisice")
    index+=1
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center right',bbox_to_anchor=(1, 0.5))
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for ax in fig.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
plt.show()