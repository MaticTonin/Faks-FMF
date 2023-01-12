import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.integrate as intg

alpha = 1
beta = 0.1
t0 = 20
p=[alpha, beta]
init_B = 0.001
y0 = np.array([1-init_B, init_B, 0])
#alpha = 10**(-5)
#y0 = np.array([200000, 1, 0])

def Epidemija(state, alpha, beta,t):
    D, B, I = state
    dD = -alpha*D*B
    dB = alpha*D*B-B*beta
    dI = B*beta
    return [dD, dB, dI]

t_max=150.0
t = np.arange(0.0, t_max, 0.01)

D,B,I = odeint(Epidemija, y0, t, args=(alpha,beta)).T
plt.title(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.6f$ in $\beta=%.2f$" %(alpha,beta))
D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta, t), t_span=(0,t_max),y0=y0,t_eval=t).y
plt.plot(t,D, label="Dovzetni")
plt.plot(t,B, label="Bolni")
plt.plot(t,I, label="Imuni")
plt.xlabel("t")
plt.ylabel("Delež")
plt.legend()
plt.show()
ALPHA=[1,0.5,0.3]
N=len(ALPHA)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N+1))
index=0
fig, (ax1,ax2,ax3) = plt.subplots(3)
fig.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\alpha>0$ $\beta=%.2f$" %(beta))
for alpha in ALPHA:
    D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta, t), t_span=(0,t_max),y0=y0,t_eval=t).y
    ax1.plot(t,D, label=r"Dovzetni pri $\alpha=%.3f$" %(alpha), color=cm_d[index])
    ax2.plot(t,B, label=r"Bolni pri $\alpha=%.3f$" %(alpha), color=cm_b[index])
    ax3.plot(t,I, label=r"Imuni pri $\alpha=%.3f$" %(alpha), color=cm_i[index])
    ax1.set_ylim(-0.05,1.05)
    ax2.set_ylim(-0.05,1.05)
    ax3.set_ylim(-0.05,1.05)
    index+=1
for ax in fig.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
    ax.legend()
plt.show()


ALPHA=[1,0.5,0.3]
N=len(ALPHA)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N+1))
index=0
fig, (ax) = plt.subplots(1)
fig.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\alpha>0$ $\beta=%.2f$" %(beta))
for alpha in ALPHA:
    D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta, t), t_span=(0,t_max),y0=y0,t_eval=t).y
    ax.plot(t,D,"-", label=r"Dovzetni pri $\alpha=%.3f$" %(alpha),color=cm_d[index])
    ax.plot(t,B,"-.", label=r"Bolni pri $\alpha=%.3f$" %(alpha),color=cm_b[index])
    ax.plot(t,I,"--", label=r"Immuni pri $\alpha=%.3f$" %(alpha), color=cm_i[index])
    index+=1
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for ax in fig.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
plt.show()


BETA=[0.1,0.01,0.001]
alpha=1
N=len(BETA)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N+1))
index=0
fig, (ax1,ax2,ax3) = plt.subplots(3)
fig.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.3f$ $\beta>0$" %(alpha))
for beta in BETA:
    D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta, t), t_span=(0,t_max),y0=y0,t_eval=t).y
    ax1.plot(t,D, label=r"Dovzetni pri $\beta=%.4f$" %(beta), color=cm_d[index])
    ax2.plot(t,B, label=r"Bolni pri $\beta=%.4f$" %(beta), color=cm_b[index])
    ax3.plot(t,I, label=r"Imuni pri $\beta=%.4f$" %(beta), color=cm_i[index])
    ax1.set_ylim(-0.05,1.05)
    ax2.set_ylim(-0.05,1.05)
    ax3.set_ylim(-0.05,1.05)
    index+=1
for ax in fig.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
    ax.legend()
plt.show()

BETA=[0.1,0.01,0.001]
alpha=1
N=len(BETA)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N+1))
index=0
fig, (ax) = plt.subplots(1)
fig.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.3f$ $\beta>0$" %(alpha))
for alpha in ALPHA:
    D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta, t), t_span=(0,t_max),y0=y0,t_eval=t).y
    ax.plot(t,D,"-", label=r"Dovzetni pri $\beta=%.4f$" %(beta),color=cm_d[index])
    ax.plot(t,B,"-.", label=r"Bolni pri $\beta=%.4f$" %(beta),color=cm_b[index])
    ax.plot(t,I,"--", label=r"Immuni pri $\beta=%.4f$" %(beta), color=cm_i[index])
    index+=1
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for ax in fig.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
plt.show()