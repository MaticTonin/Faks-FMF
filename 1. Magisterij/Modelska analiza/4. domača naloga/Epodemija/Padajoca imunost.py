import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.integrate as intg

alpha = 1
beta = 0.1
t0 = 20
p=[alpha, beta]
init_B = 0.01
y0 = np.array([1-init_B, init_B, 0])
alpha = 10**(-6)
y0 = np.array([200000, 1, 10000])
tau=90
def Epidemija(state, alpha, beta,t, tau):
    D, B, I = state
    if I<0:
        I=0
    if B>200000:
        B=200000
    dB = alpha*D*B-B*beta
    dI = B*beta-I*np.exp(t/tau)
    dD = -alpha*D*B
    return [dD, dB, dI]

t_max=400.0
t_line = np.arange(0.0, t_max, 0.01)

#D,B,I = odeint(Epidemija, y0, t, args=(alpha,beta)).T
plt.title(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.6f$ in $\beta=%.2f$, dodatek padca imunosti $\tau=%.2f$" %(alpha,beta, tau))
D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta, t, tau), t_span=(0,t_max),y0=y0,t_eval=np.arange(0.0, t_max, 0.01)).y
for i in range(len(D)):
    if I[i]<0:
        I[i]=0
    if B[i]>200000:
        B[i]=200000
plt.plot(t_line,D, label="Dovzetni")
plt.plot(t_line,B, label="Bolni")
plt.plot(t_line,I, label="Imuni")
plt.xlabel("t")
plt.ylabel("Delež")
plt.legend()
plt.show()
TAU=[90,100,110,120]
t_max=600.0
t = np.arange(0.0, t_max, 0.01)
N=len(TAU)
cm_d=cm.brg(np.linspace(0,1,N))
cm_b=cm.winter(np.linspace(0,1,N))
cm_i=cm.cool(np.linspace(0,1,N))
index=0
fig, (ax1,ax2,ax3) = plt.subplots(3)
fig.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.6f$ $\beta=%.2f$" %(alpha,beta))
for tau in TAU:
    D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta, t, tau), t_span=(0,t_max),y0=y0,t_eval=t).y
    ax1.plot(t,D, label=r"Dovzetni pri $\tau=%.2f$" %(tau), color=cm_d[index])
    ax2.plot(t,B, label=r"Bolni pri $\tau=%.2f$" %(tau), color=cm_b[index])
    ax3.plot(t,I, label=r"Imuni pri $\tau=%.2f$" %(tau), color=cm_i[index])
    index+=1
for ax in fig.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
    ax.legend()
plt.show()