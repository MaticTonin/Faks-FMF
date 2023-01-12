
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.integrate as intg

alpha = 1
beta = 1
gamma=1
delta=1
t0 = 20
p=[alpha, beta, gamma, delta]
init_B = 1
y0 = np.array([5, 5])

def Zajci(state, alpha, beta, gamma, delta,t):
    Z,L = state
    l=beta/alpha*L
    z=delta/alpha*Z
    tau=t*np.sqrt(alpha*gamma)
    p=np.sqrt(alpha/gamma)
    dz=p*z*(1-l)
    dl=l/p*(z-1)
    return [dz, dl]

t_max=2000.0
t = np.arange(0.0, t_max, 0.01)
alpha=1
#Z,L = odeint(Zajci, y0, t, args=(beta, gamma, delta)).T
plt.title(r"Prikaz obnašanja gozda pri parametrih $\alpha=%.6f$, $\beta=%.2f$, $\gamma=%.2f$, $\delta=%.2f$" %(alpha,beta,gamma, delta))
Z,L= intg.solve_ivp(lambda t, x: Zajci(x, alpha, beta, gamma, delta, t), t_span=(0,t_max),y0=y0,t_eval=t).y
plt.plot(t,Z, label="Zajci")
plt.plot(t,L, label="Lisice")
plt.xlabel("t")
plt.ylabel("Delež")
plt.legend()
plt.show()
ALPHA=[1,0.5,0.3]
N=len(ALPHA)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N))
index=0
fig, (ax1,ax2) = plt.subplots(2)
fig.suptitle(r"Prikaz obnašanja gozda pri parametrih $\alpha>0$, $\beta=%.2f$, $\gamma=%.2f$, $\delta=%.2f$" %(beta,gamma, delta))
fig1, (ax4) = plt.subplots(1)
fig1.suptitle(r"Prikaz obnašanja gozda pri parametrih $\alpha>0$, $\beta=%.2f$, $\gamma=%.2f$, $\delta=%.2f$" %(beta,gamma, delta))
for alpha in ALPHA:
    Z,L= intg.solve_ivp(lambda t, x: Zajci(x, alpha, beta, gamma, delta, t), t_span=(0,t_max),y0=y0,t_eval=t).y
    if index%(N)==1:
        ax4.plot(t,Z,"-", label=r"Zajci pri $\alpha=%.7f$" %(alpha),color=cm_d[index])
        #ax4.plot(t_line,B,"-.", label=r"Bolni pri $\eta=%.7f$" %(eta),color=cm_b[index])
        ax4.plot(t,L,"-", label=r"Lisice pri $\alpha=%.7f$" %(alpha), color=cm_i[index])
    ax1.plot(t,Z, label=r"$\alpha=%.3f$" %(alpha), color=cm_b[index])
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