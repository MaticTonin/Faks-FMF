import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.integrate as intg

alpha = 1
beta = 0.5
t0 = 20
p=[alpha, beta]
init_B = 0.0001
y0 = np.array([1-init_B, init_B, 0])
gamma=0.1
tau_1=7
tau_2=1
eta=0.01

#y0 = np.array([20000, 1, 0])

def Epidemija(state, alpha, beta,eta,t):
    D, B, I = state
    dD = -alpha*D*B+eta*I
    dB = alpha*D*B-B*beta
    dI = B*beta-eta*I
    return [dD, dB, dI]

t_max=700.0
t_line = np.arange(0.0, t_max, 0.01)
"""
#D,B,I = odeint(Epidemija, y0, t, args=(alpha,beta)).T
plt.title(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.6f$ in $\beta=%.2f$, dodatek sinusa" %(alpha,beta))
D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta,eta, t), t_span=(0,t_max),y0=y0,t_eval=np.arange(0.0, t_max, 0.01)).y
plt.plot(t_line,D, label="Dovzetni")
plt.plot(t_line,B, label="Bolni")
plt.plot(t_line,I, label="Imuni")
plt.xlabel("t")
plt.ylabel("Delež")
plt.legend()
plt.show()

alpha = 1
beta = 0.5
BETA=[0.5,0.1,0.01]
BETA=np.linspace(1,0.01,20)
N=len(BETA)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N))
index=0
fig, (ax1,ax2,ax3) = plt.subplots(3)
fig.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\eta=0.01$ $\alpha=%.2f$ in $\beta=[0.01-1]$" %(alpha))
fig1, (ax4) = plt.subplots(1)
fig1.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\eta=0.01$ $\alpha=%.2f$ in $\beta=[0.01-1]$" %(alpha))
for beta in BETA:
    D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta,eta, t), t_span=(0,t_max),y0=y0,t_eval=np.arange(0.0, t_max, 0.01)).y
    if index%5==0:
        ax4.plot(t_line,D, label=r"Dovzetni pri $\beta=%.4f$" %(beta),color=cm_d[index])
        #ax4.plot(t_line,B,"-.", label=r"Bolni pri $\beta=%.4f$" %(beta),color=cm_b[index])
        ax4.plot(t_line,I, label=r"Immuni pri $\beta=%.4f$" %(beta), color=cm_i[index])
    if index%3==0:
        ax1.plot(t_line,D, label=r"$\beta=%.4f$" %(beta), color=cm_d[index])
        ax1.set_title("Dovzetni")
        ax2.plot(t_line,B, color=cm_d[index])
        ax2.set_title("Bolni")
        ax3.plot(t_line,I, color=cm_d[index])
        ax3.set_title("Imuni")
        ax1.set_ylim(-0.05,1.05)
        ax2.set_ylim(-0.05,1.05)
        ax3.set_ylim(-0.05,1.05)
    index+=1
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center right',bbox_to_anchor=(1, 0.5))
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for ax in fig.get_axes() and fig1.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
plt.show()
plt.show()

alpha = 1
beta = 0.5
ETA=[0.1,0.01,0.001]
ETA=np.linspace(0.5,0.001,20)
N=len(ETA)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N))
index=0
fig, (ax1,ax2,ax3) = plt.subplots(3)
fig.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.2f$ $\beta=%.2f$ in $\eta>0$" %(alpha,beta))
fig1, (ax4) = plt.subplots(1)
fig1.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.2f$ $\beta=%.2f$ in $\eta[0.5,0.001]$" %(alpha,beta))
for eta in ETA:
    D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta,eta, t), t_span=(0,t_max),y0=y0,t_eval=np.arange(0.0, t_max, 0.01)).y
    if index%4==0:
        ax4.plot(t_line,D,"-", label=r"Dovzetni pri $\eta=%.7f$" %(eta),color=cm_d[index])
        #ax4.plot(t_line,B,"-.", label=r"Bolni pri $\eta=%.7f$" %(eta),color=cm_b[index])
        ax4.plot(t_line,I,"--", label=r"Immuni pri $\eta=%.7f$" %(eta), color=cm_i[index])
    if index%3==0:
        ax1.plot(t_line,D, label=r"$\eta=%.7f$" %(eta), color=cm_d[index])
        ax1.set_title("Dovzetni")
        ax2.plot(t_line,B, color=cm_d[index])
        ax2.set_title("Bolni")
        ax3.plot(t_line,I, color=cm_d[index])
        ax3.set_title("Imuni")
        ax1.set_ylim(-0.05,1.05)
        ax2.set_ylim(-0.05,1.05)
        ax3.set_ylim(-0.05,1.05)
    index+=1
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center right',bbox_to_anchor=(1, 0.5))
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for ax in fig.get_axes() and fig1.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
plt.show()
plt.show()


alpha = 1
beta = 0.5
ETA=[0.1,0.01,0.001]
ETA=np.linspace(0.01,0.0001,20)
N=len(ETA)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N))
index=0
fig, (ax1,ax2,ax3) = plt.subplots(3)
fig.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.2f$ $\beta=%.2f$ in $\eta>0$" %(alpha,beta))
fig1, (ax4) = plt.subplots(1)
fig1.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.2f$ $\beta=%.2f$ in $\eta[0.01,0.0001]$" %(alpha,beta))
for eta in ETA:
    D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta,eta, t), t_span=(0,t_max),y0=y0,t_eval=np.arange(0.0, t_max, 0.01)).y
    if index%4==0:
        ax4.plot(t_line,D,"-", label=r"Dovzetni pri $\eta=%.7f$" %(eta),color=cm_d[index])
        #ax4.plot(t_line,B,"-.", label=r"Bolni pri $\eta=%.7f$" %(eta),color=cm_b[index])
        ax4.plot(t_line,I,"--", label=r"Immuni pri $\eta=%.7f$" %(eta), color=cm_i[index])
    if index%3==0:
        ax1.plot(t_line,D, label=r"$\eta=%.7f$" %(eta), color=cm_d[index])
        ax1.set_title("Dovzetni")
        ax2.plot(t_line,B, color=cm_d[index])
        ax2.set_title("Bolni")
        ax3.plot(t_line,I, color=cm_d[index])
        ax3.set_title("Imuni")
        ax1.set_ylim(-0.05,1.05)
        ax2.set_ylim(-0.05,1.05)
        ax3.set_ylim(-0.05,1.05)
    index+=1
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center right',bbox_to_anchor=(1, 0.5))
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for ax in fig.get_axes() and fig1.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
plt.show()
plt.show()
"""

alpha = 1
beta = 0.5
t0 = 20
p=[alpha, beta]
init_B = 0.0001
y0 = np.array([1-init_B, init_B, 0])
gamma=1
tau_1=7
tau_2=1
eta=0.01

#y0 = np.array([20000, 1, 0])

def Epidemija(state, alpha, beta,gamma,t):
    D, B, I = state
    dD = -alpha*D*B-abs(np.sin(2*np.pi*t/tau_2))*gamma*D*B+eta*I
    dB = alpha*D*B-B*beta+abs(np.sin(2*np.pi*t/tau_2))*gamma*D*B
    dI = B*beta-eta*I
    return [dD, dB, dI]

alpha = 1
beta = 0.5
BETA=[0.5,0.1,0.01]
BETA=np.linspace(1,0.01,20)
N=len(BETA)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N))
index=0
fig, (ax1,ax2,ax3) = plt.subplots(3)
fig.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\eta=0.01$ $\alpha=%.2f$ in $\beta=[0.01-1]$ Z dodatkov \|sin$(2\pi\cdot t /\tau)\cdot\gamma\|$ pri $\gamma=1, \tau=1$" %(alpha))
fig1, (ax4) = plt.subplots(1)
fig1.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\eta=0.01$ $\alpha=%.2f$ in $\beta=[0.01-1]$ Z dodatkov \|sin$(2\pi\cdot t /\tau)\cdot\gamma\|$ pri $\gamma=1, \tau=1$" %(alpha))
for beta in BETA:
    D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta,eta, t), t_span=(0,t_max),y0=y0,t_eval=np.arange(0.0, t_max, 0.01)).y
    if index%5==0:
        ax4.plot(t_line,D, label=r"Dovzetni pri $\beta=%.4f$" %(beta),color=cm_d[index])
        #ax4.plot(t_line,B,"-.", label=r"Bolni pri $\beta=%.4f$" %(beta),color=cm_b[index])
        ax4.plot(t_line,I, label=r"Immuni pri $\beta=%.4f$" %(beta), color=cm_i[index])
    if index%3==0:
        ax1.plot(t_line,D, label=r"$\beta=%.4f$" %(beta), color=cm_d[index])
        ax1.set_title("Dovzetni")
        ax2.plot(t_line,B, color=cm_d[index])
        ax2.set_title("Bolni")
        ax3.plot(t_line,I, color=cm_d[index])
        ax3.set_title("Imuni")
        ax1.set_ylim(-0.05,1.05)
        ax2.set_ylim(-0.05,1.05)
        ax3.set_ylim(-0.05,1.05)
    index+=1
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center right',bbox_to_anchor=(1, 0.5))
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for ax in fig.get_axes() and fig1.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
plt.show()
plt.show()

alpha = 1
beta = 0.5
ETA=[0.1,0.01,0.001]
ETA=np.linspace(0.5,0.001,20)
N=len(ETA)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N))
index=0
fig, (ax1,ax2,ax3) = plt.subplots(3)
fig.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.2f$ $\beta=%.2f$ in $\eta>0$ Z dodatkov sin$(2\pi\cdot t /\tau)\cdot\gamma$ pri $\gamma=1, \tau=1$" %(alpha,beta))
fig1, (ax4) = plt.subplots(1)
fig1.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.2f$ $\beta=%.2f$ in $\eta[0.5,0.001]$ Z dodatkov sin$(2\pi\cdot t /\tau)\cdot\gamma$ pri $\gamma=1, \tau=7$" %(alpha,beta))
for eta in ETA:
    D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta,eta, t), t_span=(0,t_max),y0=y0,t_eval=np.arange(0.0, t_max, 0.01)).y
    if index%4==0:
        ax4.plot(t_line,D,"-", label=r"Dovzetni pri $\eta=%.7f$" %(eta),color=cm_d[index])
        #ax4.plot(t_line,B,"-.", label=r"Bolni pri $\eta=%.7f$" %(eta),color=cm_b[index])
        ax4.plot(t_line,I,"--", label=r"Immuni pri $\eta=%.7f$" %(eta), color=cm_i[index])
    if index%3==0:
        ax1.plot(t_line,D, label=r"$\eta=%.7f$" %(eta), color=cm_d[index])
        ax1.set_title("Dovzetni")
        ax2.plot(t_line,B, color=cm_d[index])
        ax2.set_title("Bolni")
        ax3.plot(t_line,I, color=cm_d[index])
        ax3.set_title("Imuni")
        ax1.set_ylim(-0.05,1.05)
        ax2.set_ylim(-0.05,1.05)
        ax3.set_ylim(-0.05,1.05)
    index+=1
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center right',bbox_to_anchor=(1, 0.5))
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for ax in fig.get_axes() and fig1.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
plt.show()
plt.show()


alpha = 1
beta = 0.5
ETA=[0.1,0.01,0.001]
ETA=np.linspace(0.01,0.0001,20)
N=len(ETA)
cm_d=cm.winter(np.linspace(0,1,N))
cm_b=cm.autumn(np.linspace(0,1,N))
cm_i=cm.spring(np.linspace(0,1,N))
index=0
fig, (ax1,ax2,ax3) = plt.subplots(3)
fig.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.2f$ $\beta=%.2f$ in $\eta>0$" %(alpha,beta))
fig1, (ax4) = plt.subplots(1)
fig1.suptitle(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.2f$ $\beta=%.2f$ in $\eta[0.01,0.0001]$" %(alpha,beta))
for eta in ETA:
    D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta,eta, t), t_span=(0,t_max),y0=y0,t_eval=np.arange(0.0, t_max, 0.01)).y
    if index%4==0:
        ax4.plot(t_line,D,"-", label=r"Dovzetni pri $\eta=%.7f$" %(eta),color=cm_d[index])
        #ax4.plot(t_line,B,"-.", label=r"Bolni pri $\eta=%.7f$" %(eta),color=cm_b[index])
        ax4.plot(t_line,I,"--", label=r"Immuni pri $\eta=%.7f$" %(eta), color=cm_i[index])
    if index%3==0:
        ax1.plot(t_line,D, label=r"$\eta=%.7f$" %(eta), color=cm_d[index])
        ax1.set_title("Dovzetni")
        ax2.plot(t_line,B, color=cm_d[index])
        ax2.set_title("Bolni")
        ax3.plot(t_line,I, color=cm_d[index])
        ax3.set_title("Imuni")
        ax1.set_ylim(-0.05,1.05)
        ax2.set_ylim(-0.05,1.05)
        ax3.set_ylim(-0.05,1.05)
    index+=1
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center right',bbox_to_anchor=(1, 0.5))
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for ax in fig.get_axes() and fig1.get_axes():
    ax.set_xlabel("t")
    ax.set_ylabel("Delež")
plt.show()
plt.show()