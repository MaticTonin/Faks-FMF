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
y0 = np.array([200000, 1, 0])

def Epidemija(state, alpha, beta,t):
    D, B, I = state
    dB = alpha*D*B-B*beta+(1-np.sin(2*np.pi*t/7))*1000
    dD = -alpha*D*B
    dI = B*beta-(1-np.sin(2*np.pi*t/7))*1000
    if dI<0:
        dI = B*beta-(1-np.sin(2*np.pi*t/7))*1000
    return [dD, dB, dI]

t_max=200.0
t_line = np.arange(0.0, t_max, 0.01)

#D,B,I = odeint(Epidemija, y0, t, args=(alpha,beta)).T
plt.title(r"Prikaz obnašanja epidemije pri parametrih $\alpha=%.6f$ in $\beta=%.2f$, dodatek sinusa" %(alpha,beta))
D,B,I= intg.solve_ivp(lambda t, x: Epidemija(x, alpha, beta, t), t_span=(0,t_max),y0=y0,t_eval=np.arange(0.0, t_max, 0.01)).y
plt.plot(t_line,D, label="Dovzetni")
plt.plot(t_line,B, label="Bolni")
plt.plot(t_line,I, label="Imuni")
plt.xlabel("t")
plt.ylabel("Delež")
plt.legend()
plt.show()