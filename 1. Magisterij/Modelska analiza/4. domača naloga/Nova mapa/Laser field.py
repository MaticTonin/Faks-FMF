import numpy as np
import scipy.integrate as intg
import matplotlib.pyplot as plt


n = 400
l = 10



t = np.linspace(0,l,n)
def diff_eq(y, p, r):
    a, f = y

    return np.array([r - p*a * (f+1),  f / p * (a - 1)])

small_F = lambda t, p, r, f0: f0 * np.exp((r/p - 1) * t / p)






def model(z,t,p,r):
    A = z[0]
    F = z[1]
    dAdt = r - p* A *(F + 1)
    dFdt = F/p*(A - 1)
    dzdt = [dAdt,dFdt]
    return dzdt


p = 0.2
r = 1
e =0.01

A0 = 1
F0 = 1
z0 = [A0,F0]
t = np.linspace(0,l,n)
y0 = z0
N = n
t0 = l
R=np.linspace(0.5,10,200)

x, y = np.meshgrid(np.linspace(0, 3, n), np.linspace(-0, 3, n))
print(x)
u = np.empty((N, N))
v = np.empty((N, N))
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(N):
    for j in range(N):
        u[i, j], v[i, j] = diff_eq(np.array([x[i, j], y[i, j]]), p, r)
color=color = 2 * np.log(np.hypot(u, v))
ax.streamplot(x, y, u, v, color=color, linewidth=1, cmap=plt.cm.ocean, density=2, arrowstyle='->', arrowsize=1.5)

index=0
for i in R:
    y0 = [i,F0]
    sol = intg.solve_ivp(lambda t, y: diff_eq(y, p, r), (0, t0), y0, t_eval=np.linspace(0, t0, N), method="DOP853", max_step=0.01)
    ax.plot(sol.y[0],sol.y[1],label=r"A_0 = %.3f" %(i), color=plt.cm.gnuplot(index/len(R)))
    index+=1
ax.scatter(r/p,0,label="(r/p,0)", color="red")
ax.scatter(1,r/p-1,label="(1,r/p-1)",color="green")
ax.legend()
plt.xlabel(r"$a$")
plt.ylabel(r"$f$")

fig.suptitle(r"Prikaz modela laserja za $F_0=%.2f$ in $p=%.2f$"%(F0,p))

plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)

ax.streamplot(x, y, u, v, color=color, linewidth=1, cmap=plt.cm.ocean, density=2, arrowstyle='->', arrowsize=1.5)
index=0
for i in R:
    y0 = [A0,i]
    sol = intg.solve_ivp(lambda t, y: diff_eq(y, p, r), (0, t0), y0, t_eval=np.linspace(0, t0, N), method="DOP853", max_step=0.01)
    ax.plot(sol.y[0],sol.y[1],label=r"F_0 = %.3f" %(i), color=plt.cm.gnuplot(index/len(R)))
    index+=1
ax.scatter(r/p,0,label="(r/p,0)", color="red")
ax.scatter(1,r/p-1,label="(1,r/p-1)",color="green")
ax.legend()
plt.xlabel(r"$a$")
plt.ylabel(r"$f$")

fig.suptitle(r"Prikaz modela laserja za $A_0=%.2f$ in $p=%.2f$"%(F0,p))

plt.show()

fig, (ax1,ax2)=plt.subplots(2)
fig.suptitle(r"Prikaz modela laserja za $F_0=%.2f$ in $p=%.2f$"%(F0,p))
for i in R:
    y0 = [A0,F0]
    sol = intg.solve_ivp(lambda t, y: diff_eq(y, p, i), (0, t0), y0, t_eval=np.linspace(0, t0, N), method="DOP853", max_step=0.01)
    ax1.plot(sol.y[0],sol.y[1],label='r = %.3f' %(i))
    y0 = [i,F0]
    sol = intg.solve_ivp(lambda t, y: diff_eq(y, p, r), (0, t0), y0, t_eval=np.linspace(0, t0, N), method="DOP853", max_step=0.01)
    ax2.plot(sol.y[0],sol.y[1],label=r"A_0 = %.3f" %(i))
    ax1.set_title(r"$A_0=%.2f$"%(A0))
    ax2.set_title(r"$r=%.2f$"%(r))
ax1.legend()
ax1.grid()
ax2.grid()
ax2.legend()
ax2.set_xlabel(r"t")
ax1.set_ylabel(r"A")
ax2.set_ylabel(r"A")
plt.show()
from tqdm import tqdm
time_index=0
time_vector=[]
for j in tqdm(R):
    sol = intg.solve_ivp(lambda t, y: diff_eq(y, p, j), (0, t0), y0, t_eval=np.linspace(0, t0, N), method="DOP853", max_step=0.01)
    time_index=0
    for i in range(len(sol.y[0])):
        if abs(sol.y[0][i]-1)<0.1 and time_index==0 and i>7:
            time=abs(t[i]-t[0])
            time_vector.append(time)
            time_index+=1
plt.title("Prikaz odvisnosti obhodnih časov od izbire razmerja $r/p$")
plt.plot(R/p, time_vector)
plt.xlabel(r"r/p")
plt.ylabel(r"t")
plt.show()