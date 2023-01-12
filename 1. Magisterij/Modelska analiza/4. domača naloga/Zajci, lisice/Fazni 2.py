import numpy as np
import scipy.integrate as intg
import matplotlib.pyplot as plt

def diff_eq(y, p):
    z, l = y

    return np.array([p*z * (1-l), l / p * (z-1)])

p = 1

N = 20
x = np.linspace(-0.5, 2, N)
y = np.linspace(-0.5, 2, N)

x, y = np.meshgrid(x, y)

u = np.empty((N, N))
v = np.empty((N, N))

for i in range(N):
    for j in range(N):
        u[i, j], v[i, j] = diff_eq(np.array([x[i, j], y[i, j]]), p)

qq=plt.quiver(x, y, u, v,np.sqrt(u**2+v**2),cmap=plt.cm.coolwarm)
plt.colorbar(qq, cmap=plt.cm.jet)

plt.xlabel(r"$z$")
plt.ylabel(r"$l$")

plt.title("Fazni portret modela zajci-lisice, p=1")

plt.show()
plt.close()