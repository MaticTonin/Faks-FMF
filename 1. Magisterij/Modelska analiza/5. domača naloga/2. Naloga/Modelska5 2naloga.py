import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


n = 2500 
l = 5
def model(w,ti):
    u = w[0]
    v = w[1]
    x = w[2]
    y = w[3]
    z = w[4]
    dudt = s*x*y - r* u* z
    dvdt = q*z**2 - p*v - t*v*y
    dxdt = r*u*z + t*v*y - s*x*y
    dydt = r*u*z - s*x*y -t*v*y
    dzdt = 2*p*v + t*v*y - 2*q*z**2 + s*x*y -r*u*z

    dwdt = [dudt, dvdt, dxdt,dydt, dzdt]
    return dwdt


p = 5
q = 4
r = 3
s = 2
t = 1

u0 = 1  #H2
v0 = 1  #Br2
x0 = 0  #HBr
y0 = 0  #H
z0 = 0  #Br

w0 = [u0, v0, x0, y0, z0]

# time points
ti = np.linspace(0,l,n)


# solve ODE
w = odeint(model,w0,ti)
# plot results
plt.title("Prikaz reakcije; $p=%.2f$, $q=%.2f$, $r=%.2f$, $s=%.2f$, $t=%.2f$" %(p,q,r,s,t))
plt.plot(ti,w[:,0],label='H2_0 = %.1f' %(u0), color=plt.cm.gnuplot(0/len(w0)))
plt.plot(ti,w[:,1],label=' Br2_0 = %.0f' %(v0), color=plt.cm.gnuplot(1/len(w0)))
plt.plot(ti,w[:,2],label='HBr_0 = %.0f' %(x0), color=plt.cm.gnuplot(2/len(w0)))
plt.plot(ti,w[:,3],label='H_0 = %.0f' %(y0), color=plt.cm.gnuplot(3/len(w0)))
plt.plot(ti,w[:,4],label='Br_0 = %.0f' %(z0), color=plt.cm.gnuplot(4/len(w0)))


plt.ylabel('Koncentracija')
plt.xlabel('čas')
plt.grid()
plt.legend(loc='upper right')

plt.show()