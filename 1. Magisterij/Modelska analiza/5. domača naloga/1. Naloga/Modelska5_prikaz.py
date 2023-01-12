import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
n=250
l=30
a=1000
b=10
def model (z,t,b):
    x,y,w=z
    dxdt=-x*(x-a*y)
    dydt=x*(x-a*y)-a*b*y
    dwdt=a*b*y
    dzdt=[dxdt,dydt,dwdt]
    return dzdt

x0=1
y0=0
w0=0
z0=[x0,y0,w0]

t=np.linspace(0,50,n)
b=10
n_lines=20
b_max=1
B=np.linspace(0.1,b_max,n_lines)
B_tixcs=[]
cm = plt.cm.winter
fig,(ax,ax2,ax3) = plt.subplots(3)

c = np.arange(1, n_lines + 1)
i=0.1
while i<b_max:
    B_tixcs.append(i)
    i+=0.1
B_tixcs=np.array(B_tixcs)


norm = mpl.colors.Normalize(vmin=B_tixcs.min(), vmax=B_tixcs.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
cmap.set_array([])

index=0
fig.suptitle("Graf odvisnosti spojin od izbire parametra $b$ z $a=1000$ in $A/A_0(0)$=1")
for b in B:
    x,y,w=odeint(model,z0,t,args=(b,)).T
    ax.plot(t,x,label="b=%.2f"%(b),c=cmap.to_rgba(index/len(B)))
    ax.set_title(r"Prikaz spojine $A/A_0$")
    ax2.set_title(r"Prikaz spojine $A^*/A_0$")
    ax2.plot(t,y,label="b=%.2f"%(b),c=cmap.to_rgba(index/len(B)))
    ax3.set_title(r"Prikaz spojine $B/A_0$")
    ax3.plot(t,w,label="b=%.2f"%(b),c=cmap.to_rgba(index/len(B)))
    index+=1
fig.colorbar(cmap, cax = fig.add_axes([0.92, 0.1, 0.03, 0.8]) ,ticks=B_tixcs)
plt.show()