import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
n=50000
a=1000
b=0.1

def model_original (z,t,b):
    x,y,w=z
    dxdt=-x*(x-a*y)
    dydt=x*(x-a*y)-a*b*y
    dwdt=a*b*y
    dzdt=[dxdt,dydt,dwdt]
    return dzdt

def model_stac (z,t,b):
    x,w=z
    dxdt=-x**2+x**3/(b+x)
    dwdt=(b*x**2)/(b+x)
    dzdt=[dxdt,dwdt]
    
    return dzdt

x0=1
y0=0
w0=0
z0=[x0,y0,w0]

t=np.linspace(0,50,n)
b=0.1
n_lines=6
b_max=1
B=[0.1,1,10]
B_tixcs=[]
cm = plt.cm.winter
for b in B:
    z0=[x0,w0]
    x,w=odeint(model_stac,z0,t,args=(b,)).T
    z0=[x0,y0,w0]
    x1,y1,w1=odeint(model_original,z0,t,args=(b,)).T
    plt.subplot(1, 2, 1)
    plt.title("Eksaktna rešitev pri $b=%.2f$ in $A/A_0(0)$=1" %(b))
    plt.plot(t,x1,label="x(t)", color=plt.cm.gnuplot(0/len(B)))
    plt.plot(t,y1,label="y(t)", color=plt.cm.gnuplot(1/len(B)))
    plt.plot(t,w1,label="z(t)", color=plt.cm.gnuplot(2/len(B)))
    plt.xlabel("t")
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.title("Stacionarna rešitev pri $b=%.2f$ in $A/A_0(0)$=1" %(b))
    plt.plot(t,x,label="x(t)", color=plt.cm.gnuplot(0/len(B)))
    plt.plot(t,y1,label="y(t)", color=plt.cm.gnuplot(1/len(B)))
    plt.plot(t,w,label="z(t)", color=plt.cm.gnuplot(2/len(B)))
    plt.xlabel("t")
    plt.legend()
    plt.grid()
    plt.show()
    plt.title("Absolutna razlika rešitev pri $b=%.2f$ in $A/A_0(0)$=1" %(b))
    plt.plot(t,abs(x-x1),label="x-x_stac", color=plt.cm.gnuplot(0/len(B)))
    plt.plot(t,abs(w-w1),label="z-z_stac", color=plt.cm.gnuplot(2/len(B)))
    plt.xlabel("t")
    plt.legend()
    plt.grid()
    plt.show()
    plt.title("Absolutna razlika rešitev pri $b=%.2f$ in $A/A_0(0)$=1" %(b))
    plt.plot(t,abs(x-x1),label="x-x_stac", color=plt.cm.gnuplot(0/len(B)))
    plt.plot(t,abs(w-w1),label="z-z_stac", color=plt.cm.gnuplot(2/len(B)))
    plt.xlabel("t")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

fig2,(ax_1,ax_3) = plt.subplots(2)
fig,(ax,ax2,ax3) = plt.subplots(3)
fig3, (ax_4)=plt.subplots(1)
c = np.arange(1, n_lines + 1)
i=0.1
while i<b_max:
    B_tixcs.append(i)
    i+=0.1
B_tixcs=np.array(B_tixcs)


norm = mpl.colors.Normalize(vmin=B_tixcs.min(), vmax=B_tixcs.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.gnuplot)
cmap.set_array([])




index=0
fig.suptitle("Graf odvisnosti spojin od izbire parametra $b$ z $a=1000$ in $A/A_0(0)$=1")
fig3.suptitle("Graf odvisnosti spojin od izbire parametra $b$ z $a=1000$ in $A/A_0(0)$=1")
for b in B:
    z0=[x0,w0]
    x,w=odeint(model_stac,z0,t,args=(b,)).T
    z0=[x0,y0,w0]
    x1,y1,w1=odeint(model_original,z0,t,args=(b,)).T
    ax.plot(t,x1,label="b=%.2f"%(b),c=cmap.to_rgba(index/len(B)))
    ax_4.plot(t,x1,"-", label="b=%.2f"%(b),c=cmap.to_rgba(index/len(B)))
    ax.set_title(r"Prikaz spojine $A/A_0$")
    ax2.plot(t,y1,c=cmap.to_rgba(index/len(B)))
    #ax_4.plot(t,y1,"-",c=cmap.to_rgba(index/len(B)))
    ax2.set_title(r"Prikaz spojine $A*/A_0$")
    ax3.set_title(r"Prikaz spojine $B/A_0$")
    ax3.plot(t,w1,c=cmap.to_rgba(index/len(B)))
    ax_4.plot(t,w1,"--",c=cmap.to_rgba(index/len(B)))
    ax_1.plot(t,abs(x-x1),label="b=%.2f"%(b),c=cmap.to_rgba(index/len(B)))
    ax_1.set_title(r"Prikaz razlike med metodama za  $A/A_0$")
    ax_3.set_title(r"Prikaz razlike med metodama za  $B/A_0$")
    ax_3.plot(t,abs(w-w1),c=cmap.to_rgba(index/len(B)))
    index+=1
ax_1.legend()
ax.legend()
ax_4.legend()
plt.show()


fig2,(ax_1,ax_3) = plt.subplots(2)
fig,(ax,ax2,ax3) = plt.subplots(3)
fig3, (ax_4)=plt.subplots(1)

index=0
fig.suptitle("Graf odvisnosti spojin od izbire parametra $b$ z $a=1000$ in $A/A_0(0)$=1")
fig3.suptitle("Graf odvisnosti spojin od izbire parametra $b$ z $a=1000$ in $A/A_0(0)$=1")
for b in B:
    z0=[x0,w0]
    x,w=odeint(model_stac,z0,t,args=(b,)).T
    z0=[x0,y0,w0]
    x1,y1,w1=odeint(model_original,z0,t,args=(b,)).T
    ax.plot(t,x1,label="b=%.2f"%(b),c=cmap.to_rgba(index/len(B)))
    ax_4.plot(t,x1,"-", label="b=%.2f"%(b),c=cmap.to_rgba(index/len(B)))
    ax.set_title(r"Prikaz spojine $A/A_0$")
    ax2.plot(t,y1,c=cmap.to_rgba(index/len(B)))
    #ax_4.plot(t,y1,"-",c=cmap.to_rgba(index/len(B)))
    ax2.set_title(r"Prikaz spojine $A*/A_0$")
    ax3.set_title(r"Prikaz spojine $B/A_0$")
    ax3.plot(t,w1,c=cmap.to_rgba(index/len(B)))
    ax_4.plot(t,w1,"--",c=cmap.to_rgba(index/len(B)))
    ax_1.plot(t,abs(x-x1),label="b=%.2f"%(b),c=cmap.to_rgba(index/len(B)))
    ax_1.set_title(r"Prikaz razlike med metodama za  $A/A_0$")
    ax_3.set_title(r"Prikaz razlike med metodama za  $B/A_0$")
    ax_3.plot(t,abs(w-w1),c=cmap.to_rgba(index/len(B)))
    index+=1
ax_1.legend()
ax_1.set_yscale('log')
ax_3.set_yscale('log')
ax.legend()
ax_4.legend()
plt.show()

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dz/dt
n = 2500 #stevilo korakov integracije
l = 30 #čas do kamor simuliramo
a = 1000
b = 10
def model(z,t,b):
    x = z[0]
    y = z[1]
    w = z[2]
    dxdt = -x*(x-a*y)
    dydt = +x*(x-a*y)-a*b*y
    dwdt = a*b*y
    dzdt = [dxdt,dydt,dwdt]
    return dzdt

# initial condition
x0 = 1
y0 = 0
w0 = 0
z0 = [x0,y0,w0]

def model2(z,t,b):
    x = z[0]
    y = z[1]
    dxdt = -(b*x**2)/(b+x)
    dydt = (b*x**2)/(b+x)
    dzdt = [dxdt,dydt]
    return dzdt

# initial condition
x0 = 1
y0 = 0
z20 = [x0,y0]


# time points
t = np.linspace(0,l,n)


b=10
# solve ODE
z = odeint(model,z0,t,args=(b,))
z2 = odeint(model2,z20,t,args=(b,))
plt.plot(t,np.abs(z[:,0] - z2[:,0]),'r', label='x')
plt.plot(t,np.abs(z[:,2] - z2[:,1]),'b', label='z')
plt.plot(t,np.abs(z[:,1] - z2[:,0]**2/(a*(z2[:,0] + b))),'g', label='y')

plt.yscale("log")

plt.ylabel('|i - i_s|')
plt.text(l/2-10, 2*10**(-5), "a = {}, b = {}".format(a, b))
plt.xlabel('čas')
plt.legend(loc='best')

plt.show()



b=1
z = odeint(model,z0,t,args=(b,))
z2 = odeint(model2,z20,t,args=(b,))
plt.plot(t,np.abs(z[:,0] - z2[:,0]),'r', label='x')
plt.plot(t,np.abs(z[:,2] - z2[:,1]),'b', label='z')
plt.plot(t,np.abs(z[:,1] - z2[:,0]**2/(a*(z2[:,0] + b))),'g', label='y')

plt.yscale("log")

plt.ylabel('|i - i_s|')
plt.text(l/2-10, 2*10**(-4), "a = {}, b = {}".format(a, b))
plt.xlabel('čas')
plt.legend(loc='best')

plt.show()


b=0.1
z = odeint(model,z0,t,args=(b,))
z2 = odeint(model2,z20,t,args=(b,))
plt.plot(t,np.abs(z[:,0] - z2[:,0]),'r', label='x')
plt.plot(t,np.abs(z[:,2] - z2[:,1]),'b', label='z')
plt.plot(t,np.abs(z[:,1] - z2[:,0]**2/(a*(z2[:,0] + b))),'g', label='y')

plt.yscale("log")

plt.ylabel('|i - i_s|')
plt.text(l/2-10, 10**(-3), "a = {}, b = {}".format(a, b))
plt.xlabel('čas')
plt.legend(loc='best')

plt.show()



# time points



plt.show()