
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from tqdm import tqdm

# function that returns dz/dt
n = 5000 #stevilo korakov integracije
l = 25 #čas do kamor simuliramo

p = 1
q = 1
r = 0.2
s = 1
m = 2.5
t=m*s
k=2*r*m*np.sqrt(p/q)
print(k,m)
print(2*m*r)
def model(w,ti):
    u = w[0]
    v = w[1]
    x = w[2]

    dudt = -k/2*u*np.sqrt(v)/(m + x/v) #
    dvdt = -k/2*u*np.sqrt(v)/(m + x/v)
    dxdt = k*u*np.sqrt(v)/(m+x/v)
    

    dwdt = [dudt, dvdt, dxdt]
    return dwdt

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
# initial condition
u0 = 1  #H2
v0 = 1 #Br2
x0 = 0  #HBr

y0 = 0  #H
z0 = 0  #Br

w0 = [u0, v0, x0]
w01 = [u0, v0, x0, y0, z0]
# time points
ti = np.linspace(0.01,l,n)

"""
# solve ODE
w = odeint(model,w0,ti)

#y0 = np.sqrt(p*w[0,1]/q)*r*w[0,0]/(s*w[0,2]+t*w[0,1])
#z0 = np.sqrt(p*w[0,1]/q)
# plot results
plt.title("Prikaz reakcije; $p=%.2f$, $q=%.2f$, $r=%.2f$, $s=%.2f$, $t=%.2f$" %(p,q,r,s,t))
plt.plot(ti,w[:,0],label='H2_0 = %.1f' %(u0), color="r")
plt.plot(ti,w[:,1],"--",label=' Br2_0 = %.0f' %(v0), color="b")
plt.plot(ti,w[:,2],label='HBr_0 = %.0f' %(x0), color="g")
plt.plot(ti,np.sqrt(p*w[:,1]/q)*r*w[:,0]/(s*w[:,2]+t*w[:,1]),'k',label='H, H_0 = %.0f' %(y0),color=plt.cm.gist_rainbow(3/len(w01)))
plt.plot(ti,np.sqrt(p*w[:,1]/q),'orange',label='Br, Br_0 = %.0f' %(z0),color=plt.cm.gist_rainbow(5/len(w01)))

plt.ylabel('Koncentracija')
plt.xlabel('čas')
plt.grid()
plt.legend(loc='upper right')
plt.show()

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
w01 = [u0, v0, x0, y0, z0]    
w1 = odeint(model,w01,ti)
# plot results
plt.title("Prikaz reakcije; $p=%.2f$, $q=%.2f$, $r=%.2f$, $s=%.2f$, $t=%.2f$" %(p,q,r,s,t))
plt.plot(ti,w1[:,0],label='H2_0 = %.1f' %(u0), color="r")
plt.plot(ti,w1[:,1],"--",label=' Br2_0 = %.1f' %(v0), color="b")
plt.plot(ti,w1[:,2],label='HBr_0 = %.1f' %(x0), color="g")
plt.plot(ti,w1[:,3],label='H_0 = %.0f' %(y0), color=plt.cm.gist_rainbow(3/len(w01)))
plt.plot(ti,w1[:,4],label='Br_0 = %.0f' %(z0), color=plt.cm.gist_rainbow(5/len(w01)))


plt.ylabel('Koncentracija')
plt.xlabel('čas')
plt.grid()
plt.legend(loc='upper right')

plt.show()

plt.title("Prikaz razlike; $p=%.2f$, $q=%.2f$, $r=%.2f$, $s=%.2f$, $t=%.2f$" %(p,q,r,s,t))
plt.plot(ti,abs(w[:,0]-w1[:,0]),label='H2_0 = %.1f' %(u0), color="r")
plt.plot(ti,abs(w[:,1]-w1[:,1]),"--",label=' Br2_0 = %.1f' %(v0), color="b")
plt.plot(ti,abs(w[:,2]-w1[:,2]),label='HBr_0 = %.1f' %(x0), color="g")
plt.plot(ti,abs(np.sqrt(p*w[:,1]/q)*r*w[:,0]/(s*w[:,2]+t*w[:,1])-w1[:,3]),label='H_0 = %.0f' %(y0), color=plt.cm.gist_rainbow(3/len(w01)))
#plt.plot(ti,abs(np.sqrt(p*w[:,1]/q)-w1[:,4]),label='Br_0 = %.0f' %(z0), color=plt.cm.gist_rainbow(5/len(w01)))


plt.ylabel('Koncentracija')
plt.xlabel('čas')
plt.grid()
plt.legend(loc='upper right')

plt.show()


plt.title("Prikaz razlike log; $p=%.2f$, $q=%.2f$, $r=%.2f$, $s=%.2f$, $t=%.2f$" %(p,q,r,s,t))
plt.plot(ti,abs(w[:,0]-w1[:,0]),label='H2_0 = %.1f' %(u0), color="r")
plt.plot(ti,abs(w[:,1]-w1[:,1]),"--",label=' Br2_0 = %.1f' %(v0), color="b")
plt.plot(ti,abs(w[:,2]-w1[:,2]),label='HBr_0 = %.1f' %(x0), color="g")
plt.plot(ti,abs(np.sqrt(p*w[:,1]/q)*r*w[:,0]/(s*w[:,2]+t*w[:,1])-w1[:,3]),label='H_0 = %.0f' %(y0), color=plt.cm.gist_rainbow(3/len(w01)))
#plt.plot(ti,abs(np.sqrt(p*w[:,1]/q)-w1[:,4]),label='Br_0 = %.0f' %(z0), color=plt.cm.gist_rainbow(5/len(w01)))
print(w[:,0][0])

plt.ylabel('Koncentracija')
plt.xlabel('čas')
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.legend(loc='lower left')

plt.show()

HBR=np.linspace(0,10,10)
H2=[0.1,1,10]
plt.title("Prikaz odvisnosti od HBr; $p=%.2f$, $q=%.2f$, $r=%.2f$, $s=%.2f$, $t=%.2f$" %(p,q,r,s,t))
for i in tqdm(H2):
    HBr_end=[]
    for j in HBR:
        if i==10:
            w01 = [u0, i, j, y0, z0]
            w1 = odeint(model,w01,ti)
        else:
            w01 = [i, v0, j, y0, z0]
            w1 = odeint(model,w01,ti)
        HBr_end.append(w1[:,2][len(w1[:,2])-1]-j)
    plt.plot(HBR,HBr_end, label=r"[H$_2$]/[Br$_2$]=$%.2f$" %(i))
plt.legend()
plt.grid()
plt.xlabel("HBr[0]")
plt.ylabel(r"HBr$_{končna}$/HBr[0]")
plt.show()

HBR=np.linspace(0,1,50)
H2=np.linspace(0,1,10)
plt.title("Prikaz odvisnosti od HBr; $p=%.2f$, $q=%.2f$, $r=%.2f$, $s=%.2f$, $t=%.2f$" %(p,q,r,s,t))
index=0
for i in tqdm(H2):
    HBr_end=[]
    for j in HBR:
        print(j)
        w1=0
        if i>=1:
            w01 = [u0, i, j, y0, z0]
            w1 = odeint(model,w01,ti)
        else:
            w01 = [i, v0, j, y0, z0]
            w1 = odeint(model,w01,ti)
        print(w1[:,2][len(w1[:,2])-1])
        HBr_end.append(w1[:,2][len(w1[:,2])-1])
    plt.plot(HBR,HBr_end, label=r"[H$_2$]/[Br$_2$]=$%.2f$" %(i), color=plt.cm.gist_rainbow(index/len(H2)))
    index+=1
plt.legend()
plt.grid()
plt.xlabel("HBr[0]")
plt.ylabel(r"HBr$_{končna}$-HBr[0]")
plt.show()"""
"""
H2=np.linspace(0,10,10)
plt.title("Prikaz odvisnosti od HBr; $p=%.2f$, $q=%.2f$, $r=%.2f$, $s=%.2f$, $t=%.2f$" %(p,q,r,s,t))
index=0
HBr_end=[]
for i in tqdm(H2):
    w1=0
    if i>=1:
        w01 = [u0, v0, i, y0, z0]
        w1 = odeint(model,w01,ti)
    else:
        w01 = [u0, v0, i, y0, z0]
        w1 = odeint(model,w01,ti)
    plt.plot(ti,w1[:,2]-i,label=r"$HBr_[0]=%.2f$" %(i),color=plt.cm.gist_rainbow(index/len(H2)))
    #print(w1[:,2][len(w1[:,2])-1])
    HBr_end.append(w1[:,2][len(w1[:,2])-1])
    index+=1
#plt.plot(H2, HBr_end,label=r"[HBr(0)=$%.2f$" %(x0), color=plt.cm.gist_rainbow(index/len(H2)))
plt.legend()
plt.grid()
plt.xlabel("[H$_2$]/[Br$_2$]")
plt.ylabel(r"HBr$_{končna}-HBr_{začetna}$")
plt.show()


from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot
w01 = [u0, v0, x0, y0, z0]
w1 = odeint(model,w01,ti)
u, v ,x,y,z = w1[:,0], w1[:,1], w1[:,2],w1[:,3],w1[:,4]
dxdy= r*u*z + t*v*y - s*x*y
# define the true objective function
def objective(X,k,m):
    u,v,x=X
    return  k*u*np.sqrt(v)/(m+x/v)
 
# load the dataset
# choose the input and output variables
m0 = 2.5
t=m*s
k0=2*r*m*np.sqrt(p/q)
# curve fit
p0=[k,m]
popt, _ = curve_fit(objective, (u,v,x), dxdy,p0)
# summarize the parameter values
m, k = popt
print('y = %.5f * x + %.5f' % (k, m))
# plot input vs output
# define a sequence of inputs between the smallest and largest known inputs
# create a line plot for the mapping function
y_line=k*u*np.sqrt(v)/(m+x/v)
pyplot.plot(ti, k0*u*np.sqrt(v)/(m0+x/v), '-', label="Eksaktna $m=%.2f, k=%.2f$" %(m0,k0),color='red')
pyplot.plot(ti, y_line, '--',label="Približek $m=%.2f, k=%.2f$" %(m,k), color='red')
plt.legend()
plt.grid()
plt.xlabel("t")
plt.ylabel(r"HBr$_{končna}-HBr_{začetna}$")
pyplot.show()
time=[100,1000,5000,10000,20000]
s = 1
m=2.5
index=0
t=m*s
r=1
k0=2*r*m*np.sqrt(p/q)
w01 = [u0, v0, x0, y0, z0]
w1 = odeint(model,w01,ti)
u1, v1 ,x1,y1,z1 = w1[:,0], w1[:,1], w1[:,2],w1[:,3],w1[:,4]
dxdy= r*u1*z1 + t*v1*y1 - s*x1*y1
plt.title("Prikaz prilagajanja za x; $p=%.2f$, $q=%.2f$, $r=%.2f$, $s=%.2f$, $t=%.2f$" %(p,q,r,s,t))
for i in time:
    ti = np.linspace(0,l,n)
    m0 = 2.5
    w01 = [u0, v0, x0, y0, z0]
    w1 = odeint(model,w01,ti)
    u, v ,x,y,z = w1[:,0], w1[:,1], w1[:,2],w1[:,3],w1[:,4]
    u,v,x,y,z=u[i:len(time)-i], v[i:len(time)-i] ,x[i:len(time)-i],y[i:len(time)-i],z[i:len(time)-i]
    s1 = 1
    t=m*s1
    r1=0.2
    k0=2*r1*m*np.sqrt(p/q)
# curve fit
    p0=[m0,k0]
    popt, _ = curve_fit(objective, (u,v,x), dxdy[i:len(time)-i],p0)
# summarize the parameter values
    m, k = popt
    print('y = %.5f * x + %.5f' % (k, m))
# plot input vs output
# define a sequence of inputs between the smallest and largest known inputs
# create a line plot for the mapping function
    y_line=k*u*np.sqrt(v)/(m+x/v)
    pyplot.plot(ti[i:len(time)-i], y_line,label="Približek $m=%.2f, k=%.2f$ za t=$%.1f$" %(m,k,i), color=plt.cm.gist_rainbow(index/len(time)))
    if index==len(time)-1:
        pyplot.plot(ti, k0*u1*np.sqrt(v1)/(m0+x1/v1), '-', label="Eksaktna $m=%.2f, k=%.2f$" %(m0,k0),color="black")
    plt.legend()
    plt.grid()
    plt.xlabel("t")
    plt.xlim(-0.5,25.5)
    plt.ylabel(r"x$")
    index+=1
pyplot.show()


time=[20,30,40,50,60,70,80,90,100]
s = 1
m=2.5
index=0
t=m*s
r=1
k0=2*r*m*np.sqrt(p/q)
w01 = [u0, v0, x0, y0, z0]
w1 = odeint(model,w01,ti)
u1, v1 ,x1,y1,z1 = w1[:,0], w1[:,1], w1[:,2],w1[:,3],w1[:,4]
dxdy= r*u1*z1 + t*v1*y1 - s*x1*y1
plt.title("Prikaz prilagajanja za x; $p=%.2f$, $q=%.2f$, $r=%.2f$, $s=%.2f$, $t=%.2f$" %(p,q,r,s,t))
for i in time:
    ti = np.linspace(0,l,n)
    m0 = 2.5
    w01 = [u0, v0, x0, y0, z0]
    w1 = odeint(model,w01,ti)
    u, v ,x,y,z = w1[:,0], w1[:,1], w1[:,2],w1[:,3],w1[:,4]
    s1 = 1
    t=m*s1
    r1=0.2
    k0=2*r1*m*np.sqrt(p/q)
# curve fit
    p0=[m0,k0]
    popt, _ = curve_fit(objective, (u,v,x), dxdy,p0)
# summarize the parameter values
    m, k = popt
    print('y = %.5f * x + %.5f' % (k, m))
# plot input vs output
# define a sequence of inputs between the smallest and largest known inputs
# create a line plot for the mapping function
    y_line=k*u*np.sqrt(v)/(m+x/v)
    pyplot.plot(ti, y_line,label="Približek $m=%.2f, k=%.2f$ za t=$%.1f$" %(m,k,i), color=plt.cm.gist_rainbow(index/len(time)))
    if index==len(time)-1:
        pyplot.plot(ti, k0*u1*np.sqrt(v1)/(m0+x1/v1), '-', label="Eksaktna $m=%.2f, k=%.2f$" %(m0,k0),color="black")
    plt.legend()
    plt.grid()
    plt.xlabel("t")
    plt.xlim(-0.5,25.5)
    plt.ylabel(r"x$")
    index+=1
pyplot.show()
"""


from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot
s = 1
m=2.5
m0=m
index=0
t=m*s
r=1
k0=2*r*m*np.sqrt(p/q)
w01 = [u0, v0, x0, y0, z0]
w1 = odeint(model,w01,ti)
u1, v1 ,x1,y1,z1 = w1[:,0], w1[:,1], w1[:,2],w1[:,3],w1[:,4]
x_o=[]
for i in range(len(x1)):
    if i!=0:
        x_o.append((x1[i]-x1[i-1])/(ti[i]-ti[i-1]))
    if i==0:
        x_o.append((x1[i]/ti[i]))
dxdy= k*u1*v1**(1/2)/x_o-m
x_o=np.array(x_o)

def model(w,ti):
    u = w[0]
    v = w[1]
    x = w[2]

    dudt = -k/2*u*np.sqrt(v)/(m + x/v) #
    dvdt = -k/2*u*np.sqrt(v)/(m + x/v)
    dxdt = k*u*np.sqrt(v)/(m+x/v)
    

    dwdt = [dudt, dvdt, dxdt]
    return dwdt

def premica(X,k,m):
    u,v,x_o=X
    return  k*u*v**(1/2)/x_o-m

plt.title("Prikaz prilagajanja za x; $p=%.2f$, $q=%.2f$, $r=%.2f$, $s=%.2f$, $t=%.2f$" %(p,q,r,s,t))
x_o=[0]
w01 = [u0, v0, x0]
w1 = odeint(model,w01,ti)
u1, v1 ,x1 = w1[:,0], w1[:,1], w1[:,2]
p0=[m0,k0]
u1,v1,x1,x_o,dxdy=u1[2:len(u1)-1], v1[2:len(u1)-1], x1[2:len(u1)-1], x_o[2:len(u1)-1], dxdy[2:len(u1)-1]
x_o=np.array(x_o)
u1=np.array(u1)
v1=np.array(v1)
dxdy=np.array(dxdy)
popt, _ = curve_fit(premica, (u1,v1,x_o), dxdy,p0)
m, k = popt
y_line=k[2:len(u1)-1]*u1[2:len(u1)-1]*v1[2:len(u1)-1]**(1/2)/x_o[2:len(u1)-1]-m
pyplot.plot(ti[2:len(u1)-1], y_line,label="Približek $m=%.2f, k=%.2f$ za t=$%.1f$" %(m,k,i), color=plt.cm.gist_rainbow(index/len(x1)))
plt.legend()
plt.grid()
plt.xlabel("t")
plt.xlim(-0.5,25.5)
plt.ylabel(r"x$")
index+=1
pyplot.show()