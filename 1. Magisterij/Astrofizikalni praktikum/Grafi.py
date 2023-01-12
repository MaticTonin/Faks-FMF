
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.linalg as lin
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.optimize as sp




THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data_Milky= np.loadtxt(THIS_FOLDER + "\\Milky.txt",delimiter="\t", dtype=str)
print(data_Milky[0])
R, delta_R, vc, delta_vc, wc, delta_wc, Reference, dodatek=data_Milky.T
R=np.array(R,dtype=float)
vc=np.array(vc,dtype=float)
wc=np.array(wc,dtype=float)
data = np.loadtxt(THIS_FOLDER + "\\NGC.txt", dtype=str)
#plt.plot(R,vc,"x")
plt.plot(R,vc,"x")
plt.show()

plt.plot(vc,wc,"x")
plt.show()
M=6.5*10**10*2*10**30 #Mas sonca 
kilo_parsec=3.086*10**19 # to m
a=4*kilo_parsec
G=6.6*10**(-11)
def making_sigma(r):
    sigma=M/(2*np.pi*(a)**2)*np.exp(-r/a)
    return sigma
def making_dm(r):
    sigma=making_sigma(r)
    dm=2*np.pi*r*sigma
    return dm
def ending_f(r):
    me=-2*M*np.exp(-r/a)/a
    return me
radius=np.linspace(0,20,200)

delta_r=0.01*kilo_parsec
masa=[0]
radius=0.01*kilo_parsec
radius_linspace=[0]
velocity=[0]
hitrost_gosta=[0]
for i in range(2779):
    if i!=0:
        dm=making_dm(radius)
        masa.append((masa[i-1]+dm*delta_r))
        velocity.append(np.sqrt(G*masa[i]/radius)/1000)
        hitrost_gosta.append(np.sqrt(G*M/radius)/1000)
        radius+=delta_r
        radius_linspace.append(radius/kilo_parsec)
plt.plot(radius_linspace,masa)
plt.title("Masa")
plt.show()


f = lambda r, M, a : np.sqrt(2*np.pi*r*M*G*M/(2*np.pi*r*(a)**2)*np.exp(-r/a))
args = [M, a]
fit = sp.curve_fit(f, radius_linspace, vc, p0=args)
plt.plot(radius_linspace,f(radius_linspace, fit[0][0], fit[0][1]), label="%.5f, %.5f" %(fit[0][0]/10**30,fit[0][1]/kilo_parsec))
plt.plot(R,vc,"x")
plt.plot(radius_linspace,hitrost_gosta,label="inf")
plt.plot(radius_linspace,velocity, label="konst")
plt.title("Hitrost")
plt.legend()
plt.show()





plt.plot(radius,making_dm(radius)*radius)
plt.show()

plt.plot(radius,ending_f(radius))
plt.show()
mass=making_dm(radius)
v=[]
for i in range(len(radius)):
    v.append(np.sqrt(G*mass[i]))
plt.plot(radius,v)
plt.show()
    
