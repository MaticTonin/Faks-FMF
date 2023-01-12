import numpy as np
import matplotlib.pyplot as plt

n = 0   
nu= 0.01
nu_list = np.logspace(-3,3, 80)
R = 1
N = 10**4
x=[]
y=[]
theta = np.linspace( 0 , 2 * np.pi , 150 )
a = R * np.cos( theta )
b = R * np.sin( theta )
color=[]
x_generated=[]
y_generated=[]
N_list=[]
def generating_data(N,nu):
    n = 0   
    color=[]
    x=[]
    y=[]
    x_generated=[]
    y_generated=[]
    N_list=[]
    for i in range(N):
        N_list.append(i+1)
        radij=np.random.rand()
        r_i = (radij)**(1/3)
        psi= 2*np.pi*np.random.rand()
        predznak =np.random.rand()*2-1
        theta_i = 2 * np.random.rand() - 1
        #s = -nu * np.log(np.random.rand())
        s=nu
        d = - r_i * theta_i + R * np.sqrt(1 - (1 - theta_i) * (r_i/R)**2)
        x_generated.append(predznak*r_i*np.cos(psi))
        y_generated.append(predznak*r_i*np.sin(psi))
        if s >= d:
            print(d)
            n = n + 1
            theta=np.arcsin(radij/R)
            x.append(predznak*radij*np.cos(psi))
            y.append(predznak*radij*np.sin(psi))
            color.append(i)
    return x,y,x_generated,y_generated,color,N_list,n

x=[]
y=[]
x_generated=[]
y_generated=[]
for i in range(N):
    N_list.append(i+1)
    r_i = R*(np.random.rand())**(1/3)
    predznak =np.random.rand()*2-1
    theta_i = 2 * np.random.rand() - 1
    psi= 2*np.pi*np.random.rand()
    s = -nu * np.log(np.random.rand())
    d = - r_i * theta_i + R * np.sqrt(1 - (1 - theta_i**2) * (r_i/R)**2)
    x_generated.append(predznak*r_i*np.cos(psi))
    y_generated.append(predznak*r_i*np.sin(psi))
    if s >= d:
        n = n + 1
        x.append(predznak*r_i*np.cos(psi))
        y.append(predznak*r_i*np.sin(psi))
        color.append(i)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(a,b, color="Black")
ax.set_title("Prikaz generiranih podatkov")
p=ax.scatter(x_generated,y_generated,c=N_list, cmap="rainbow")
cbar=fig.colorbar(p)
cbar.set_label('N', rotation=270)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(a,b, color="Black")
ax.set_title(r"Prikaz pobeglih žarkov za $\nu=%.3f$" %(nu))
p=ax.scatter(x,y,c=color, cmap="rainbow")
cbar=fig.colorbar(p)
cbar.set_label('N', rotation=270)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
plt.show()


nu_list=np.linspace(0.01,1,8)
for i in nu_list:
    x,y,x_generated,y_generated,color,N_list,n=generating_data(N,i)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(a,b, color="Black")
    ax.set_title(r"Prikaz pobeglih žarkov za $\nu=%.3f$" %(i))
    p=ax.scatter(x,y,c=color, cmap="rainbow")
    cbar=fig.colorbar(p)
    cbar.set_label('N', rotation=270)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    plt.show()

nu_list = np.logspace(-3,3, 80)
for nu in nu_list:
    n = 0
    for i in range(N):
        r_i = (np.random.rand())**(1/3)
        theta_i = np.arccos(2 * np.random.rand() - 1)
        s = -nu * np.log(np.random.rand())
        d = - r_i * theta_i + R * np.sqrt(1 - (1 - theta_i**2) * (r_i/R)**2)
        if s >= d:
            n = n + 1   
    plt.scatter(nu, n/N, color = 'r', s = 8 )
plt.title("Prika verjetnosti za pobeg")
plt.xlabel(r'$\mu$/R')
plt.ylabel('$P=n/N$')
plt.xscale('log')
plt.grid(which="both", ls = '--')

plt.show()