import numpy as np
import numpy.linalg as lin
import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import time


def matrix_A(N):
    A=np.zeros((N,N))
    for i in range(1,N-1):
        A[i,i-1]=1
        A[i,i]=-2
        A[i,i+1]=1
    #Ker more biti matrika v vsakem primeru tridiagonalna za neskončno jamo
    A[0,0] = -2
    A[0, 1] = 1

    A[N-1, N-2] = 1
    A[N-1, N-1] = -2
    return A


a=-np.pi/2
b=np.pi/2
y_a=0
y_b=0
N=1000
h=(b-a)/N

def derivate(icons, lambd):
    y, v= icons
    value=-lambd*y
    return value

from scipy.integrate import solve_ivp

def Strelska(f, a, b, lambd):
    def dydt_y(t,icons):
        v=icons[1]
        return np.array([v, f(icons,lambd)])
    
    sol = solve_ivp(dydt_y, (a, b), y0=np.array([0, 1]), rtol = 1e-8, atol = 1e-10)
    return sol.y[0,-1]

def bisekcija(f,a,b, eps):
    levi = f(a)
    desni =f(b)
    x_mid= (b+a)/2
    sredica= f(x_mid)
    while b-a > eps:
        if levi * desni ==0: break

        elif levi* sredica <0:
            b=x_mid
            x_mid=(a+b)/2
            desni=sredica
            sredica=f(x_mid)
        
        elif desni * sredica <0:
            a=x_mid
            x_mid=(a+b)/2
            levi=sredica
            sredica=f(x_mid)

    return np.array([a,b]) #interval ničle

def bisekcija_funkcija_lambda(f, a, b):
    def fun(lamb):
        return Strelska(f,a,b,lamb)
    return fun

izd_fun= bisekcija_funkcija_lambda(derivate, a, b)
lamb=[]
while (1):
    lamb.append(bisekcija(izd_fun, 0.1, 2, 1e-10)[0])
    lamb.append(bisekcija(izd_fun, 3.5, 4.5, 1e-10)[0])
    lamb.append(bisekcija(izd_fun, 8.5, 9.5, 1e-10)[0])
    lamb.append(bisekcija(izd_fun, 15.1, 16.5, 1e-10)[0])
    lamb.append(bisekcija(izd_fun, 24.5, 25.5, 1e-10)[0])
    lamb.append(bisekcija(izd_fun, 35.5, 36.5, 1e-10)[0])
    lamb.append(bisekcija(izd_fun, 47.5, 49.5, 1e-10)[0])
    break
print(lamb)
def dydt_y(t,icons):
    v=icons[1]
    return np.array([v, derivate(icons,lamb[0])])
sol = solve_ivp(dydt_y, (a, b), y0=np.array([0, 1]), rtol = 1e-8, atol = 1e-10)

def Error(Y,value):
    Error=[]
    for i in range(len(Y)):
        Error.append(abs(Y[i]-value[i]))
    return Error

import matplotlib.pyplot as plt
lamb_list=np.linspace(0.5,100,10)
y=[Strelska(derivate, a, b, lamb) for lamb in lamb_list]
y=y/max(y)
nicle=[]
y_nic=[]
for i in range(len(y)):
    if y[i]<=10**(-16):
        y_nic.append(y[i])
        nicle.append(lamb_list[i])
plt.plot(lamb_list,y )
plt.title(r'Prikaz iskanja ničel s Strelsko metodo')
plt.plot(1,0, ".", color="red")
plt.text(1,0.05,r'$\lambda_1=1$')
plt.plot(4,0, ".", color="red")
plt.text(4,-0.05,r'$\lambda_2=4$')
plt.plot(9,0, ".", color="red")
plt.text(9,0.05,r'$\lambda_3=9$')
plt.plot(16,0, ".", color="red")
plt.text(16,-0.05,r'$\lambda_4=16$')
plt.plot(24.99,0, ".", color="red")
plt.text(24.99,0,r'$\lambda_5=24.99$')
plt.plot(35.99,0, ".", color="red")
plt.text(35.99,0,r'$\lambda_6=35.99$')
plt.plot(48.99,0, ".", color="red")
plt.text(48.99,0,r'$\lambda_7=48.99$')
plt.plot(63.99,0, ".", color="red")
plt.text(63.99,0,r'$\lambda_8=63.99$')
plt.plot(80.99,0, ".", color="red")
plt.text(80.99,0,r'$\lambda_9=80.99$')
plt.plot(99.99,0, ".", color="red")
plt.text(97.99,0.01,r'$\lambda_{10}=99.99$')

plt.xlabel(r"$\lambda$")
plt.ylabel(r"$y_\lambda(b)-y(b)$ ")
plt.legend()
plt.show()
labdaa=[1,4,9,16,25,36,49]
x=np.linspace(a,b,N)
n=1
psi_main4=np.cos(n*x)

def dydt_y(t,icons):
    v=icons[1]
    return np.array([v, derivate(icons,lamb[0])])

sol = solve_ivp(dydt_y, (a, b), y0=np.array([0, 1]), rtol = 1e-8, atol = 1e-10)
plt.plot(sol.t, sol.y[0]/max(sol.y[0]), label=r"$\psi$")
plt.plot(x,psi_main4,'-', label=r"$\psi$"+str(n))
plt.title(r'Prikaz približkov Strelsko metodo za $\psi(x)_0$')
plt.xlabel("x")
plt.ylabel(r"$\psi (x)$ ")
plt.legend()
plt.show()

plt.plot(sol.t, np.log(Error(sol.y[0]/max(sol.y[0]),psi_main4)), label=r"$\psi$")
plt.title(r'Prikaz napak Strelsko metodo za $\psi(x)_0$')
plt.xlabel("x")
plt.ylabel(r"$\psi (x)$ ")
plt.legend()
plt.show()
n=1
while n<=6:
    if n%2==1:
        psi_main4=np.cos((n)*x)
        def dydt_y(t,icons):
            v=icons[1]
            return np.array([v, derivate(icons,labdaa[n-1])])
        sol = solve_ivp(dydt_y, (a, b), y0=np.array([0, 1]), rtol = 1e-8, atol = 1e-10)
        plt.plot(sol.t,sol.y[0]/max(sol.y[0])+labdaa[n-1], label=r"$\psi$"+str(n-1))
    if n%2==0:
        psi_main4=np.sin((n)*x)
        def dydt_y(t,icons):
            v=icons[1]
            return np.array([v, derivate(icons,labdaa[n-1])])
        sol = solve_ivp(dydt_y, (a, b), y0=np.array([0, 1]), rtol = 1e-8, atol = 1e-10)
        plt.plot(sol.t,sol.y[0]/max(sol.y[0])+labdaa[n-1], label=r"$\psi$"+str(n-1))
    n+=1
plt.title(r'Prikaz približkov Strelsko metodo za $\psi(x)$')
plt.xlabel("x")
plt.ylabel(r"$\psi (x)$ ")
plt.legend()
plt.show()



n=1
while n<=6:
    if n%2==1:
        psi_main0=np.sin((n)*x+np.pi/2)
        def dydt_y(t,icons):
            v=icons[1]
            return np.array([v, derivate(icons,labdaa[n-1])])
        sol = solve_ivp(dydt_y, (a, b), y0=np.array([0, 1]), rtol = 1e-8, atol = 1e-10)
        plt.plot(sol.t, np.log(Error(sol.y[0]/max(sol.y[0]),psi_main0)), label=r"$\psi$"+str(n-1))
    if n%2==0:
        psi_main4=np.sin((n)*x)
        def dydt_y(t,icons):
            v=icons[1]
            return np.array([v, derivate(icons,labdaa[n-1])])
        sol = solve_ivp(dydt_y, (a, b), y0=np.array([0, 1]), rtol = 1e-8, atol = 1e-10)
        plt.plot(sol.t, np.log(Error(sol.y[0]/max(sol.y[0]),psi_main4)), label=r"$\psi$"+str(n-1))
    n+=1
plt.title(r'Prikaz napak približkov Strelsko metodo za $\psi(x)$')
plt.xlabel("x")
plt.ylabel(r"$\psi (x)$ ")
plt.legend()
plt.show()
razlika=[]
a=[]
for i in range(len(labdaa)):
    razlika.append(abs(lamb[i]-labdaa[i]))
    a.append(i)

plt.plot(a, np.log(razlika))
plt.title(r'Prikaz napake strelske metode za $E(x)$ pri različnih N')
plt.xlabel("x")
plt.ylabel(r"Error($E (x)$) ")
plt.legend()
plt.show()