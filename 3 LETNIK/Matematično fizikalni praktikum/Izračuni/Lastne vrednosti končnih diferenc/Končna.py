import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
def bisekcija(f, a, b, eps):
    left = f(a)
    right = f(b)
    c = (b + a) / 2
    middle = f(c)

    while b - a > eps:
        if left * right == 0: break
        
        elif left * middle < 0:
            b = c
            c = (a + b) / 2
            right = middle
            middle = f(c)
        
        elif right * middle < 0:
            a = c
            c = (a + b) / 2
            left = middle
            middle = f(c)
        
        else: raise ValueError("Invalid initial interval, a, b = {}, {}; f(a), f(b) = {}, {}".format(a, b, left, right))

    return (a+b) / 2


def brent(f, a, b, eps):
    A = f(a)
    B = f(b)

    if B * A > 0: 
        raise ValueError("Root not bracketed, a, b = {}, {}; f(a), f(b) = {}, {}".format(a, b, A, B))

    if np.abs(A) < np.abs(B):
        a, b = b, a
        A, B = B, A

    c = a
    C = f(c)

    flag = True

    #set value for first while loop

    while B != 0 and abs(b-a) > eps:
        if A != C and B != C:
            s = (a * B * C) / ((A-B) * (A - C)) + (b * A * C) / ((B - A) * (B - C)) + (c * A * B) / ((C - A) * (C - B))

        else:
            s = b - B * (b - a) / (B - A)

        if flag:
            if s > b or s < (a + 3 * b) / 4 or np.abs(s - b) >= np.abs(b-c) / 2 or np.abs(b-c) < eps:
                s = (b + a) / 2
                flag = True
            else:
                flag = False
        else:
            if np.abs(s-b) >= np.abs(c-d) / 2 or np.abs(c-d) < eps:
                s = (b + a) / 2
                flag = True
            else:
                flag = False
        
        S = f(s)
        d = c
        c = b
        C = f(c)

        if A * S < 0:
            b = s
            B = S
        else:
            a = s
            A = S
        if np.abs(A) < np.abs(B):
            a, b = b, a
            A, B = B, A
        
    if B == 0: return b
    else: return (a+b) / 2


def tangent(G, a, b, eps):
    while abs(b - a) > eps:
        A = G(a)
        B = G(b)
        
        x = -(b - a) / (B - A) * B + b
        a = b
        b = x 
    return (a + b) / 2

def schrod(t, u, lamb):
    y, v = u
    return - lamb * y

V_list = np.linspace(25, 100, 10)

a = -np.pi / 2
b = np.pi / 2

E_list = np.empty(len(V_list))

"""for i in range(len(V_list)):
    V_0 = V_list[i]

    def shoot(f, a, b, lamb):
        def deriv(t, u):
            v = u[1]
            return np.array([v, f(t, u, lamb)])

        sol = solve_ivp(deriv, (a, b), y0=np.array([1 / np.sqrt(V_0 - lamb), 1]), rtol = 1e-11, atol = 1e-15)
        return sol.y[0, -1], sol.y[1, -1]

    def lamb_fun(f, a, b):
        def fun(lamb):
            y, v = shoot(f, a, b, lamb)
            return y * np.sqrt(V_0 - lamb) + v
        return fun

    my_fun = lamb_fun(schrod, a, b)

    E_list[i] = bisekcija(my_fun, 0, 2, 1e-10)"""

V_0 =60
def shoot(f, a, b, lamb):
    def deriv(t, u):
        v = u[1]
        return np.array([v, f(t, u, lamb)])
    sol = solve_ivp(deriv, (a, b), y0=np.array([1 / np.sqrt(V_0 - lamb), 1]), rtol = 1e-11, atol = 1e-15)
    return sol.y[0, -1], sol.y[1, -1]
def lamb_fun(f, a, b):
    def fun(lamb):
        y, v = shoot(f, a, b, lamb)
        return y * np.sqrt(V_0 - lamb) + v
    return fun
my_fun = lamb_fun(schrod, a, b)
E_list = bisekcija(my_fun, 0, 2, 1e-10)
lamb_list=np.linspace(0.5,50,100)
y=[shoot(schrod, a, b, lamb)[0] for lamb in lamb_list]
y=y/max(y)
nicle=[]
y_nic=[]
for i in range(len(y)):
    if y[i]<=10**(-16):
        y_nic.append(y[i])
        nicle.append(lamb_list[i])
izd_fun= lamb_fun(schrod, a, b)
lambd=[]
"""while (1):
    lambd.append(bisekcija(izd_fun, 0.1, 2, 1e-10))
    lambd.append(bisekcija(izd_fun, 3.5, 4.5, 1e-10))
    lambd.append(bisekcija(izd_fun, 8.5, 9.5, 1e-10))
    lambd.append(bisekcija(izd_fun, 15.1, 16.5, 1e-10))
    lambd.append(bisekcija(izd_fun, 24.5, 25.5, 1e-10))
    lambd.append(bisekcija(izd_fun, 35.5, 36.5, 1e-10))
    lambd.append(bisekcija(izd_fun, 47.5, 49.5, 1e-10))
    break
print(lambd)"""
labdaa=[0.944,3.6995,8.292,14.7108,22.9243,32.88825,44.486]
plt.plot(lamb_list,y)
plt.title(r'Prikaz iskanja ničel s Strelsko metodo pri potenicalu V='+str(V_0))
plt.plot(labdaa[0],0, ".", color="red")
plt.text(labdaa[0],0.05,r'$\lambda_1=$'+str(labdaa[0]))
plt.plot(labdaa[1],0, ".", color="red")
plt.text(labdaa[1],-0.05,r'$\lambda_2=$'+str(labdaa[1]))
plt.plot(labdaa[2],0, ".", color="red")
plt.text(labdaa[2],0.05,r'$\lambda_3=$'+str(labdaa[2]))
plt.plot(labdaa[3],0, ".", color="red")
plt.text(labdaa[3],-0.05,r'$\lambda_4=$'+str(labdaa[3]))
plt.plot(labdaa[4],0, ".", color="red")
plt.text(labdaa[4],0,r'$\lambda_5=$'+str(labdaa[4]))
plt.plot(labdaa[5],0, ".", color="red")
plt.text(labdaa[5],0,r'$\lambda_6=$'+str(labdaa[5]))
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$y_\lambda(b)-y(b)$ ")
plt.legend()
plt.show()
N=1000
n=1
x=np.linspace(a,b,N)
labdaa=[0.944,3.6995,8.292,14.7108,22.9243,32.88825,44.486]
while n<=6:
    if n%2==1:
        psi_main4=np.cos((n)*x)
        def dydt_y(t,icons):
            v=icons[1]
            return np.array([v, schrod(x,icons,labdaa[n-1])])
        sol = solve_ivp(dydt_y, (a, b), y0=np.array([0, 1]), rtol = 1e-8, atol = 1e-10)
        plt.plot(sol.t,sol.y[0]/max(sol.y[0])+labdaa[n-1], label=r"$\psi$"+str(n-1))
    if n%2==0:
        psi_main4=np.sin((n)*x)
        def dydt_y(t,icons):
            v=icons[1]
            return np.array([v, schrod(x,icons,labdaa[n-1])])
        sol = solve_ivp(dydt_y, (a, b), y0=np.array([0, 1]), rtol = 1e-8, atol = 1e-10)
        plt.plot(sol.t,sol.y[0]/max(sol.y[0])+labdaa[n-1], label=r"$\psi$"+str(n-1))
    n+=1
plt.title(r'Prikaz približkov Strelsko metodo za $\psi(x)$ v potencialu V='+str(V_0))
plt.xlabel("x")
plt.ylabel(r"$\psi (x)$ ")
plt.legend()
plt.show()

V_list = np.linspace(10, 1500, 200)

a = -np.pi / 2
b = np.pi / 2

E_list = np.empty(len(V_list))
cas=[]
for i in range(len(V_list)):
    V_0 = V_list[i]
    start1 = time.time()
    def shoot(f, a, b, lamb):
        def deriv(t, u):
            v = u[1]
            return np.array([v, f(t, u, lamb)])

        sol = solve_ivp(deriv, (a, b), y0=np.array([1 / np.sqrt(V_0 - lamb), 1]), rtol = 1e-11, atol = 1e-15)
        return sol.y[0, -1], sol.y[1, -1]

    def lamb_fun(f, a, b):
        def fun(lamb):
            y, v = shoot(f, a, b, lamb)
            return y * np.sqrt(V_0 - lamb) + v
        return fun

    my_fun = lamb_fun(schrod, a, b)
    E_list[i] = brent(my_fun, 0.2, 2, 1e-15)
    end1 = time.time()
    cas.append(end1-start1)
a=[]
for i in range(200):
    a.append(1)
"""plt.xlabel(r"$V_0$")
plt.ylabel(r"E")
plt.title("Prikaz odvisnosti prve lastne vrednosti od potenicala V")
plt.plot(V_list, a,"-", color="red")
plt.scatter(V_list, E_list, marker=".")

plt.show()
plt.close() """

plt.xlabel(r"$V_0$")
plt.ylabel(r"t")
plt.title("Prikaz časovne odvisnosti prve lastne vrednosti o potenicala V")
plt.scatter(V_list, cas, marker=".")

plt.show()
plt.close() 