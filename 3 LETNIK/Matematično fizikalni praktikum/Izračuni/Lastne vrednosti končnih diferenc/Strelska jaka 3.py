import numpy as np

def RK4(fun, h, a, b, y0):
    """
    Runge-Kutta 4. order
    IVP with step: 
        k1 = fun(x_n, y_n)
        k2 = fun(x_n + h/2, y_n + h*k1 / 2)
        k3 = fun(x_n + h/2, y_n + h*k2 / 2)
        k4 = fun(x_n + h, y_n + h*k3)

        y_(n+1) = y(n) + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    fun(x, y) = dy/dx
    h - step size
    a - x0 = a, startpoint
    b - x_last = b, endpoint
    y0 - initial value

    """
    #
    #fun = np.vectorize(fun)
    y0 = np.array(y0)

    xlist = np.array([a + i*h for i in range(int((b-a) / h) + 1)])
    try:
        ylist = np.empty((len(xlist), len(y0)))
    except TypeError:
        ylist = np.empty(len(xlist))
    ylist[0] = y0

    #to perform this operation once, not N = (b-a) / h times
    h2 = h / 2
    h6 = h / 6

    for i in range(len(xlist) - 1):
        k1 = fun(xlist[i], ylist[i])
        k2 = fun(xlist[i] + h2, ylist[i] + h2*k1)
        k3 = fun(xlist[i] + h2, ylist[i] + h2*k2)
        k4 = fun(xlist[i + 1], ylist[i] + h*k3)

        ylist[i + 1] =  ylist[i] + h6 * (k1 + 2*k2 + 2*k3 + k4) 


    return xlist, ylist.T

from scipy.integrate import solve_ivp

def shoot(f, a, b, lamb):
    def deriv(t, u):
        v = u[1]
        return np.array([v, f(t, u, lamb)])

    sol = solve_ivp(deriv, (a, b), y0=np.array([0, 1]), rtol = 1e-11, atol = 1e-15)
    return sol.y[0, -1]

def schrod(t, u, lamb):
    y, v = u
    return - lamb * y


a = -np.pi / 2
b = np.pi / 2

def tangent(G, a, b, eps):
    while abs(b - a) > eps:
        A = G(a)
        B = G(b)
        
        x = -(b - a) / (B - A) * B + b
        a = b
        b = x 
    return (a + b) / 2

lamb_list = np.linspace(0, 100, 1000)

def lamb_fun(f, a, b):
    def fun(lamb):
        return shoot(f, a, b, lamb)
    return fun

tan = lamb_fun(schrod, a, b)

""" sol = tangent(tan, 5, 1.8, 1e-10)

print(sol) """

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

""" sol = bisekcija(tan, 1.8, 5, 1e-10)

print(sol) """

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
N0 = 6
N = 8
zeros = np.array([i**2 for i in range(N0, N)])
start_int = np.empty((N-N0, 2))

for i in range(N-N0):
    start_int[i, 0] = zeros[i] - 0.1
    start_int[i, 1] = zeros[i] + 0.1

M = 7
eps_list = np.logspace(-10, -5, M)

from time import time

time_list_bis = np.empty(M)
time_list_tan = np.empty(M)
time_list_brent = np.empty(M)
eps_list_bis = np.empty(M)
eps_list_tan = np.empty(M)
eps_list_brent = np.empty(M)

for i in range(len(eps_list)):
    st = time()
    eps = 0
    for p0 in start_int:
        sol = bisekcija(tan, p0[0], p0[1], eps_list[i])
        eps += np.abs(sol - np.sum(p0)/2)
    
    time_list_bis[i] = time() - st
    eps_list_bis[i] = eps / (N - 1) 
    print(i)

for i in range(len(eps_list)):
    st = time()
    eps = 0
    for p0 in start_int:
        sol = tangent(tan, p0[0], p0[1], eps_list[i])
        eps += np.abs(sol - np.sum(p0)/2)
    
    time_list_tan[i] = time() - st
    eps_list_tan[i] = eps / (N - 1)
    print(i)

for i in range(len(eps_list)):
    st = time()
    eps = 0
    for p0 in start_int:
        sol = brent(tan, p0[0], p0[1], eps_list[i])
        eps += np.abs(sol - np.sum(p0)/2)

    
    time_list_brent[i] = time() - st
    eps_list_brent[i] = eps / (N - 1)
    print(i)

import matplotlib.pyplot as plt

time_list_bis = np.log(time_list_bis) / np.log(10)
time_list_tan = np.log(time_list_tan) / np.log(10)
time_list_brent = np.log(time_list_brent) / np.log(10)
eps_list_bis = np.log(eps_list_bis) / np.log(10)
eps_list_tan = np.log(eps_list_tan) / np.log(10)
eps_list_brent = np.log(eps_list_brent) / np.log(10)

from scipy.optimize import curve_fit

fun = lambda x, k, n: k*x + n

par, cov = curve_fit(fun, eps_list_bis, time_list_bis)

plt.plot(eps_list_bis, fun(eps_list_bis, *par), label="bisekcija, k = {:.2f}".format(par[0]), c = "blue")
plt.scatter(eps_list_bis, time_list_bis, marker=".", c="black")


par, cov = curve_fit(fun, eps_list_tan, time_list_tan)

plt.plot(eps_list_tan, fun(eps_list_tan, *par), label="tangentna, k = {:.2f}".format(par[0]), c="red")
plt.scatter(eps_list_tan, time_list_tan, marker=".", c="black")

par, cov = curve_fit(fun, eps_list_brent, time_list_brent)

plt.plot(eps_list_tan, fun(eps_list_tan, *par), label="brent, k = {:.2f}".format(par[0]), c="green")
plt.scatter(eps_list_brent, time_list_brent, marker=".", c="black")

plt.title("Časovna zahtevnost različnih algoritmov za iskanje ničel")

plt.xlabel(r"log$_{10}$eps")
plt.ylabel(r"log$_{10}$t")

plt.legend()
plt.show()
plt.close() 
