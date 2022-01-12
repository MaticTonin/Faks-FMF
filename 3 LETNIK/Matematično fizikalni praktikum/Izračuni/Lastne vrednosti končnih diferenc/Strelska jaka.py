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

    sol = solve_ivp(deriv, (a, b), y0=np.array([0, 1]), rtol = 1e-8, atol = 1e-10)
    print(a, b)
    return sol.y[0, -1]

def schrod(t, u, lamb):
    y, v = u
    return - lamb * y

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

    return np.array([a, b])

a = -np.pi / 2
b = np.pi / 2

def bisec_fun(f, a, b):
    def fun(lamb):
        return shoot(f, a, b, lamb)
    return fun

my_fun = bisec_fun(schrod, a, b)

lamb = bisekcija(my_fun, 0.1, 2, 1e-10)

print(lamb)

def deriv(t, u):
    v = u[1]
    return np.array([v, schrod(t, u, lamb[0])])


sol = solve_ivp(deriv, (a, b), y0=np.array([0, 1]), rtol = 1e-8, atol = 1e-10)

import matplotlib.pyplot as plt

plt.plot(sol.t, sol.y[0])

plt.show()
plt.close()