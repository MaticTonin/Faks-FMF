import numpy as np

def second_difference_matrix(N):
    A = np.zeros((N, N))

    for i in range(1, N-1):
        A[i, i-1] = 1
        A[i, i] = -2
        A[i, i+1] = 1
    
    #boundary
    A[0,0] = -2
    A[0, 1] = 1

    A[N-1, N-2] = 1
    A[N-1, N-1] = -2

    return A

def schrod(t, y, lamb):
    return - lamb * y

a, b = -np.pi/2, np.pi/2

N = 1000

h = (b - a) / N

def function(f, N, alpha, beta, a, b):
    h = (b-a) / N
    h_2 = h**2
    h2 = 2 * h
    def vec(y):
        w = np.empty(len(y))

        w[0] = h_2 * f(a, y[0]) - alpha
        for i in range(1, len(w) - 1):
            w[i] = h_2 * f(a + i*h, y[i])
        w[-1] = h_2 * f(b, y[-1]) - beta
        
        return w
    return vec

import numpy.linalg as lin

def piccard(A, yfun, y0, eps):
    A = lin.inv(A)
    w = np.dot(A, yfun(y0))
    w = w / lin.norm(w)


    while lin.norm(y0 - w) > eps:
        y0 = np.copy(w)
        w = np.dot(A, yfun(y0))
        w = w / lin.norm(w)
    
    return w

def schrod_lambda(f, lamb):
    def fun(t, y):
        return f(t, y, lamb)
    return fun

schrod_1 = schrod_lambda(schrod, 4)
print(schrod_1(1, 10))

my_fun = function(schrod_1, N, 0, 0, a, b)

A = second_difference_matrix(N)

y0 = np.array([i * (N - i) * (N/2 - i) for i in range(N)])

y = piccard(A, my_fun, y0, 1e-10)
x = np.linspace(a, b, N) 

A = second_difference_matrix(N)
w, v = lin.eigh(A) 

print(w / h**2, v)

x = np.linspace(a, b, N)
y = v.T[-5]

import matplotlib.pyplot as plt

plt.scatter(x, y, marker=".")

plt.show()
plt.close()