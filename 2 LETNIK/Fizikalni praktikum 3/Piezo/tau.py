import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp
import mycode as mc
import sys
import sympy as smp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "M1005g.dat")

#data load and manipulation
x, xerror, y = np.loadtxt(my_file, delimiter=";", unpack="True")

a = [3 for i in range(len(x))]
yerror = np.array(a)
tau = np.array([])
err = np.array([])

l = len(x)
#print(len(yerror))

for i in range(l):
      for j in range(l):
            if j > i:
                  difft = x[j] - x[i] 
                  ratioy = np.log(y[j]/y[i])
                  tau1 = - difft / ratioy

                  dtau1 = ((xerror[i] + xerror[j]) / difft - (y[j] * yerror[j] + (y[j])**2 * yerror[i]) / (ratioy * y[i]**2)) * tau1
                  tau = np.append(tau, [tau1])
                  err = np.append(err, [dtau1])

s = 0
sd = 0
count = 0

def minimum(a, j):
    m = (a[j], j)
    for i in range(j + 1, len(a)):
        if a[i] < m[0]: m = (a[i], i)
    
    return m

def uredi_z_vstavljanjem(a, b):
    for i in range(len(a)-1):
        m = minimum(a, i)
        a[i], a[m[1]] = a[m[1]], a[i]
        b[i], b[m[1]] = b[m[1]], b[i]
    return a

uredi_z_vstavljanjem(err, tau)

print(err)
print(tau)

for i in range(l):
      count += 1
      s += tau[i]/err[i]**2
      sd += 1/err[i]**2

av = s/sd
sigma = 1/np.sqrt(sd)
      
print("mi=: ")
print(av)
print("sigma= ")
print(sigma)
