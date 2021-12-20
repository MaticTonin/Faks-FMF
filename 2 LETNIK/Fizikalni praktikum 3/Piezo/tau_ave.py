import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp
import mycode as mc
import sys
import sympy as smp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Tau.dat")
#data load and manipulation
x, xerror = np.loadtxt(my_file, delimiter=";", unpack="True")

s = 0
sd = 0

for i in range(len(x)):
      s += x[i]/xerror[i]**2
      sd += 1/xerror[i]**2

av = s/sd
sigma = 1/np.sqrt(sd)
      
print("mi=: ")
print(av)
print("sigma= ")
print(sigma)
