import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp


#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Matematika.txt")

#data load and manipulation
x, y = np.loadtxt(my_file, delimiter=" ", unpack="True")

l = len(x)
#print(len(yerror))
vsota=0
stevka=0
for i in range(l):
      if y[i] !=0:
            vsota+=y[i]
            stevka+=1

av=vsota/stevka
      
print("mi=: ")
print(av)
