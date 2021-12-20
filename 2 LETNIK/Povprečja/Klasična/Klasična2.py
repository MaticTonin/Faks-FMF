import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file1 = os.path.join(THIS_FOLDER, "2.kolokvij.txt")
#data load and manipulation
y, t, b, u , v, x = np.loadtxt(my_file1, delimiter=" ", unpack="True")



vsota=0
stevka=0
nad1=0
for i in range(len(x)):
      if x[i] !=0:
            vsota+=x[i]
            stevka+=1
      if x[i] >= 1.5:
            nad1+=1
            
av=vsota/stevka
proc=av/3
proc1=nad1/stevka

plt.hist(x, range=(0,4), bins=stevka, color="red", label="Število točk na študenta")

plt.xlabel("Točke")
plt.ylabel("Št študentov")
plt.legend()
plt.title("Klasična fizika, 2. kolokvij")
print(av)
print(proc)
print(nad1)
print(proc1)
print(stevka)
plt.show()
plt.close()
