import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file1 = os.path.join(THIS_FOLDER, "Mat1.txt")
#data load and manipulation
x, a, b, c, d, y = np.loadtxt(my_file1, delimiter=" ", unpack="True")



vsota=0
stevka=0
for i in range(len(x)):
      if y[i] !=0:
            vsota+=y[i]
            stevka+=1
            
av=vsota/stevka



plt.hist(y, range=(0,125), bins=70, color="red", label="podatki")
print("mi=: ")
print(av)

plt.xlabel("Točke")
plt.ylabel("Št študentov")
plt.legend()
plt.title("Matematika 3, 2. kolokvij")

plt.show()
plt.close()
