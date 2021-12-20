import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file1 = os.path.join(THIS_FOLDER, "Matematiika3izpit.txt")
#data load and manipulation
y, x = np.loadtxt(my_file1, delimiter="\t", unpack="True")



vsota=0
stevka=0
for i in range(len(x)):
      if x[i] !=0:
            vsota+=x[i]
            stevka+=1
            
av=vsota/stevka



plt.hist(x, range=(0,60), bins=stevka, color="red", label="Število točk na študenta")

plt.xlabel("Točke")
plt.ylabel("Št študentov")
plt.legend()
plt.title("Matematika 3, 1. izpit")
print(av)
plt.show()
plt.close()
