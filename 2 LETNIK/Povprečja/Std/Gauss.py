import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file1 = os.path.join(THIS_FOLDER, "1_kolokvij.txt")


#data load and manipulation
V, prva, druga, x1 = np.loadtxt(my_file1, delimiter="\t", unpack="True")



x = np.array(x1)


av = 0
n = 0
for i in x:
    av += i
    n += 1
av/=n

st = 0
for i in x:
    st += (i-av)**2

st/=n
st = np.sqrt(st)

print("Povprečne točke: "+str(av)+" Povprečni procenti: " +str(st)+ " Max proceti: "+str(max(x)))



plt.hist(x, range=(0,2), bins=25, color="blue", label="Avg,pt: "+str(av))

plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Statistična termodinamika")

plt.show()
plt.close()
