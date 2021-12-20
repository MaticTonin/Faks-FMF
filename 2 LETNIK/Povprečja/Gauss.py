import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file1 = os.path.join(THIS_FOLDER, "Mat1.txt")


#data load and manipulation
x1 = np.loadtxt(my_file1, delimiter=" ", unpack="True")



x = np.array([])


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

print(av, st)



plt.hist(x, range=(0,125), bins=25, color="blue", label="podatki")

plt.xlabel("h[m]")
plt.ylabel("B[T]")
plt.legend()
plt.title("B(h)")

plt.show()
plt.close()
