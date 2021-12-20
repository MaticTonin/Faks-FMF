import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file1 = os.path.join(THIS_FOLDER, "Matematiika 3 20,21 kolokviji.txt")

import numpy as n
import matplotlib.pyplot as plt

cm = plt.cm.jet

#data load and manipulation
V, x1, x2 = np.loadtxt(my_file1, delimiter=" ", unpack="True")



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



plt.subplot(1, 3, 1)

n, bins, patches = plt.hist(x, range=(0,100), bins=25, edgecolor="black",color="blue", label="Avg,pt: "+str(av))
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/25)) # notice the i/25
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Matematika 3 1. kolokvij")




x = np.array(x2)


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


plt.subplot(1, 3, 2)
n, bins, patches = plt.hist(x, range=(0,100), bins=25,edgecolor="black", color="blue", label="Avg,pt: "+str(av))
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/25)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Matematika 3, 2 kolokvij")


x=[]
for i in range(len(x1)):
    x.append(x1[i]+x2[i])


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




plt.subplot(1, 3, 3)
n, bins, patches = plt.hist(x, range=(0,200), bins=25,edgecolor="black", color="blue", label="Avg,pt: "+str(av))
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/25)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Matematika 3, skupno")

plt.show()