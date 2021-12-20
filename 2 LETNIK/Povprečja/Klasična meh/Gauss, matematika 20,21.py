import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp
import tabula

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


import numpy as n
import matplotlib.pyplot as plt
my_file1 = os.path.join(THIS_FOLDER, "1. kolokvij.txt")
cm = plt.cm.jet

#data load and manipulation
index, skup = np.loadtxt(my_file1, delimiter=" ", unpack="True", dtype=str)

print(skup)
skup1=[]
for i in skup:
    print(i[1])
    if i[2]=="/":
        skup1.append(float(i[0:2])/float(i[-2:]))
    if i[1]=="/":
        skup1.append(float(i[0:1])/float(i[-2:]))



x = np.array(skup1)
av = 0
n = 0
nad=0
for i in x:
    if i>=1.5:
        nad+=1
    av += i
    n += 1
av/=n

st = 0
for i in x:
    st += (i-av)**2

st/=n
st = np.sqrt(st)

print("Povprečne točke: "+str(av)+" Povprečni procenti: " +str(st)+ " Max proceti: "+str(max(x)))



n, bins, patches = plt.hist(x, range=(0,2.5), bins=30,edgecolor="black", color="blue", label="Avg: "+str(av)+ "\n"+"Nad 1.5: " + str(nad)+ "\n"+"Vseh: "+str(n))
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/30)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Moderna fizika Skupno povprečje")
plt.show()


