import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp
import tabula

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


import numpy as n
import matplotlib.pyplot as plt
my_file1 = os.path.join(THIS_FOLDER, "2. kolokvij.txt")
cm = plt.cm.ocean



#data load and manipulation
index, prvi, drugi, skup, ocena = np.loadtxt(my_file1, unpack="True", dtype=str)


skup1=[]
for i in skup:
    if i[2]=="/":
        skup1.append(float(i[0:2])/float(i[-2:]))
    if i[1]=="/":
        skup1.append(float(i[0:1])/float(i[-2:]))

prvi1=[]

for i in prvi:
    if i[2]=="/" and len(i)>1:
        prvi1.append(float(i[0:2])/float(i[-2:]))
    if i[1]=="/" and len(i)>1:
        prvi1.append(float(i[0:1])/float(i[-2:]))
    else:
        prvi1.append(0)

drugi1=[]

for i in drugi:
    if len(i)>1 and i[2]=="/":
        drugi1.append(float(i[0:2])/float(i[-2:]))
    if  len(i)>1 and i[1]=="/":
        drugi1.append(float(i[0:1])/float(i[-2:]))
    else:
        drugi1.append(0)





x = np.array(prvi1)
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

x1=[]
for i in range(len(x)):
    if x[i]!=0:
        x1.append(x[i])
st/=n
st = np.sqrt(st)

print("Povprečne točke: "+str(av)+" Povprečni procenti: " +str(st)+ " Max proceti: "+str(max(x1)))



n, bins, patches = plt.hist(x1, range=(0,2.5), bins=30,edgecolor="black", color="blue", label="Avg: "+str(av)+ "\n"+"Nad 1.5: " + str(nad)+ "\n"+"Vseh: "+str(n))
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/30)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Klasična mehanika Skupno povprečje prvega kolokvija")
plt.show()

x = np.array(drugi1)
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
x1=[]
for i in range(len(x)):
    if x[i]!=0:
        x1.append(x[i])
print("Povprečne točke: "+str(av)+" Povprečni procenti: " +str(st)+ " Max proceti: "+str(max(x1)))



n, bins, patches = plt.hist(x1, range=(0,3), bins=30, edgecolor="black", color="blue", label="Avg: "+str(av)+ "\n"+"Nad 1.5: " + str(nad)+ "\n"+"Vseh: "+str(n))
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/30)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Klasična mehanika Skupno povprečje drugega kolokvija")
plt.show()

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
x1=[]
for i in range(len(x)):
    if x[i]!=0:
        x1.append(x[i])
print("Povprečne točke: "+str(av)+" Povprečni procenti: " +str(st)+ " Max proceti: "+str(max(x1)))



n, bins, patches = plt.hist(x1, range=(0,5), bins=30,edgecolor="black", color="blue", label="Avg: "+str(av)+ "\n"+"Nad 1.5: " + str(nad)+ "\n"+"Vseh: "+str(n))
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/30)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Klasična mehanika Skupno povprečje")
plt.show()

x = np.array(ocena, dtype=float)

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
x1=[]
for i in range(len(x)):
    if x[i]!=0:
        x1.append(x[i])
print("Povprečne točke: "+str(av)+" Povprečni procenti: " +str(st)+ " Max proceti: "+str(max(x1)))



n, bins, patches = plt.hist(x1, range=(5,10), bins=6,edgecolor="black", color="blue", label="Avg: "+str(av)+ "\n"+"Nad 1.5: " + str(nad)+ "\n"+"Vseh: "+str(n))
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/6)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Klasična mehanika Skupno povprečje")
plt.show()



