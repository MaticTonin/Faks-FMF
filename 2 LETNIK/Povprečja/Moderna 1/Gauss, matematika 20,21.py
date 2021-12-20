import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp
import tabula

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file1 = os.path.join(THIS_FOLDER, "Matematiika 3 20,21 kolokviji.txt")


file = os.path.join(THIS_FOLDER, "Ocene_MF1_2021.pdf")
table = tabula.read_pdf(file,pages=1,multiple_tables=True)
print(table[1])
print(table[0])


my_file1 = os.path.join(THIS_FOLDER, "Kolokviji ocene.txt")
import numpy as n
import matplotlib.pyplot as plt

cm = plt.cm.jet

#data load and manipulation
index, V,x1,x2,x3,x4,y1,skup, O = np.loadtxt(my_file1, delimiter="\t", unpack="True")



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



plt.subplot(2, 2, 1)

n, bins, patches = plt.hist(x, range=(0,1), bins=25, edgecolor="black",color="blue")
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/25)) # notice the i/25
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Moderna fizika 1 naloga")




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


plt.subplot(2, 2, 2)
n, bins, patches = plt.hist(x, range=(0,1), bins=25,edgecolor="black", color="blue")
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/25)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Moderna fizika 2 naloga")



x = np.array(x3)
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


plt.subplot(2, 2, 3)
n, bins, patches = plt.hist(x, range=(0,1), bins=25,edgecolor="black", color="blue")
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/25)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Moderna fizika 3. naloga")


x = np.array(x3)
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


plt.subplot(2, 2, 4)
n, bins, patches = plt.hist(x, range=(0,1), bins=25,edgecolor="black", color="blue")
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/25)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Moderna fizika 4. naloga")
plt.show()




x = np.array(y1)
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


plt.subplot(1, 2, 1)
n, bins, patches = plt.hist(x, range=(0,4), bins=25,edgecolor="black", color="blue")
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/25)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Moderna fizika 2. kolokvij")

x = np.array(skup)
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


plt.subplot(1, 2, 2)
n, bins, patches = plt.hist(x, range=(0,8), bins=25,edgecolor="black", color="blue")
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/25)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Moderna fizika Skupno povprečje")
plt.show()


x = np.array(O)
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


n, bins, patches = plt.hist(x, range=(5,10), bins=25,edgecolor="black", color="blue", label="Avg,pt: "+str(av))
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/25)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Moderna fizika oCENE")

plt.show()