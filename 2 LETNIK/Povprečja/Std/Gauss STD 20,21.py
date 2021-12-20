import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp
import tabula
#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


file = os.path.join(THIS_FOLDER, "Kolokviji 2020 STD.pdf")
table1 = tabula.read_pdf(file,pages=1,multiple_tables=False)
table2 = tabula.read_pdf(file,pages=2,multiple_tables=False)



my_file1 = os.path.join(THIS_FOLDER, "Kolokviji 2020 STD.txt")
import numpy as n
import matplotlib.pyplot as plt

cm = plt.cm.jet

#data load and manipulation
index, V,x1,x2,x3,x4,dn,skup, O = np.loadtxt(my_file1, delimiter=" ", unpack="True")



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
plt.title("Statistična naloga 1 naloga")




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
plt.title("Statistična naloga 2 naloga")



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
plt.title("Statistična naloga 3. naloga")


x = np.array(x4)
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
plt.title("Statistična naloga 4. naloga")
plt.show()



x = np.array(x1+x2)
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
n, bins, patches = plt.hist(x, range=(0,2), bins=25,edgecolor="black", color="blue")
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/25)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Statistična naloga 1. kolokvij")

x = np.array(x3+x4)
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
n, bins, patches = plt.hist(x, range=(0,2), bins=25,edgecolor="black", color="blue")
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/25)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Statistična naloga 2. kolokvij")
plt.show()

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



n, bins, patches = plt.hist(x, range=(0,4), bins=25,edgecolor="black", color="blue")
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/25)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Statistična naloga Skupno povprečje")
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


n, bins, patches = plt.hist(x, range=(5,10), bins=6,edgecolor="black", color="blue", label="Avg,pt: "+str(av))
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/25)) 
plt.xlabel("Procenti")
plt.ylabel("Študenti")
plt.legend()
plt.title("Statistična naloga oCENE")

plt.show()