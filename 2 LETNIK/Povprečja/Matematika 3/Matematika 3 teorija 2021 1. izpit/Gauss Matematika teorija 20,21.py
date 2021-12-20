import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp
import tabula
from scipy.stats import norm
#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

my_file1 = os.path.join(THIS_FOLDER, "Teorija.txt")
import numpy as n
import matplotlib.pyplot as plt

cm = plt.cm.jet
cm1 = plt.cm.twilight

#data load and manipulation
V, proc = np.loadtxt(my_file1, delimiter="\t", unpack="True")


x = np.array(proc)
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

mean,std=norm.fit(proc)

n, bins, patches = plt.hist(x, range=(0,100), bins=35,edgecolor="black",normed=True, color="blue", label="Avg,pt: "+str(av))

for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/35)) 
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
y = norm.pdf(x, mean, std)
for i in range(len(y)):
    plt.scatter(x[i],y[i], color=cm1(i/1000))
plt.xlabel("Točke")
plt.ylabel("Študenti")
plt.legend()
plt.title("Matematika teorija 1 izpit")

plt.show()


ocene=[]
for i in x:
    if i<50:
        ocene.append(5)
    elif 50<=i<60:
        ocene.append(6)
    elif 60<=i<70:
        ocene.append(7)
    elif 70<=i<80:
        ocene.append(8)
    elif 80<=i<90:
        ocene.append(9)
    elif 90<=i<100:
        ocene.append(10)
sum=0
for i in ocene:
    sum+=i

av=sum/len(ocene)
print(proc)

n, bins, patches = plt.hist(ocene, range=(5,10), bins=6,edgecolor="black", color="blue", label="Avg,pt: "+str(av))
print(n[0])
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/6)) 
plt.xlabel("Ocene")
plt.ylabel("Študenti")
plt.legend()
plt.title("Matematika teorija 1 izpit, ocene")

plt.show()