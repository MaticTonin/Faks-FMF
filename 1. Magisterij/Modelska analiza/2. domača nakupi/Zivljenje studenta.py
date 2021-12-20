from operator import index
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
cm = plt.cm.jet
cm1 = plt.cm.winter
cm2 = plt.cm.autumn
import os
import emcee
import scipy as sc
from scipy.stats import norm
import corner
from scipy.optimize import curve_fit
import os
from sklearn import preprocessing
from mpl_toolkits import mplot3d
import function as nf
import re


import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

data_file = THIS_FOLDER + "\\tabela-zivil.dat"

data = np.genfromtxt(data_file)
data = data[:,1:]

names = np.genfromtxt(data_file, dtype=np.str)
names = names[:, :1]

#calories
def izdelava_b(cena):
    return np.array([2000, -50, -50, -1000, -18, -60, -3500, 2])

def izdelava_A(data_set,index):
    data1=data_set
    A_ub = np.array([-i[1:8] for i in data1])
    A_ub = np.concatenate((A_ub, np.array([[0.1 for i in data1]]).T), axis=1) #weight
    return A_ub
def resoult_uporabljene(x):
    resoult=[]
    indexation=[]
    for k in range(len(x)):
        if x[k]<10**(-3):
            resoult.append(0)
        else:
            resoult.append(x[k])
            indexation.append(k)
    return resoult, indexation

def izdelava_c(data_set, index):
    data1=data_set
    c = np.array([i[9] for i in data1])
    return c
def izdelava_b_vector(data_set,index,x):
    data1=data_set
    vector_b = np.array([x[i] * data1[i] for i in range(len(data1))])
    return vector_b
def izdelava_kolicin(vector_b,i):
    cal=vector_b.T[0]
    cal_sum=np.sum(cal)
    mast=vector_b.T[1]
    mast_sum=np.sum(mast)
    carb=vector_b.T[2]
    carb_sum=np.sum(carb)
    prot=vector_b.T[3]
    prot_sum=np.sum(prot)
    Ca=vector_b.T[4]
    Ca_sum=np.sum(Ca)*0.001
    Fe=vector_b.T[5]
    Fe_sum=np.sum(Fe)*0.001
    C=vector_b.T[6]
    C_sum=np.sum(C)*0.001
    K=vector_b.T[7]
    K_sum=np.sum(K)*0.001
    Na=vector_b.T[8]
    Na_sum=np.sum(Na)*0.001
    if (i+2)%7!=0 and ((i+1)%7!=0):
        money=vector_b.T[9]
    else: 
        money=0
    money_sum=np.sum(money)
    return cal_sum, mast_sum,carb_sum,prot_sum,Ca_sum,Fe_sum,C_sum,K_sum,Na_sum, money_sum

def izdelava_relavantnih(x,names,day, m,cena, kalorije):
    resoult=[]
    indexation=[]
    for i in range(len(x)):
        if x[i]<10**(-2):
            resoult.append(0)
        else:
            resoult.append(x[i])
            indexation.append(i)

    sum=0
    k=0
    for i in indexation:
        sum+=sol.x[i]


    resoult_not=[]
    resoult=[]
    for i in indexation:
        resoult.append(sol.x[i]/sum)
        resoult_not.append(sol.x[i])


    plt.bar(x=[i for i in range(len(resoult_not))], height=resoult_not, label="Cena %.2f, kalorije %.2f" %(cena, kalorije))
    plt.xticks([i for i in range(len(resoult_not))],labels=[names[int(i), 0] for i in indexation])
    plt.xlabel("Živilo")
    plt.ylabel("Masa živila na 100g")
    plt.title("Optimizacija cene, zgolj relavantne količine, dan="+str(day)+"  zaporedni:"+str(m+1))
    plt.legend()
prihranki=40
N=20
MATRIKA=[]
dnevi=np.linspace(0, N,N)
snovi=np.linspace(0,10,11)
denar=[]
ind=[0]
cena=3
meja=[]
teden=["Ponedeljek", "Torek", "Sreda", "Četrtek", "Petek","Sobota","Nedelja"]

for i in range(len(dnevi)):
    b=izdelava_b(prihranki)
    if i%7==0 and i!=0:
        data = np.genfromtxt(data_file)
        data = data[:,1:]
        #data=np.delete(np.array(data),start,0)
        names = np.genfromtxt(data_file, dtype=np.str)
        names = names[:, :1]
        prihranki+=20
    if i%7!=0:
        data=np.delete(np.array(data),ind,0)
        names=np.delete(np.array(names),ind,0)
    ind=[]
    if cena>prihranki:
        cena=prihranki/4
        b=izdelava_b(cena)
        meja.append(i)
    A=izdelava_A(data,ind)
    c=izdelava_c(data,ind)
    sol = opt.linprog(c, A.T, b)
    x=np.array(sol.x)
    vector_b=izdelava_b_vector(data,ind,x)
    resoult, indexation=resoult_uporabljene(x)
    for j in indexation:
        print(names[int(j)])
    for j in indexation:
        ind.append(j)
    if i==0:
        ind=indexation
        start=indexation
    if prihranki<0:
        dnevi=np.linspace(0,i,i)
        break
    if (i+1)%7!=0 and ((i+2)%7!=0):
        money=vector_b.T[9]
        money_sum=np.sum(money)
        prihranki=prihranki-money_sum
        denar.append(prihranki)
    else: 
        money=0
        money_sum=np.sum(money)
        prihranki=prihranki-money_sum
        denar.append(prihranki)
    kolicine=np.array(izdelava_kolicin(vector_b, i))
    element=np.append(kolicine,prihranki)
    print(sum(vector_b[0]))
    relavantne=izdelava_relavantnih(x,names,teden[i%7],i,money_sum, sum(vector_b.T[0]))
    MATRIKA.append(element)
    print(i)
print(meja)
teden=["Ponedeljek", "Torek", "Sreda", "Četrtek", "Petek","Sobota","Nedelja"]
x_label=[]
size=int(len(dnevi/7))
for i in range(N):
    x_label.append(teden[i%7])

MATRIKA=np.array(MATRIKA)
snovi=MATRIKA.T[0:3]
minerali=MATRIKA.T[4:11]
cenovno=MATRIKA.T[9:10]
fig, ax = plt.subplots()
im= ax.imshow(snovi,origin='lower')
ax.set_yticks(np.arange(len(snovi)))
ax.set_xticks(np.arange(len(dnevi)))
ax.set_xticklabels(x_label)
ax.set_yticklabels(["kalorije","mascobe",	"ogljikovi hidrati",	"proteini",	"Ca",	"Fe", "Vitamin C",	"K", "Na","Cena", "Prihranki"])
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
for i in range(len(dnevi)):
    for j in range(len(snovi)):
            text = ax.text(i, j, "%2.f" %(snovi[j][i]),
                ha="center", va="center", color="w")
ax.set_title("Prikaz zaužitih količin po dnevih, snovi")
plt.show()

fig, ax = plt.subplots()
im= ax.imshow(minerali)
ax.set_yticks(np.arange(len(minerali)))
ax.set_xticks(np.arange(len(dnevi)))
ax.set_xticklabels(x_label)
ax.set_yticklabels(["Ca",	"Fe", "Vitamin C",	"K", "Na","Cena", "Prihranki"])
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
for i in range(len(dnevi)):
    for j in range(len(minerali)):
            text = ax.text(i, j, "%.2f" %(minerali[j][i]),
                ha="center", va="center", color="w")
ax.set_title("Prikaz zaužitih količin po dnevih, minerali in cena")

plt.show()
prihranki=40
N=31
data_file = THIS_FOLDER + "\\tabela-zivil.dat"

data = np.genfromtxt(data_file)
data = data[:,1:]

names = np.genfromtxt(data_file, dtype=np.str)
names = names[:, :1]
MATRIKA=[]
dnevi5=np.linspace(0, N,N)
snovi=np.linspace(0,10,11)
denar5=[]
ind=[0]
cena=3
meja5=[]
teden=["Ponedeljek", "Torek", "Sreda", "Četrtek", "Petek","Sobota","Nedelja"]
i=0
for i in range(len(dnevi)):
    b=izdelava_b(prihranki)
    if i%7==0 and i!=0:
        data = np.genfromtxt(data_file)
        data = data[:,1:]
        #data=np.delete(np.array(data),start,0)
        names = np.genfromtxt(data_file, dtype=np.str)
        names = names[:, :1]
        prihranki+=5
    if i%7!=0:
        data=np.delete(np.array(data),ind,0)
        names=np.delete(np.array(names),ind,0)
    ind=[]
    if cena>prihranki:
        cena=prihranki/4
        b=izdelava_b(cena)
        meja.append(i)
    A=izdelava_A(data,ind)
    c=izdelava_c(data,ind)
    sol = opt.linprog(c, A.T, b)
    x=np.array(sol.x)
    vector_b=izdelava_b_vector(data,ind,x)
    resoult, indexation=resoult_uporabljene(x)
    for j in indexation:
        print(names[int(j)])
    for j in indexation:
        ind.append(j)
    if i==0:
        ind=indexation
        start=indexation
    if prihranki<0:
        meja5=i
        dnevi5=np.linspace(0,i,i)
        break
    if (i+1)%7!=0 and (i%7!=0):
        money=vector_b.T[9]
        money_sum=np.sum(money)
        prihranki=prihranki-money_sum
        denar5.append(prihranki)
    else: 
        money=0
        money_sum=np.sum(money)
        prihranki=prihranki-money_sum
        denar5.append(prihranki)
    kolicine=np.array(izdelava_kolicin(vector_b, i))
    element=np.append(kolicine,prihranki)
    print(sum(vector_b[0]))
    relavantne=izdelava_relavantnih(x,names,teden[i%7],i,money,sum(vector_b.T[0]))
    MATRIKA.append(element)
    print(i)

plt.show()
prihranki=40
N=31
MATRIKA=[]
data_file = THIS_FOLDER + "\\tabela-zivil.dat"

data = np.genfromtxt(data_file)
data = data[:,1:]

names = np.genfromtxt(data_file, dtype=np.str)
names = names[:, :1]
dnevi10=np.linspace(0, N,N)
snovi=np.linspace(0,10,11)
denar10=[]
ind=[0]
cena=3
meja10=[]
teden=["Ponedeljek", "Torek", "Sreda", "Četrtek", "Petek","Sobota","Nedelja"]
i=0
for i in range(len(dnevi)):
    b=izdelava_b(prihranki)
    if i%7==0 and i!=0:
        data = np.genfromtxt(data_file)
        data = data[:,1:]
        #data=np.delete(np.array(data),start,0)
        names = np.genfromtxt(data_file, dtype=np.str)
        names = names[:, :1]
        prihranki+=10
    if i%7!=0:
        data=np.delete(np.array(data),ind,0)
        names=np.delete(np.array(names),ind,0)
    ind=[]
    if cena>prihranki:
        cena=prihranki/4
        b=izdelava_b(cena)
        meja.append(i)
    A=izdelava_A(data,ind)
    c=izdelava_c(data,ind)
    sol = opt.linprog(c, A.T, b)
    x=np.array(sol.x)
    vector_b=izdelava_b_vector(data,ind,x)
    resoult, indexation=resoult_uporabljene(x)
    for j in indexation:
        print(names[int(j)])
    for j in indexation:
        ind.append(j)
    if i==0:
        ind=indexation
        start=indexation
    if prihranki<0:
        meja10=i
        dnevi10=np.linspace(0,i,i)
        break
    if (i+1)%7!=0 and (i%7!=0):
        money=vector_b.T[9]
        money_sum=np.sum(money)
        prihranki=prihranki-money_sum
        denar10.append(prihranki)
    else: 
        money=0
        money_sum=np.sum(money)
        prihranki=prihranki-money_sum
        denar10.append(prihranki)
    kolicine=np.array(izdelava_kolicin(vector_b, i))
    element=np.append(kolicine,prihranki)
    print(sum(vector_b[0]))
    relavantne=izdelava_relavantnih(x,names,teden[i%7],i,money,sum(vector_b.T[0]))
    MATRIKA.append(element)
    print(i)
print(meja)
plt.show()
prihranki=40
N=31
MATRIKA=[]
data_file = THIS_FOLDER + "\\tabela-zivil.dat"

data = np.genfromtxt(data_file)
data = data[:,1:]

names = np.genfromtxt(data_file, dtype=np.str)
names = names[:, :1]
dnevi20=np.linspace(0, N,N)
snovi=np.linspace(0,10,11)
denar20=[]
ind=[0]
cena=3
meja=[]
teden=["Ponedeljek", "Torek", "Sreda", "Četrtek", "Petek","Sobota","Nedelja"]
i=0
for i in range(len(dnevi)):
    b=izdelava_b(prihranki)
    if i%7==0 and i!=0:
        data = np.genfromtxt(data_file)
        data = data[:,1:]
        #data=np.delete(np.array(data),start,0)
        names = np.genfromtxt(data_file, dtype=np.str)
        names = names[:, :1]
        prihranki+=20
    if i%7!=0:
        data=np.delete(np.array(data),ind,0)
        names=np.delete(np.array(names),ind,0)
    ind=[]
    if cena>prihranki:
        cena=prihranki/4
        b=izdelava_b(cena)
        meja.append(i)
    A=izdelava_A(data,ind)
    c=izdelava_c(data,ind)
    sol = opt.linprog(c, A.T, b)
    x=np.array(sol.x)
    vector_b=izdelava_b_vector(data,ind,x)
    resoult, indexation=resoult_uporabljene(x)
    for j in indexation:
        print(names[int(j)])
    for j in indexation:
        ind.append(j)
    if i==0:
        ind=indexation
        start=indexation
    if prihranki<0:
        meja20=i
        dnevi20=np.linspace(0,i,i)
        break
    if (i+1)%7!=0 and (i%7!=0):
        money=vector_b.T[9]
        money_sum=np.sum(money)
        prihranki=prihranki-money_sum
        denar20.append(prihranki)
    else: 
        money=0
        money_sum=np.sum(money)
        prihranki=prihranki-money_sum
        denar20.append(prihranki)
    kolicine=np.array(izdelava_kolicin(vector_b, i))
    element=np.append(kolicine,prihranki)
    print(sum(vector_b[0]))
    relavantne=izdelava_relavantnih(x,names,teden[i%7],i,money,sum(vector_b.T[0]))
    MATRIKA.append(element)
    print(i)
plt.show()
plt.plot(dnevi5,denar5, label="5 evrov prihodka")
plt.plot(dnevi10,denar10, label="10 evrov prihodka")
plt.plot(dnevi20,denar20, label="20 evrov prihodka")
plt.title("Prikaz spreminjanja prihodkov v odvisnosti od časa")
plt.xticks(dnevi,x_label, rotation=45, ha="right", rotation_mode="anchor")
plt.ylabel("Denar [$]")
plt.legend()
plt.show()


top = MATRIKA[0]
bottom = np.zeros_like(top)
width = depth = 1
fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(111, projection='3d')
x3 = dnevi
y3 = snovi
z3 = np.zeros(11)

dx = np.ones(11)
dy = np.ones(11)
dz = MATRIKA[0]
print(MATRIKA[0])

ax1.bar3d(x3, y3, z3, dx, dy, dz)
ax1.set_title('Shaded')
plt.show()

A_ub = np.array([-i[1:8] for i in data])
A_ub = np.concatenate((A_ub, np.array([[0.1 for i in data]]).T), axis=1) #weight
A_ub = np.concatenate((A_ub, np.array([[-i[8] for i in data]]).T), axis=1)
A_ub = np.concatenate((A_ub, np.array([[i[8] for i in data]]).T), axis=1)
A_ub = np.concatenate((A_ub, np.array([[i[9] for i in data]]).T), axis=1)

b_ub = np.array([-70, -310, -50, -1000, -18, -60, -3500, 2, -500, 2400, 5])

sol = opt.linprog(c, A_ub.T, b_ub)
nutrients = np.array([	"mascobe",	"ogljikovi hidrati",	"proteini",	"Ca",	"Fe", "Vitamin C",	"K", "Na"])

x=np.array(sol.x)
resoult=[]
indexation=[]
for i in range(len(x)):
    if x[i]<10**(-6):
        resoult.append(0)
    else:
        resoult.append(x[i])
        indexation.append(i)

sum=0
k=0
for i in indexation:
    sum+=sol.x[i]


resoult_not=[]
resoult=[]
for i in indexation:
    resoult.append(sol.x[i]/sum)
    resoult_not.append(sol.x[i])


plt.bar(x=[i for i in range(len(resoult_not))], height=resoult_not)
plt.xticks([i for i in range(len(resoult_not))],labels=[names[int(i), 0] for i in indexation])

plt.xlabel("Živilo")
plt.ylabel("Masa živila na 100g")
plt.title("Optimizacija kalorij, zgolj relavantne količine")

#plt.yscale("log")
vector_b = np.array([sol.x[i] * data[i] for i in range(len(data))])
sum_mass=0
sum_price=0
cal=vector_b.T[0]
cal_sum=np.sum(cal)
mast=vector_b.T[1]
mast_sum=np.sum(mast)
carb=vector_b.T[2]
carb_sum=np.sum(carb)
prot=vector_b.T[3]
prot_sum=np.sum(prot)
Ca=vector_b.T[4]
Ca_sum=np.sum(Ca)*0.001
Fe=vector_b.T[5]
Fe_sum=np.sum(Fe)*0.001
C=vector_b.T[6]
C_sum=np.sum(C)*0.001
K=vector_b.T[7]
K_sum=np.sum(K)*0.001
Na=vector_b.T[8]
Na_sum=np.sum(Na)*0.001
Cena=vector_b.T[9]
Cena_sum=np.sum(Cena)
for i in range(len(vector_b)):
    sum_price+=vector_b[i][len(vector_b[0])-1]
    for j in range(len(vector_b[0])):
        if j<4 and j!=len(vector_b)-1:
            sum_mass+=vector_b[i][j]
        elif j!=len(vector_b)-1:
            sum_mass+=vector_b[i][j]*0.001
plt.show()
legend=["Masa: %.2f" %(sum_mass), "Cena: %.2f" %(sum_price)]
plt.pie(resoult, labels = [names[int(i), 0] for i in indexation], autopct='%1.1f%%')
plt.xlabel("živilo")
plt.legend(legend)
plt.title("Optimalna razporeditev hrane za najmanj kalorij")
# show plot
plt.show()


legend=["Masa: %.2f" %(sum_mass), "Cena: %.2f" %(sum_price)]
plt.subplot(2,1,1)
plt.pie([mast_sum,carb_sum,prot_sum,Ca_sum+Fe_sum+C_sum+K_sum+Na_sum], labels = ["mascobe",	"ogljikovi hidrati",	"proteini", "minerali"], autopct='%1.1f%%')
plt.legend(legend,loc="lower left")
plt.title("Optimalna razporeditev hrane za najmanj kalorij, snovi")
plt.subplot(2,1,2)
plt.pie([Ca_sum,Fe_sum+C_sum,K_sum,Na_sum], labels = ["Ca",	"Fe+ Vitamin C",	"K", "Na"], autopct='%1.1f%%')
plt.title("Optimalna razporeditev hrane za najmanj kalorij, minerali")
# show plot
plt.show()