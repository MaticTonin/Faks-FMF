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
c = np.array([i[0] for i in data])

A_ub = np.array([-i[1:8] for i in data])
A_ub = np.concatenate((A_ub, np.array([[0.1 for i in data]]).T), axis=1) #weight
#A_ub = np.concatenate((A_ub, np.array([[-i[8] for i in data]]).T), axis=1)
#A_ub = np.concatenate((A_ub, np.array([[i[8] for i in data]]).T), axis=1)

b_ub = np.array([-70, -310, -50, -1000, 0, 0, 0, 2])
#b_ub = np.array([-70, -310, -50, -1000, -18, -60, -3500, 2, -500, 2400])

sol = opt.linprog(c, A_ub.T, b_ub)

print(np.dot(A_ub.T, sol.x))

plt.scatter([i for i in range(len(sol.x))], sol.x, marker=".")

plt.xlabel("zaporedna številka")
plt.ylabel("optimalna količina [100g]")

plt.title("Optimalna razporeditev hrane, najmanj kalorij")

#plt.yscale("log")

plt.show()
plt.close()

data = np.genfromtxt(data_file)
data = data[:,1:]

names = np.genfromtxt(data_file, dtype=np.str)
names = names[:, :1]

#calories
c = np.array([i[0] for i in data])

A_ub = np.array([-i[1:8] for i in data])
A_ub = np.concatenate((A_ub, np.array([[0.1 for i in data]]).T), axis=1) #weight
A_ub = np.concatenate((A_ub, np.array([[-i[8] for i in data]]).T), axis=1)
A_ub = np.concatenate((A_ub, np.array([[i[8] for i in data]]).T), axis=1)

b_ub = np.array([-70, -310, -50, -1000, -18, -60, -3500, 2, -500, 2400])

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
"""

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(THIS_FOLDER + "\\tabela-zivil.dat", dtype=str)
dataset=data
data_column=x = np.delete(dataset, (0), axis=0)
A_raw=np.delete(data_column, (0), axis=1)

b_type=data[0]
b_limit=[0,-70,-310,-50,-1000,-18,0,0,0,2400]
b, index_B=nf.making_b_string(b_limit)
#b, index_B=nf.making_b_input(data)

A_raw=np.array(A_raw, dtype=float)
A = np.array([-i[1:8] for i in A_raw], dtype=float)
A = np.concatenate((A, np.array([[0.1 for i in A_raw]]).T), axis=1) 
index=0
c=np.transpose(nf.making_c(index, A_raw))
b= np.array([-70, -310, -50, -1000, -18, -60, -3500, 2400])
print(A)
print(b)
print(c)
x=sc.optimize.linprog(c,A_ub=A,b_ub=b)
print(x)
x=np.array(x.x)
resoult=[]
indexation=[]
for i in range(len(x)):
    if x[i]<10**(-6):
        resoult.append(0)
    else:
        resoult.append(x[i])
        indexation.append(i)
print(resoult)
zivila=[]
for i in range(len(data_column)):
    zivila.append([data_column[i][0]])
for i in indexation:
    print(data_column[int(i)][0])
"""