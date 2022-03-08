import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Razlika, vzpor.txt")
my_file1 = os.path.join(THIS_FOLDER, "Razlika, prav.txt")
#data load and manipulation
Tv, Cv= np.loadtxt(my_file, delimiter="\t", unpack="True")
Tp, Cp= np.loadtxt(my_file1, delimiter="\t", unpack="True")

C0p=0.06
C0v=0.05
Ep=[]
Ev=[]
Tpp=[]
Tvv=[]
for i in range(len(Cp)):
    Ep.append(Cp[i]/C0p)
    Tpp.append(Tp[i])
for i in range(len(Cv)):
    Ev.append(Cv[i]/C0v)
    Tvv.append(Tv[i])
print(Ep)

print("\n")
print(Ev)
E=[]
i=0
j=0
dE=[]
for i in range(len(Cp)):
    E.append((2*Cp[i]/C0p+Cv[i]/C0v)/3)
for i in range(len(Cp)):
    dE.append(Cv[i]/C0v-Cp[i]/C0p)
#plotting
fig = plt.figure()
ax = fig.add_subplot(111)


ax.plot(Tvv, dE, color="green", marker=".", label="\\epsilon,razlika")

for i in range(len(Cp)):
    if np.abs(Ep[i]-Ev[i])<=500:
        print("Vrednost T vzporeden je %.2E, epsilon je %.2E " %(Tvv[i], Ev[i]))
        print("Vrednost T pravokoten je %.2E epsilon je %.2E " %(Tpp[i],Ep[i]))

#ax.errorbar(x, y, label='meritve', barsabove="True", linestyle="None", color="black", capsize=5)
#ax.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

#ax.set_xlim(0.45, 0.57)
#ax.set_ylim(-1.3, 0.33)

plt.xlabel("$T[K]$")
plt.ylabel("$\\epsilon$")
plt.legend()
plt.title("Graf iskanja faznega prehoda")

plt.savefig("Fazni prehod,1.png")
plt.show()
plt.close()
