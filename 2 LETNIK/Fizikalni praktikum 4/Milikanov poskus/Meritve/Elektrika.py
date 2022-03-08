import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

#data location
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "metoda2_U200_in_opis.txt") 

#data load and manipulation
a,b = np.loadtxt(my_file, delimiter="\t", unpack="True")
avgne_0=0
sumne_0=0
avgr=0
sumr=0
e_0=1.6*10**(-19)
#Izračun radija:
for i in range(len(a)):
    r= ((9*18.3*10**(-6)*(a[i]-b[i])*10**(-6))/(4*(973-1.194)*9.81))**(1/2)
    sumr+=r
    print(r*10**6)
    
avgr=sumr/len(a)
print("Povprečen radij je %.2E " %avgr)

print("\n")
print("\n")

#Izračun naboja
for i in range(len(a)):
    r= ((9*18.3*10**(-6)*(a[i]-b[i])*10**(-6))/(4*(973-1.194)*9.81))**(1/2)
    sumr+=r
    ne_0= (3*3.14*r*18.3*10**(-6)*(a[i]+b[i])*10**(-6)*0.005)/200
    if int(ne_0/e_0)==0:
        sumne_0+=ne_0
        
    else:
        sumne_0+=ne_0/(ne_0/e_0)
    print(ne_0*10**(19))
    
avgne_0=sumne_0/(len(a))
print("Povprečen naboj je %.2E " %avgne_0)
print(len(a))
print("\n")
print("\n")



#Funkcija Radij-Hitrost
#fitting function
my_file = os.path.join(THIS_FOLDER, "Radij-hitrosti,E.txt")
x,y = np.loadtxt(my_file, delimiter="\t", unpack="True")
f = lambda x, a, b : a*x + b
args = [20, 0]
fit = sp.curve_fit(f, x, y, p0=args)
fitlabel = "$%.2E \cdot r \quad %.2E$"%(fit[0][0], fit[0][1])
print(fitlabel)
t0 = np.linspace(0.1, 1.15, 100)
#plotting
plt.scatter(x, y, color="black", marker=".")
plt.plot(t0, f(t0, fit[0][0], fit[0][1]), label=fitlabel)
plt.xlabel("Radij [mikro m]")
plt.ylabel("Hitrost [mikro m/s] ")
plt.legend()
print(fit[1])
plt.title("Hitrosti v odvisnosti od radija, olje v električnem polju")
plt.savefig("Elektrika,Radij-hitrost.png")
plt.show()
print("\n")
print("\n")



#Pravilen histogram 
my_file = os.path.join(THIS_FOLDER, "Histogram,po velikosti,H.txt")
n= np.loadtxt(my_file, delimiter="\t", unpack="True")
n_bins = 500
fig, ax = plt.subplots(figsize=(8,4))
n6, bins, patches = ax.hist(n , n_bins, histtype='step', cumulative=True, label="Metoda 1")
ax.grid(True)
ax.set_xlabel('e (C)')
ax.set_ylabel('N(e)')
plt.title("Histogram naših elektronov")
plt.savefig("Pravilen histogram,E.png")
plt.show()
ax.legend(loc='right')
plt.show()


        
        
