import os
import numpy as np
from numpy.lib.index_tricks import r_
from seaborn import colors
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from scipy.stats import anderson
from matplotlib import pyplot
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from pingouin import mwu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sp
from matplotlib.ticker import FuncFormatter
# t-test for independent samples

colors = ['#81b882', '#ece0b5', '#d9b26c', '#c57d43', '#934d27']

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
maximums=[]
index=0
kot=[0]
for i in os.listdir(THIS_FOLDER):
    if i[-8:]=="10cm.txt" and index<6:
        print(i[5:8])
        plt.subplot(2,1,1)
        plt.title("Prikaz števila trkov v odvisnosti od kota pri razdalji "+str(i[-9:-4]))
        t,counts,PDF=np.loadtxt(THIS_FOLDER + "\\"+i, dtype=float, skiprows=13).T
        max=0
        for j in counts:
            if j>max:
                max=j
        maximums.append(max)
        if index!=0:
            kot.append(float(i[6:8]))
        plt.plot(t,counts, label=str(i[:-4])+" MAX: "+ str(max))
        plt.legend()
        index+=1
    if i[-8:]=="10cm.txt" and index>=6:
        plt.subplot(2,1,2)
        t,counts,PDF=np.loadtxt(THIS_FOLDER + "\\"+i, dtype=float, skiprows=13).T
        max=0
        for j in counts:
            if j>max:
                max=j
        maximums.append(max)
        kot.append(float(i[6:8]))
        plt.plot(t,counts, label=str(i[:-4])+" MAX: "+ str(max))
        plt.legend()
        index+=1
plt.show()

f1= lambda x, A, T, C : A*np.exp(x/T)+C
#f1= lambda x, M0, T1 : M0*np.exp(-2*x/T1)
args = [614,-1,220]
print(kot)
fit1 = sp.curve_fit(f1, kot, maximums, p0=args)
fitlabel = r"Fit: $MAX=%.2f \cdot exp(\frac{\phi[°]}{%.2f})+ %.2f$"%(fit1[0][0], fit1[0][1],fit1[0][2])
print(fit1[0][0],fit1[0][1],fit1[0][2])
plt.title("Prikaz vhov v odvisnosti od kota pri razdalji 10cm")
plt.plot(kot, maximums, label="Podatki")
plt.plot(kot, f1(kot, fit1[0][0], fit1[0][1], fit1[0][2]), label=fitlabel)
plt.xlabel(r"$\phi[°]$")
plt.ylabel(r"MAX[counts]")
plt.legend()
plt.show()

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
maximums=[]
index=0
kot=[0]
for i in os.listdir(THIS_FOLDER):
    if i[-8:]=="10cm.txt" and index<6:
        print(i[6:8])
        plt.subplot(2,1,1)
        plt.title("Prikaz števila trkov v odvisnosti od kota pri razdalji "+str(i[-8:-4]))
        t,counts,PDF=np.loadtxt(THIS_FOLDER + "\\"+i, dtype=float, skiprows=13).T
        max=0
        for j in counts:
            if j>max:
                max=j
        maximums.append(max)
        if index!=0:
            kot.append(float(i[6:8]))
        plt.plot(t,counts, label=str(i[:-4])+" MAX: "+ str(max))
        plt.legend()
        index+=1
    if i[-8:]=="10cm.txt" and index>=6:
        plt.subplot(2,1,2)
        t,counts,PDF=np.loadtxt(THIS_FOLDER + "\\"+i, dtype=float, skiprows=13).T
        max=0
        for j in counts:
            if j>max:
                max=j
        maximums.append(max)
        kot.append(float(i[6:8]))
        plt.plot(t,counts, label=str(i[:-4])+" MAX: "+ str(max))
        plt.legend()
        index+=1
plt.show()

f1= lambda x, A, T, C : A*np.exp(x/T)+C
#f1= lambda x, M0, T1 : M0*np.exp(-2*x/T1)
args = [614,-1,220]
print(kot)
fit1 = sp.curve_fit(f1, kot, maximums, p0=args)
fitlabel = r"Fit: $MAX=%.2f \cdot exp(\frac{\phi[°]}{%.2f})+ %.2f$"%(fit1[0][0], fit1[0][1],fit1[0][2])
print(fit1[0][0],fit1[0][1],fit1[0][2])
plt.title("Prikaz vhov v odvisnosti od kota pri razdalji 10cm")
plt.plot(kot, maximums, label="Podatki")
plt.plot(kot, f1(kot, fit1[0][0], fit1[0][1], fit1[0][2]), label=fitlabel)
plt.xlabel(r"$\phi[°]$")
plt.ylabel(r"MAX[counts]")
plt.legend()
plt.show()