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
plt.title("Prikaz odvisnosti hallove napetosti od temperature")
T,I,U1,U2=np.loadtxt(THIS_FOLDER + "\\Narascanje.txt", dtype=float).T
I=I/ 1e3
U_H=1/2*(U1-U2)

c = 0.95 * 1e-3
B = 0.173
Ierr = np.array([0.00001 for i in I]) / 1e3
Verr = np.array([0.0001 for i in U_H])

R = 1 / I
Rerr = np.abs(Ierr / I) * R
plt.title("Prikaz odvisnosti upornosti od temperature")
Rh = U_H * c / (I * B)
plt.plot(T+273, R,"x-")
plt.xlabel(r"T[K]")
plt.ylabel(r"R$[\Omega]$")
plt.legend()
plt.show()
Rh = U_H * c / (I * B)
Rherr = (abs(Verr / U_H) + abs(Ierr / I) + 1 / 95 + 1 / 173) * Rh
plt.title("Prikaz odvisnosti Hallove konstante od temperature")
Rh = U_H * c / (I * B)
plt.plot(T+273, Rh,"x-")
plt.xlabel(r"T[K]")
plt.ylabel(r"$R_H[m^3/As]")
plt.legend()
plt.show()
e0 = 1.602176634 * 1e-19

n = - 1 / (Rh * e0)
print(n)
nerr = np.abs(Rherr / Rh) * n

kB = 1.38064852 * 1e-23
T += 273
boltz = 1 / (kB * T)

nerr /= np.abs(n)
n = np.log(np.abs(n))


import scipy.optimize as opt


fit_fun = lambda x, k, n: k * x + n

start_pars =[1, -0.1]

k1 = 4
k2 = 4

pars, cov = opt.curve_fit(fit_fun, boltz[:k1], n[:k1], sigma=nerr[:k1], p0=start_pars, absolute_sigma=True)
pars2, cov2 = opt.curve_fit(fit_fun, boltz[-k2:], n[-k2:], sigma=nerr[-k2:], p0=start_pars, absolute_sigma=True)

JtoeV = 6.242 * 1e+18
print(pars[0] * 2 * JtoeV, np.sqrt(cov[0,0]) * 2 * JtoeV)
print(pars2[0] * 2 * JtoeV, np.sqrt(cov2[0,0]) * 2 * JtoeV)
#plot


import matplotlib.pyplot as plt


plt.errorbar(boltz, n, nerr, marker=".", linestyle="none", capsize=3, fmt="black", label="meritve")
plt.plot(boltz[:k1], [fit_fun(i, *pars) for i in boltz[:k1]], label="model, ko prevladajo valenčni elektroni, $k=%f$"%(pars[0] * 2 * JtoeV))
plt.plot(boltz[-k2:], [fit_fun(i, *pars2) for i in boltz[-k2:]], label="model, ko prevladajo donorski elektroni $k=%f$"%(pars2[0] * 2 * JtoeV))


plt.title("Gostota nosilcev naboja v odvisnosti od Boltzmannovega faktorja, logaritemska skala")
plt.xlabel(r"$\ln(n)[1/m^3]$")
plt.ylabel(r"$\ln(1/k_B T)[1/J]$")
plt.legend(fontsize=10)

#plt.savefig(r"C:\Users\Jaka\Dropbox\Faks\3.1letnik\FP5\Difuzija\plot.pdf", bbox_inches="tight")

plt.show()
plt.close()
plt.title("Prikaz odvisnosti hallove napetosti od temperature")

Rh = U_H * c / (I * B)
plt.plot(T+273, U_H,"x-")
plt.xlabel(r"T[K]")
plt.ylabel(r"$U_H$[V]")
plt.legend()
plt.show()