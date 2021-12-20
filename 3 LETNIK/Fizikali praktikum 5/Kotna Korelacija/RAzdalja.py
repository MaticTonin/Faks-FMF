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
l,counts=np.loadtxt(THIS_FOLDER + "\\Razdalja.txt", dtype=float ).T
f1= lambda x, A, T, C : A*np.exp(x/T)+C
#f1= lambda x, M0, T1 : M0*np.exp(-2*x/T1)
args = [614,-1,220]
fit1 = sp.curve_fit(f1, l, counts, p0=args)
length=np.linspace(l[0],l[len(l)-1],100)
fitlabel = r"Fit: $MAX=%.2f \cdot exp(\frac{l[cm]}{%.2f})+ %.2f$"%(fit1[0][0], fit1[0][1],fit1[0][2])
print(fit1[0][0],fit1[0][1],fit1[0][2])
plt.title("Prikaz maksimumov v odvisnosti od razdalje")
plt.plot(length, f1(length, fit1[0][0], fit1[0][1], fit1[0][2]), label=fitlabel)
plt.plot(l,counts, label="Podatki")
plt.xlabel(r"$l[cm]$")
plt.ylabel(r"MAX[counts]")
plt.legend()
plt.show()