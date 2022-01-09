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
plt.title("Prikaz števila trkov za časovno ločjivost pri kotu 0° in času 303s")
t0,counts0,PDF0=np.loadtxt(THIS_FOLDER + "\\TDC_0_1.txt", dtype=float, skiprows=13).T
t1,counts1,PDF1=np.loadtxt(THIS_FOLDER + "\\TDC_1_1.txt", dtype=float, skiprows=13).T

f12= lambda x, M0, T1,C : M0*np.exp(-x/T1)+C
#f1= lambda x, M0, T1 : M0*np.exp(-2*x/T1)
args = [970,1.2,100]
fit12 = sp.curve_fit(f12, t0, counts0, p0=args)
fit11 = sp.curve_fit(f12, t1, counts1, p0=args)
fitlabel0 = r"TDC_0_1 Count$=%.2f\cdot \exp(\frac{t}{%.2f})+%.2f$"%(fit12[0][0], fit12[0][1],fit12[0][2])
fitlabel1 = r"TDC_1_1 Count$=%.2f\cdot \exp(\frac{t}{%.2f})%.2f$"%(fit11[0][0], fit11[0][1],fit11[0][2])
plt.plot(t0, counts0, label="TDC_0")
plt.plot(t1, counts1, label="TDC_1")
plt.plot(t0, f12(t0, fit12[0][0], fit12[0][1], fit12[0][2]), label=fitlabel0)
plt.plot(t0, f12(t0, fit11[0][0], fit11[0][1], fit11[0][2]), label=fitlabel1)
plt.xlabel(r"t[ns]")
plt.ylabel("Counts")
plt.legend()
plt.show()