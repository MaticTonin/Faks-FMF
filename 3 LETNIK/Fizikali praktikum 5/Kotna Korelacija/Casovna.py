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
t,counts,PDF=np.loadtxt(THIS_FOLDER + "\\Časovna ločljivost.txt", dtype=float, skiprows=13).T
plt.plot(t, counts, label="Stddev=0.0354")
plt.xlabel(r"t[ns]")
plt.ylabel("Counts")
plt.legend()
plt.show()