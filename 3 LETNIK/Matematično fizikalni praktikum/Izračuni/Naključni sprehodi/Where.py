import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation
from scipy.optimize import curve_fit

a = np.arange(10, 100).reshape(9, 10)

print(a)

ind = np.where(a%2 > 0)

print(ind)
print(ind[0])
print(ind[1])
print(ind[0][0])
print(ind[1][0])

ind2 = np.where(a[set(ind[0])]%2>0)
print(i, ind2[0][0], a[i][ind2[0][0]])
