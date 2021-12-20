import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.core.numeric import identity

lihe= []
N=200
a=[]
for n in range(N):
    if n%2!=0:
        print('' + ''.join('{:+}'.format(n)) + ',' )
        a.append('' + ''.join('{:+}'.format(n)))
print(a)