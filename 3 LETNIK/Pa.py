import numpy as np
from numpy.lib.function_base import meshgrid
import scipy.special as spec
import matplotlib.pyplot as plt

import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file1 = os.path.join(THIS_FOLDER, "Črne_luknje.txt")

besedilo= np.loadtxt(my_file1, delimiter=" ", unpack="True", dtype=string)

def pa(besedilo):
    index=0
    for i in besedilo:
        if i="pa":
            index+=1 
            print("Jon is fucking triggered")
    return print(index)
