import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as sp

import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "Tabela_A.txt")
AUID,RA_s,DEC_s  ,LABEL  ,V,V_eff,BV,BV_eff = np.loadtxt(my_file, delimiter="\t", unpack="True")

print("\hline")
print("% & % & % \\" %(RA_s, DEC_s, BV))

