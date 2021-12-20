import numpy as np
import numpy.linalg as lin
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.optimize as sp
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data_file = THIS_FOLDER + "\matictonin1.txt"
data1= np.loadtxt(data_file)
n1, ch11, ch12=data1.T
n1=np.array(n1/1000*1.24)
data_file = THIS_FOLDER + "\matictonin2txt.txt"
data2= np.loadtxt(data_file)
n2, ch21, ch22=data2.T
n2=np.array(n2/1000*1.24)
plt.subplot(2,1,1)
plt.title("Meritev valovanja z merilcem moci")
plt.plot(n1,ch11*2-25,"x-", label="Ch1")
plt.plot(n1,ch12*200,"x-", label="Ch2")
plt.legend()
plt.xlabel("x[cm]")
plt.ylabel("U[V]")


plt.subplot(2,1,2)
plt.title("Meritev valovanja s steno")
plt.plot(n2,ch21-15,"x-", label="Ch1")
plt.plot(n2,ch22*20,"x-", label="Ch2")
plt.legend()
plt.xlabel("x[cm]")
plt.ylabel("U[V]")
plt.show()


plt.title("Prikaz obeh valovanj")
plt.plot(n2[1456:4528],ch12[1456:4528]*200+8.32+17.48,"x-", label="Z merilcem moči")
plt.plot(n2[1456:4528],ch22[1456:4528]*20+17.48,"x-", label="Brez merilca moči")
plt.legend()
plt.xlabel("x[cm]")
plt.ylabel("U[mV]")
plt.show()


data_file = THIS_FOLDER + "\Moc.txt"
data2= np.loadtxt(data_file)
U, P=data2.T
plt.title("Prikaz moči")
plt.plot(U,P,"x-", label="Merjenje moči")
plt.legend()
plt.xlabel("U[V]")
plt.ylabel("U[mW]")
plt.show()