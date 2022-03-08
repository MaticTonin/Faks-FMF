import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('VBstd.wcs')
print(data[:][0])

x = []
y = []

for i in range(len(data)):
    x.append(data[i][4]+0.87)
    y.append(data[i][2]+4.47)




fig, axis = plt.subplots(figsize=(8,6))
axis.scatter(x,y,marker='.', color='purple')
axis.invert_yaxis()
axis.set_xlabel('B-V')
axis.set_ylabel('V')


xkoleno = 0.457
ykoleno = 12.671

xsonce = 0.656
ysonce = 14.67

plt.scatter(xsonce, ysonce, marker='x', color='yellow')
plt.scatter(xkoleno, ykoleno, marker='x', color='red')
plt.ylim(18.5, 10)
plt.title("HR diagram kopice M67")
plt.savefig("HRdiagram.png")
plt.show()
