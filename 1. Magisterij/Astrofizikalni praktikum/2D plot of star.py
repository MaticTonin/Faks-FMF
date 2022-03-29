from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np
dt=100
fig = plt.figure()
ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1))
line, = plt.plot([], [], ".", lw=5, label="Time:0")
palette = ['blue', 'red', 'green', 
           'darkorange', 'maroon', 'black']
L=plt.legend(loc=1)
x_all=np.array([])
y_all=np.array([])
n = 1000
koti=np.linspace(0,2*np.pi,5)

def init():
    line.set_data([], [])
    return line,

def animate(j):
    x_all=np.array([])
    y_all=np.array([])
    for i in koti:
        angle = np.linspace(i+2*np.pi*j/100,2*np.pi+i+2*np.pi*j/50, n)
        radius = np.linspace(0,1.0-j/100,n)
        x = radius * np.cos(angle)
        x=np.array(x)
        x_all=np.concatenate((x_all,x[1:]))
        y = radius * np.sin(angle)
        y=np.array(y)
        y_all=np.concatenate((y_all,y[1:]))
    line.set_data(x_all[1:],y_all[1:])
    lab = 'Time:'+str(round(dt+dt*j,2))
    L.get_texts()[0].set_text(lab)
    return line,
  
plt.title("Bar Chart Animation")

anim =FuncAnimation(fig, animate, init_func=init,
                               frames=50, interval=100)

plt.show()




import os
from sklearn import preprocessing
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 

n = 265
angle = np.linspace(0,2*np.pi, n)
radius = np.linspace(0,1.,n)

x = radius * np.cos(angle)
y = radius * np.sin(angle)

angle1 = np.linspace(np.pi,2*np.pi+np.pi, n)
radius1 = np.linspace(0,1,n)

x1 = radius1 * np.cos(angle1)
y1 = radius1 * np.sin(angle1)

angle2 = np.linspace(3*np.pi/2,2*np.pi+3*np.pi/2, n)
radius2 = np.linspace(0,1,n)
 
x2 = radius2 * np.cos(angle2)
y2 = radius2 * np.sin(angle2)

angle3 = np.linspace(np.pi/2,2*np.pi+np.pi/2, n)
radius3 = np.linspace(0,1,n)

x3 = radius3 * np.cos(angle3)
y3 = radius3 * np.sin(angle3)

plt.plot(x,y, "x-", c = angle,color="red")
plt.plot(x1,y1, "x-", c = angle1 ,color="blue")
plt.plot(x2,y2, "x-", c = angle2,color="green")
plt.plot(x3,y3, "x-", c = angle3,color="yellow")
plt.show()

koti=np.linspace(0,2*np.pi,8)
vrtenje=np.linspace(0,0.5,300)
x_all=np.array([])
y_all=np.array([])
for i in koti:
    angle = np.linspace(i,2*np.pi+i, n)
    radius = np.linspace(0,1.0,n)
    x = radius * np.cos(angle)
    x=np.array(x)
    x_all=np.concatenate((x_all,x))
    y = radius * np.sin(angle)
    y=np.array(y)
    y_all=np.concatenate((y_all,y))
plt.plot(x_all,y_all, "x", c = angle, color="red")
plt.show()
for j in vrtenje:
    for i in koti:
        angle = np.linspace(i,2*np.pi+i, n)
        radius = np.linspace(0,1.0-j,n)
        x = radius * np.cos(angle)
        x=np.array(x)
        x_all=np.concatenate((x_all,x[1:]))
        y = radius * np.sin(angle)
        y=np.array(y)
        y_all=np.concatenate((y_all,y[1:]))
        print(y_all[1:])
    plt.plot(x_all[1:],y_all[1:], "x-", c = angle, color="red")
    plt.pause(60)
plt.show()
