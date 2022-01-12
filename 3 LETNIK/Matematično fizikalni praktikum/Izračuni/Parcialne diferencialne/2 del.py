import numpy as np
from numpy.lib.function_base import meshgrid
import scipy.special as spec
import matplotlib.pyplot as plt
import time
import numpy as np
from mpl_toolkits import mplot3d
from numpy import fft
from scipy import linalg
from scipy import integrate
import scipy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
np.set_printoptions(linewidth=np.inf)

def Potencial(x):
    return 0

def matrix_A(h_x,h_t, x, N):
    b=h_t/(2*h_x**2)*1j
    a=-b/2
    A=np.zeros((N,N), dtype=complex)
    for i in range(N):
        d=1+b+h_t/2*Potencial(x[i])*1j
        A[i,i]=complex(d.real,d.imag)
        A[i,i-1]=complex(a.real,a.imag)
        A[i-1,i]=complex(a.real,a.imag)
    return A


def Wawe_func(A,psi_0):
    A_1=np.matrix.conjugate(A)
    N=len(A[0,:])
    Wawe=[]
    solution=np.array(psi_0, dtype=complex)
    for i in range(N):
        Wawe.append(solution)
        #solution=scipy.linalg.solve_banded()
        solution=np.linalg.solve(A,np.dot(A_1,psi_0))
        psi_0=solution
    return np.asarray(Wawe, dtype=complex)

def making_psi_0(x):
    func=[]
    for i in range(len(x)):
        prvi=np.exp(1j*k*(x[i]-lambd))
        norm=(2*np.pi*sigma**2)**(-1/4)
        drugi=np.exp(-(x[i]-lambd)**2/(2*sigma)**2)
        func.append(norm*prvi*drugi)
    return func

def main_func(x,t):
    N=len(x)
    Psi=np.zeros((N,N), dtype=complex)
    d=0
    for i in range(N):
        for j in range(N):
            norm=(2*np.pi*sigma**2)**(-1/4)
            koren=(1+1j*t[j]/(2*sigma**2))**(-1/2)
            nad=-(x[i]-lambd)**2/(2*sigma)**2+1j*k*(x[i]-lambd)-1j*k**2*t[j]/2
            pod=1+1j*t[j]/(2*sigma**2)
            Psi[i,j]=norm/koren*np.exp(nad/pod)
    return Psi
sigma=1/20
k=50*np.pi
lambd=0.25
N=1000
x=np.linspace(-1.5,1.5,N)
#x=np.linspace(-0.5,1.5,N)
t=np.linspace(0,0.01,N)
h_x=x[1]-x[0]
h_t=t[1]-t[0]

A=matrix_A(h_x,h_t,x,N)
Wawe=Wawe_func(A,making_psi_0(x))
PSI=main_func(x,t)
"""
X,Y=meshgrid(x,t)
Z=Wawe

ax = plt.axes(projection='3d')
ax.contour3D(X, Y , Z, N//2, cmap='gnuplot')
ax.set_title(r"Test 1")
ax.set_xlabel('X')
ax.set_ylabel('t')
ax.set_zlabel('T')
plt.show()



x_gra = np.array([x for i in range(N)])
fig, ax = plt.subplots()
segments = [np.column_stack([x1, y]) for x1, y in zip(x_gra, PSI[:N, :].real)]
lc = LineCollection(segments, cmap='hot')
lc.set_array(np.asarray(t))
ax.add_collection(lc)
ax.autoscale()
ax.set_title(r"Test")
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$T(x)$')
axcb = fig.colorbar(lc)
axcb.set_label(r'$t$')
plt.show()
plt.close()

x_gra = np.array([x for i in range(N)])
fig, ax = plt.subplots()
segments = [np.column_stack([x1, y]) for x1, y in zip(x_gra, Wawe[:, :N].real)]
lc = LineCollection(segments, cmap='gnuplot')
lc.set_array(np.asarray(t))
ax.add_collection(lc)
ax.autoscale()
ax.set_title(r"Test")
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$T(x)$')
axcb = fig.colorbar(lc)
axcb.set_label(r'$t$')
plt.show()
plt.close()
"""
fig, ax = plt.subplots()


line1, = ax.plot(x, Wawe[0,:N].real, color="blue", label="Moja")
line2, = ax.plot(x, x/80, color="red", label="Analitična")
time_template = 't = %.3f'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def animate(i):
    line1.set_ydata(Wawe[i,:N].real)  # update the data.
    line2.set_ydata(PSI[:N,i].real)  # update the data.
    time_text.set_text(time_template %(t[i]))
    return line1, line2, time_text


ani = animation.FuncAnimation(
    fig, animate, interval=20, frames=N, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)
plt.legend()
plt.show()
"""
plt.plot(x, Wawe[0,:N].real, '-',color="Navy", label="t={:.7f}".format(t[0]))
plt.plot(x, PSI[:N,0].real, '-',color="Red", label="t={:.7f}".format(t[0]))

plt.plot(x, Wawe[N//10,:N].real, '-',color="Orange", label="t={:.7f}".format(t[N//10]))
plt.plot(x, PSI[:N,N//10].real, '-',color="Pink", label="treal={:.7f}".format(t[N//10]))

plt.plot(x, Wawe[N-1,:N].real, '-',color="Green", label="t={:.7f}".format(t[N-1]))
plt.plot(x, PSI[:N,N-1].real, '-',color="Yellow", label="treal={:.7f}".format(t[N-1]))
#plt.plot(x, TK[100,:N_T].real, '-',color="SkyBlue",label="t={:.7f}".format(t[100]))
#plt.plot(x, TF[700,:N_T].real, '-',color="Crimson",label="t={:.7f}".format(t[700]))
plt.xlabel("x")
plt.ylabel("T[K]")
plt.title(r"Test 3")
plt.legend()
plt.show()"""
fig, ax = plt.subplots()


line1, = ax.plot(x, Wawe[0,:N].real, color="blue", label="Moja")
line2, = ax.plot(x, x/80, color="red", label="Analitična")
time_template = 't = %.6f'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def animate(i):
    line1.set_ydata(Wawe[i,:N].real)  # update the data.
    line2.set_ydata(PSI[:N,i].real)  # update the data.
    time_text.set_text(time_template %(t[i]))
    return line1, line2, time_text


ani = animation.FuncAnimation(
    fig, animate, interval=20, frames=N, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)
#ani.save("N=1000,Gauss.mp4")
h_t="{:.6f}".format(h_t)
plt.xlabel(r"x")
plt.ylabel(r"\psi (x)")
plt.title(r"Prikaz realnega časovnega razvoja Gaussovega paketa za h_t="+str(h_t))
plt.legend()
plt.show()

fig, ax = plt.subplots()
line1, = ax.plot(x, Wawe[0,:N].real, color="blue", label="Moja")
line2, = ax.plot(x, x/80, color="red", label="Analitična")
time_template = 't = %.6f'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
def animate(i):
    line1.set_ydata(Wawe[i,:N].imag)  # update the data.
    line2.set_ydata(PSI[:N,i].imag)  # update the data.
    time_text.set_text(time_template %(t[i]))
    return line1, line2, time_text


ani = animation.FuncAnimation(
    fig, animate, interval=20, frames=N, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)
#ani.save("N=1000,Gauss.mp4")
plt.xlabel(r"x")
plt.ylabel(r"\psi (x)")
plt.title(r"Prikaz imag. časovnega razvoja Gaussovega paketa za h_t="+str(h_t))
plt.legend()
plt.show()

n=0
while(n==0):
    plt.plot(x, Wawe[n,:N].real, '-', color="blue", label="t={:.7f}".format(t[n]))
    plt.plot(x, PSI[:N,n].real, 'x',color="Red", label="Analitična t={:.7f}".format(t[n]))
    n+=250
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.title(r"Prikaz realnega časovnega razvoja Gaussovega paketa za h_t="+str(h_t))
plt.legend()
plt.show() 


plt.plot(x, Wawe[0,:N].real, '-', color="blue", label="t={:.7f}".format(t[0]))
plt.plot(x, PSI[:N,0].real, '-',color="Red", label="Analitična t={:.7f}".format(t[0]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.title(r"Prikaz realnega časovnega razvoja Gaussovega paketa za h_t="+str(h_t))
plt.legend()
plt.show() 

plt.plot(x, Wawe[0,:N].imag, '-', color="blue", label="t={:.7f}".format(t[0]))
plt.plot(x, PSI[:N,0].imag, '-',color="Red", label="Analitična t={:.7f}".format(t[0]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.title(r"Prikaz imag. časovnega razvoja Gaussovega paketa za h_t="+str(h_t))
plt.legend()
plt.show() 



plt.title(r"Prikaz realnega časovnega razvoja Gaussovega paketa za h_t="+str(h_t))
plt.subplot(2, 2, 1)
index=0
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].real, '-',color="Red", label="t={:.7f}".format(t[index]))
plt.plot(x, Wawe[index,:N].real, '-',color="Blue", label="Analitična t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 2)
index=N//4
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].real, '-',color="Red", label="t={:.7f}".format(t[index]))
plt.plot(x, Wawe[index,:N].real, '-',color="Blue", label="Analitična t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 3)
index=502
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].real, '-',color="Red", label="t={:.7f}".format(t[index]))
plt.plot(x, Wawe[index,:N].real, '-',color="Blue", label="Analitična t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 4)
index=N-1
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].real, '-',color="Red", label="t={:.7f}".format(t[index]))
plt.plot(x, Wawe[index,:N].real, '-',color="Blue", label="Analitična t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()
plt.show()   

#Imaginarni razvoj
plt.title(r"Prikaz imag. časovnega razvoja Gaussovega paketa za h_t="+str(h_t))
plt.subplot(2, 2, 1)
index=0
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].imag, '-',color="Red", label="t={:.7f}".format(t[index]))
plt.plot(x, Wawe[index,:N].imag, '-',color="Blue", label="Analitična t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")

plt.legend()

plt.subplot(2, 2, 2)
index=N//4
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].imag, '-',color="Red", label="t={:.7f}".format(t[index]))
plt.plot(x, Wawe[index,:N].imag, '-',color="Blue", label="Analitična t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 3)
index=N//2
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].imag, '-',color="Red", label="t={:.7f}".format(t[index]))
plt.plot(x, Wawe[index,:N].imag, '-',color="Blue", label="Analitična t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 4)
index=N-3
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].imag, '-',color="Red", label="t={:.7f}".format(t[index]))
plt.plot(x, Wawe[index,:N].imag, '-',color="Blue", label="Analitična t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()
plt.show() 

#Odstopanja
plt.subplot(2, 2, 1)
index=0
tindex="{:.3f}".format(t[index])
plt.plot(x, abs(Wawe[index,:N].real-PSI[:N,index].real), '-',color="Blue", label="Napaka t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 2)
index=N//4
tindex="{:.3f}".format(t[index])
plt.plot(x, abs(Wawe[index,:N].real-PSI[:N,index].real), '-',color="Blue", label="Napaka t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 3)
index=N//2
tindex="{:.3f}".format(t[index])
plt.plot(x, abs(Wawe[index,:N].real-PSI[:N,index].real), '-',color="Blue", label="Napaka t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 4)
index=N-1
tindex="{:.3f}".format(t[index])
plt.plot(x, abs(Wawe[index,:N].real-PSI[:N,index].real), '-',color="Blue", label="Napaka t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()
plt.show()   

#Imaginarni razvoj napake
plt.title(r"Prikaz imag. časovnega razvoja Gaussovega paketa za h_t="+str(h_t))
plt.subplot(2, 2, 1)
index=0
tindex="{:.3f}".format(t[index])
plt.plot(x, abs(Wawe[index,:N].imag-PSI[:N,index].imag), '-',color="Blue", label="Napaka t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")

plt.legend()

plt.subplot(2, 2, 2)
index=N//4
tindex="{:.3f}".format(t[index])
plt.plot(x, abs(Wawe[index,:N].imag-PSI[:N,index].imag), '-',color="Blue", label="Napaka t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 3)
index=N//2
tindex="{:.3f}".format(t[index])
plt.plot(x, abs(Wawe[index,:N].imag-PSI[:N,index].imag), '-',color="Blue", label="Napaka t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 4)
index=N-3
tindex="{:.3f}".format(t[index])
plt.plot(x, abs(Wawe[index,:N].imag-PSI[:N,index].imag), '-',color="Blue", label="Analitična t={:.7f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()
plt.show() 


#Napake in odstopanja
n=500
Whole=[]
XErrors=[]
Errors=[]
N_bar=[]
Cas=[]
Anal=[]
while n<=1200:
    Napaka=np.zeros((n,n))
    x1=np.linspace(-0.5,1.5,N)
    t1=np.linspace(0,0.02,N)
    h_x=x1[1]-x1[0]
    h_t=t1[1]-t1[0]
    start2 = time.time()
    A=matrix_A(h_x,h_t,x1,n)
    Wawe=Wawe_func(A,making_psi_0(x1))
    PSI=main_func(x1,t1)
    end2 = time.time()
    for i in range(n):
        for j in range(n):
            Napaka[i,j]=abs(PSI[j,i]-Wawe[i,j])
    XErrors.append(x1)
    Errors.append(Napaka)
    N_bar.append(n)
    Whole.append(Wawe)
    Anal.append(PSI)
    Cas.append(-start2+end2)
    print(n)
    n+=200
print("Done")

Error=Errors[0]


"""plt.plot(XErrors[0], Error[0,:len(XErrors[0])], '-',color="Green", label="t={:.7f}".format(t[0]))
#plt.plot(x, TK[100,:N_T].real, '-',color="SkyBlue",label="t={:.7f}".format(t[100]))
#plt.plot(x, TF[700,:N_T].real, '-',color="Crimson",label="t={:.7f}".format(t[700]))
plt.xlabel("x")
plt.ylabel("T[K]")
plt.title(r"Napaka 3")
plt.legend()
plt.show()"""


Wawe1=np.asarray(Whole[1])
PSI=np.asarray(Anal[1])
N=int(N_bar[1])
x=np.array(XErrors[1])
fig, ax = plt.subplots()

line1, = ax.plot(x, Wawe1[0,:N].real, color="blue", label="Moja")
line2, = ax.plot(x, x/80, color="red")

def animate1(i):
    line1.set_ydata(Wawe1[i,:N].real)  # update the data.
    line2.set_ydata(PSI[:N,i].real)  # update the data.
    return line1, line2


ani1 = animation.FuncAnimation(
    fig, animate1, interval=20, frames=N, blit=True, save_count=50)
plt.legend()
plt.show()
Wawe1=np.asarray(Whole[2])
PSI=np.asarray(Anal[2])
N=int(N_bar[2])
x=np.array(XErrors[2])
fig, ax = plt.subplots()

line1, = ax.plot(x, Wawe1[0,:N].real, color="green", label="Moja")
line2, = ax.plot(x, x/80, color="yellow")

def animate1(i):
    line1.set_ydata(Wawe1[i,:N].real)  # update the data.
    line2.set_ydata(PSI[:N,i].real)  # update the data.
    return line1, line2


ani2 = animation.FuncAnimation(
    fig, animate1, interval=20, frames=N, blit=True, save_count=50)
plt.legend()
plt.show()

"""fig, ax = plt.subplots()

Wawe2=Whole[1]
N=int(N_bar[1])
x=np.array(XErrors[1])
line3, = ax.plot(x, Wawe2[0,:N].real, color="green", label="Moja")

def animate2(i):
    line3.set_ydata(Wawe2[i,:N].real)  # update the data.
    return line1,line2,line3

ani = animation.FuncAnimation(
    fig, animate2, interval=20, frames=N, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)
plt.legend()
plt.show()"""

