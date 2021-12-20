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

def Potencial(x):
    return 1/2*k*x**2

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
        func.append((alpha/(np.pi)**(1/2))**(1/2)*np.exp(-alpha**2*(x[i]-lambd)**2/2))
    return func

def main_func(x,t):
    zeta_l=lambd*alpha
    N=len(x)
    Psi=np.zeros((N,N), dtype=complex)
    d=0
    for i in range(N):
        zeta=alpha*x[i]
        for j in range(N):
            cos=(zeta-zeta_l*np.cos(omega*t[j]))**2
            sin=(omega*t[j]/2+zeta*zeta_l*np.sin(omega*t[j])-1/4*zeta_l**2*np.sin(2*omega*t[j]))
            d=-1/2*cos-1j*sin
            Psi[i,j]=(alpha/(np.pi)**(1/2))**(1/2)*np.exp(d)
    return Psi
omega=0.2
k=omega**2
alpha=k**(1/4)
lambd=10
N=1000
period=2*np.pi/omega
x=np.linspace(-40,40,N)
t=np.linspace(0,period/2,N)
h_x=x[1]-x[0]
h_t=t[1]-t[0]

A=matrix_A(h_x,h_t,x,N)
Wawe=Wawe_func(A,making_psi_0(x))
PSI=main_func(x,t)

"""X,Y=meshgrid(x,t)
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
plt.close()"""

fig, ax = plt.subplots()


line1, = ax.plot(x, Wawe[0,:N].real, color="blue", label="Moja")
line2, = ax.plot(x, x/80, color="red", label="Analitična")



def animate(i):
    line1.set_ydata(Wawe[i,:N].real)  # update the data.
    line2.set_ydata(PSI[:N,i].real)  # update the data.
    return line1, line2


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
h_t="{:.4f}".format(h_t)
plt.xlabel(r"x")
plt.ylabel(r"\psi (x)")
plt.title(r"Prikaz časovnega razvoja Gaussovega paketa za h_t="+str(h_t))
plt.legend()
plt.show()
n=0
while(n==0):
    plt.plot(x, Wawe[n,:N].real, '-', color="blue", label="t={:.2f}".format(t[n]))
    plt.plot(x, PSI[:N,n].real, 'x',color="Red", label="Analitična t={:.2f}".format(t[n]))
    n+=250
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.title(r"Prikaz časovnega razvoja Gaussovega paketa za h_t="+str(h_t))
plt.legend()
plt.show() 


plt.subplot(2, 2, 1)
index=0
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].real, 'x',color="Red", label="t={:.2f}".format(t[index]))
plt.plot(x, Wawe[index,:N].real, '-',color="Blue", label="Analitična t={:.2f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 2)
index=N//5
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].real, 'x',color="Red", label="t={:.2f}".format(t[index]))
plt.plot(x, Wawe[index,:N].real, '-',color="Blue", label="Analitična t={:.2f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 3)
index=N//4
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].real, 'x',color="Red", label="t={:.2f}".format(t[index]))
plt.plot(x, Wawe[index,:N].real, '-',color="Blue", label="Analitična t={:.2f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 4)
index=N//2
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].real, 'x',color="Red", label="t={:.2f}".format(t[index]))
plt.plot(x, Wawe[index,:N].real, '-',color="Blue", label="Analitična t={:.2f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()
plt.show()   

#Imaginarni razvoj
plt.title(r"Prikaz imaginarnega časovnega razvoja Gaussovega paketa za h_t="+str(h_t))
plt.subplot(2, 2, 1)
index=0
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].imag, 'x',color="Red", label="t={:.2f}".format(t[index]))
plt.plot(x, Wawe[index,:N].imag, '-',color="Blue", label="Analitična t={:.2f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")

plt.legend()

plt.subplot(2, 2, 2)
index=N//5
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].imag, 'x',color="Red", label="t={:.2f}".format(t[index]))
plt.plot(x, Wawe[index,:N].imag, '-',color="Blue", label="Analitična t={:.2f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 3)
index=N//4
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].imag, 'x',color="Red", label="t={:.2f}".format(t[index]))
plt.plot(x, Wawe[index,:N].imag, '-',color="Blue", label="Analitična t={:.2f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()

plt.subplot(2, 2, 4)
index=N//2
tindex="{:.3f}".format(t[index])
plt.plot(x, PSI[:N,index].imag, 'x',color="Red", label="t={:.2f}".format(t[index]))
plt.plot(x, Wawe[index,:N].imag, '-',color="Blue", label="Analitična t={:.2f}".format(t[index]))
plt.xlabel(r"x")
plt.ylabel(r"$\psi (x)$")
plt.legend()
plt.show() 

    
plt.plot(x, Wawe[0,:N].real, '-',color="Navy", label="t={:.2f}".format(t[0]))
plt.plot(x, PSI[:N,0].real, '-',color="Red", label="t={:.2f}".format(t[0]))

plt.plot(x, Wawe[N//10,:N].real, '-',color="Orange", label="t={:.2f}".format(t[N//10]))
plt.plot(x, PSI[:N,N//10].real, '-',color="Pink", label="treal={:.2f}".format(t[N//10]))

plt.plot(x, Wawe[N-1,:N].real, '-',color="Green", label="t={:.2f}".format(t[N-1]))
plt.plot(x, PSI[:N,N-1].real, '-',color="Yellow", label="treal={:.2f}".format(t[N-1]))
#plt.plot(x, TK[100,:N_T].real, '-',color="SkyBlue",label="t={:.2f}".format(t[100]))
#plt.plot(x, TF[700,:N_T].real, '-',color="Crimson",label="t={:.2f}".format(t[700]))
plt.xlabel("x")
plt.ylabel("T[K]")
plt.title(r"Test 3")
plt.legend()
plt.show()


#Napake in odstopanja
n=200
Whole=[]
XErrors=[]
Errors=[]
N_bar=[]
Cas=[]
Anal=[]
while n<=500:
    Napaka=np.zeros((n,n))
    x1=np.linspace(-40,40,n)
    t1=np.linspace(0,2*period,n)
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
    n+=100
print("Done")

Error=Errors[0]


"""plt.plot(XErrors[0], Error[0,:len(XErrors[0])], '-',color="Green", label="t={:.2f}".format(t[0]))
#plt.plot(x, TK[100,:N_T].real, '-',color="SkyBlue",label="t={:.2f}".format(t[100]))
#plt.plot(x, TF[700,:N_T].real, '-',color="Crimson",label="t={:.2f}".format(t[700]))
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

