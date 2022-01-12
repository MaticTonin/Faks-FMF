import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import time
N=500000 #KORAKI
lenght_T= 70 # Dolčina časa
h=lenght_T/N
k=0.1
y_zunaj=-5
y_0=21

start1 = time.time()
def Derivate(y_0):
    value=-k*(y_0-y_zunaj)
    return value

"""Eulerjeva metoda ___
kjer aproksimiramo odvod kot  y_{n+1}=y(x)+h dy/dx_x """

def Derivate(y_0):
    value=-k*(y_0-y_zunaj)
    return value

def Eulerjeva(y_0, h):
    y_1=y_0+h*Derivate(y_0)
    return y_1

t=np.linspace(0,lenght_T, N)
def izdelava_Euler(N,h,y_0):
    Euler=[]
    for i in range(N):
        if i==0:
            Euler.append(y_0)
        if i>=1:
            Euler.append(Eulerjeva(Euler[i-1],h))
    return Euler

"""Simetrizirana Eulerjeva metoda"""

def Simetric_Euler(y_minus,y_middle, h):
    y_plus=y_minus+2*h*Derivate(y_middle)
    return y_plus

def izdelava_Simetric(y_0, h, N):
    Simetric=[]
    for i in range(N):
        if i==0:
            Simetric.append(y_0)
        if i==1:
            Simetric.append(Eulerjeva(Simetric[i-1], h))
        if i>=2:
            Simetric.append(Simetric_Euler(Simetric[i-2],Simetric[i-1], h))
    return Simetric

"""Heunova metoda"""
def Heun_metod(y_0, h):
    y_delta=y_0+h*Derivate(y_0)
    y= y_0+ h/2 *(Derivate(y_0)+Derivate(y_delta))
    return y
def izdelava_Heun(y_0,h, N):
    Heun=[]
    for i in range(N):
        if i==0:
            Heun.append(y_0)
        if i>=1:
            Heun.append(Heun_metod(Heun[i-1],h))
    return Heun
"""Midpoint metoda"""
def Midpoint_method(y_0, h):
    k_1=Derivate(y_0)
    k_2=Derivate(y_0+1/2*h*k_1)
    y_1=y_0+h*k_2
    return y_1
def izdelava_Midpoint(y_0, h, N):
    Midpoint=[]
    for i in range(N):
        if i==0:
            Midpoint.append(y_0)
        if i>=1:
            Midpoint.append(Midpoint_method(Midpoint[i-1],h))
    return Midpoint

"""Runge-Kutta metoda s 4 koraki"""
def RK4_method(y_0, h):
    k_1=Derivate(y_0)
    k_2=Derivate(y_0+h/2*k_1)
    k_3=Derivate(y_0+h/2*k_2)
    k_4=Derivate(y_0+h*k_3)
    y_1=y_0+h/6*(k_1+2*k_2+2*k_3+k_4)
    return y_1
def izdelava_RK4(y_0, h, N):
    RK4=[]
    for i in range(N):
        if i==0:
            RK4.append(y_0)
        if i>=1:
            RK4.append(RK4_method(RK4[i-1],h))
    return RK4
end1 = time.time()
def rkf( f, a, b, x0, tol, hmax, hmin ):
    """Runge-Kutta-Fehlberg method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        t, x = rkf(f, a, b, x0, tol, hmax, hmin)

    INPUT:
        f     - function equal to dx/dt = f(x,t)
        a     - left-hand endpoint of interval (initial condition is here)
        b     - right-hand endpoint of interval
        x0    - initial x value: x0 = x(a)
        tol   - maximum value of local truncation error estimate
        hmax  - maximum step size
        hmin  - minimum step size

    OUTPUT:
        t     - NumPy array of independent variable values
        x     - NumPy array of corresponding solution function values

    NOTES:
        This function implements 4th-5th order Runge-Kutta-Fehlberg Method
        to solve the initial value problem

           dx
           -- = f(x,t),     x(a) = x0
           dt

        on the interval [a,b].

        Based on pseudocode presented in "Numerical Analysis", 6th Edition,
        by Burden and Faires, Brooks-Cole, 1997.
    """

    # Coefficients used to compute the independent variable argument of f

    a2  =   2.500000000000000e-01  #  1/4
    a3  =   3.750000000000000e-01  #  3/8
    a4  =   9.230769230769231e-01  #  12/13
    a5  =   1.000000000000000e+00  #  1
    a6  =   5.000000000000000e-01  #  1/2

    # Coefficients used to compute the dependent variable argument of f

    b21 =   2.500000000000000e-01  #  1/4
    b31 =   9.375000000000000e-02  #  3/32
    b32 =   2.812500000000000e-01  #  9/32
    b41 =   8.793809740555303e-01  #  1932/2197
    b42 =  -3.277196176604461e+00  # -7200/2197
    b43 =   3.320892125625853e+00  #  7296/2197
    b51 =   2.032407407407407e+00  #  439/216
    b52 =  -8.000000000000000e+00  # -8
    b53 =   7.173489278752436e+00  #  3680/513
    b54 =  -2.058966861598441e-01  # -845/4104
    b61 =  -2.962962962962963e-01  # -8/27
    b62 =   2.000000000000000e+00  #  2
    b63 =  -1.381676413255361e+00  # -3544/2565
    b64 =   4.529727095516569e-01  #  1859/4104
    b65 =  -2.750000000000000e-01  # -11/40

    # Coefficients used to compute local truncation error estimate.  These
    # come from subtracting a 4th order RK estimate from a 5th order RK
    # estimate.

    r1  =   2.777777777777778e-03  #  1/360
    r3  =  -2.994152046783626e-02  # -128/4275
    r4  =  -2.919989367357789e-02  # -2197/75240
    r5  =   2.000000000000000e-02  #  1/50
    r6  =   3.636363636363636e-02  #  2/55

    # Coefficients used to compute 4th order RK estimate

    c1  =   1.157407407407407e-01  #  25/216
    c3  =   5.489278752436647e-01  #  1408/2565
    c4  =   5.353313840155945e-01  #  2197/4104
    c5  =  -2.000000000000000e-01  # -1/5

    # Set t and x according to initial condition and assume that h starts
    # with a value that is as large as possible.

    t = a
    x = x0
    h = hmax

    # Initialize arrays that will be returned

    T = np.array( [t] )
    X = np.array( [x] )

    while t < b:

        # Adjust step size when we get to last interval

        if t + h > b:
            h = b - t;

        # Compute values needed to compute truncation error estimate and
        # the 4th order RK estimate.

        k1 = h * f( x, t )
        k2 = h * f( x + b21 * k1, t + a2 * h )
        k3 = h * f( x + b31 * k1 + b32 * k2, t + a3 * h )
        k4 = h * f( x + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * h )
        k5 = h * f( x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * h )
        k6 = h * f( x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, \
                    t + a6 * h )

        # Compute the estimate of the local truncation error.  If it's small
        # enough then we accept this step and save the 4th order estimate.

        r = abs( r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6 ) / h
        if len( np.shape( r ) ) > 0:
            r = max( r )
        if r <= tol:
            t = t + h
            x = x + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
            T = np.append( T, t )
            X = np.append( X, [x], 0 )

        # Now compute next step size, and make sure that it is not too big or
        # too small.

        h = h * min( max( 0.84 * ( tol / r )**0.25, 0.1 ), 4.0 )

        if h > hmax:
            h = hmax
        elif h < hmin:
            print ("Error: stepsize should be smaller than %e." % hmin)
            break

    # endwhile

    return ( T, X )
def f( y_0, t ):
#        return x * numpy.sin( t )
    return -k*(y_0-y_zunaj)
"""Dejanska funkcija """
def main_function(t):
    y=y_zunaj+np.exp(-k*t)*(y_0-y_zunaj)
    return y
y_main=main_function(t)
Euler=izdelava_Euler(N,h,y_0)
Simetric=izdelava_Simetric(y_0,h,N)
Heun=izdelava_Heun(y_0,h,N)
Midpoint=izdelava_Midpoint(y_0,h,N)
RK4=izdelava_RK4(y_0, h, N)
t_rkf500000, x_rkf500000 = rkf( f, 0, lenght_T, y_0, h, 1, h )
plt.plot(t,Euler,'-',color="blue", label="Eulerjeva")
plt.plot(t,Simetric,'-',color="green",label="Eulerjeva simetrična")
plt.plot(t,y_main,'-',color="red", label="Dejanska funkcija")
plt.plot(t,Heun,'-',color="purple", label="Heunova")
plt.plot(t,Midpoint,'-',color="orange", label="Midpoint")
plt.plot(t,RK4,'-',color="pink", label="Runge Kutta 4")
plt.title('Prikaz približkov')
plt.xlabel("t")
plt.ylabel("T(t)")
plt.legend()
plt.show()


def Error(Y,value):
    Error=[]
    for i in range(len(Y)):
        Error.append(abs(Y[i]-value[i]))
    return Error
y_mainr500000=main_function(t_rkf500000)
plt.plot(t,np.log(Error(Euler,y_main)),'.',color="blue", label="Eulerjeva")
plt.plot(t,np.log(Error(Simetric,y_main)),'.',color="green",label="Eulerjeva simetrična")
plt.plot(t,np.log(Error(Heun,y_main)),'-',color="purple", label="Heunova")
plt.plot(t,np.log(Error(Midpoint,y_main)),'.',color="orange", label="Midpoint")
plt.plot(t,np.log(Error(RK4,y_main)),'-',color="pink", label="Runge Kutta 4")
plt.plot(t_rkf500000,np.log(Error(x_rkf500000,y_mainr500000)),'.', label="Runge Kutta Fehlberg")
plt.title('Prikaz približkov')
plt.xlabel("t")
plt.ylabel("Error(t)")
plt.legend()
plt.show()

ODS=Error(Heun, Midpoint)

plt.plot(t,ODS,'.',color="red", label="Odstopanje Midpoint od Heunove")
plt.title('Odstopanje Midpoint od Heunove')
plt.xlabel("t")
plt.ylabel("Error(t)")
plt.legend()
plt.show()

ODS=Error(Heun, RK4)

plt.plot(t,ODS,'.',color="red", label="Odstopanje RK4 od Heunove")
plt.title('Odstopanje RK4 od Heunove')
plt.xlabel("t")
plt.ylabel("Error(t)")
plt.legend()
plt.show()