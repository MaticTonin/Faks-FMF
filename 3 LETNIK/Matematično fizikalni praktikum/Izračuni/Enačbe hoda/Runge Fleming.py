from operator import length_hint
import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import time

k=0.1
y_zunaj=-5
y_0=21
N=100000 #KORAKI
lenght_T= 70 # Dolčina časa
h=lenght_T/N
k=0.1
y_zunaj=-5
y_0=21
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
A=1
k=0.1
d=10
def f( y_0, t ):
#        return x * numpy.sin( t )
    return -k*(y_0-y_zunaj)+ A*np.sin(2*np.pi/24*(t-d))
def main_function(t):
    return [y_zunaj + (y_0 - y_zunaj)*np.power(np.e, -k*i) 
    + A*(12*k*np.sin(np.pi/12*(i-d))-np.pi*np.cos(np.pi/12*(i-d)))/(12*k*k+np.pi*np.pi/12) 
    + A*np.power(np.e, -k*i)*(12*k*np.sin(np.pi/12*d)+np.pi*np.cos(np.pi/12*d))/(12*k*k+np.pi*np.pi/12)for i in t]
    
t_rkf, x_rkf = rkf( f, 0, lenght_T, y_0, h, 50*h, h )
plt.plot(t_rkf,x_rkf,'-',color="pink", label="Runge Kutta 4")
plt.title('Prikaz približkov')
plt.xlabel("t")
plt.ylabel("Error(t)")
plt.legend()
plt.show()

N=100000 #KORAKI
lenght_T= 70# Dolčina časa
h=lenght_T/N
t=np.linspace(0,lenght_T, N)
y_main=main_function(t)

N=10 #KORAKI
h=lenght_T/N
t10=np.linspace(0,lenght_T, N)
t_rkf10, x_rkf10 = rkf( f, 0, lenght_T, y_0, h, 200*h, h )

print(len(x_rkf10))


N=20 #KORAKI
h=lenght_T/N
t20=np.linspace(0,lenght_T, N)
t_rkf20, x_rkf20 = rkf( f, 0, lenght_T, y_0, h,200*h, h )

N=50 #KORAKI
h=lenght_T/N
t50=np.linspace(0,lenght_T, N)
t_rkf50, x_rkf50 = rkf( f, 0, lenght_T, y_0, h, 200*h, h )

N=100 #KORAKI
h=lenght_T/N
t100=np.linspace(0,lenght_T, N)
t_rkf100, x_rkf100 = rkf( f, 0, lenght_T, y_0, h, 200*h, h )

N=300 #KORAKI
h=lenght_T/N
t300=np.linspace(0,lenght_T, N)
t_rkf300, x_rkf300 = rkf( f, 0, lenght_T, y_0, h, 200*h, h )

N=1000 #KORAKI
h=lenght_T/N
t1000=np.linspace(0,lenght_T, N)
t_rkf1000, x_rkf1000 = rkf( f, 0, lenght_T, y_0, h, 200*h, h )

N=5000 #KORAKI
h=lenght_T/N
t5000=np.linspace(0,lenght_T, N)
t_rkf5000, x_rkf5000 = rkf( f, 0, lenght_T, y_0, h, 100*h, h )


N=10000 #KORAKI
h=lenght_T/N
t10000=np.linspace(0,lenght_T, N)
t_rkf10000, x_rkf10000 = rkf( f, 0, lenght_T, y_0, h, 200*h, h )

N=50000 #KORAKI
h=lenght_T/N
t50000=np.linspace(0,lenght_T, N)
t_rkf50000, x_rkf50000 = rkf( f, 0, lenght_T, y_0, h, 200*h, h )

N=100000 #KORAKI
h=lenght_T/N
t100000=np.linspace(0,lenght_T, N)
t_rkf100000, x_rkf100000 = rkf( f, 0, lenght_T, y_0, h, 200*h, h )

N=500000 #KORAKI
h=lenght_T/N
t500000=np.linspace(0,lenght_T, N)
t_rkf500000, x_rkf500000 = rkf( f, 0, lenght_T, y_0, h, 200*h, h )

N=1000000 #KORAKI
h=lenght_T/N
t1000000=np.linspace(0,lenght_T, N)
t_rkf1000000, x_rkf1000000 = rkf( f, 0, lenght_T, y_0, h, 200*h, h )

plt.plot(t_rkf10,x_rkf10,'-', label="N=10")
plt.plot(t_rkf20,x_rkf20,'-', label="N=20")
plt.plot(t_rkf50,x_rkf50,'-', label="N=50")
plt.plot(t_rkf100,x_rkf100,'-', label="N=100")
plt.plot(t_rkf300,x_rkf300,'-', label="N=300")
plt.plot(t_rkf1000,x_rkf1000,'-', label="N=1000")
plt.plot(t,y_main,'-', label="Dejanska funkcija")
plt.title('Prikaz približkov Runge-Kutta-Fehlberg funkcije v odvisnosti od št korakov')
plt.xlabel("t")
plt.ylabel("T(t)")
#plt.axis([-1,75, -7,25])
plt.legend()
plt.show()

plt.plot(t_rkf10,x_rkf10,'-', label="N=10")
plt.plot(t_rkf20,x_rkf20,'-', label="N=20")
plt.plot(t_rkf50,x_rkf50,'-', label="N=50")
plt.plot(t_rkf100,x_rkf100,'-', label="N=100")
plt.plot(t_rkf300,x_rkf300,'-', label="N=300")
plt.plot(t_rkf1000,x_rkf1000,'-', label="N=1000")
plt.plot(t,y_main,'-', label="Dejanska funkcija")
plt.title('Prikaz približkov Runge-Kutta-Fehlberg funkcije v odvisnosti od št korakov, približana')
plt.xlabel("t")
plt.ylabel("T(t)")
plt.axis([5,20, -5,10])
plt.legend()
plt.show()

plt.plot(t_rkf10000,x_rkf10000,'-', label="N=10000")
plt.plot(t_rkf50000,x_rkf50000,'-', label="N=50000")
plt.plot(t_rkf100000,x_rkf100000,'-', label="N=100000")
plt.plot(t_rkf1000000,x_rkf1000000,'-', label="N=1000000")
plt.plot(t,y_main,'-', label="Dejanska funkcija")
plt.title('Prikaz približkov Runge-Kutta-Fehlberg funkcije v odvisnosti od št korakov')
plt.xlabel("t")
plt.ylabel("T(t)")
#plt.axis([-1,75, -7,25])
plt.legend()
plt.show()

plt.plot(t_rkf10000,x_rkf10000,'-', label="N=10000")
plt.plot(t_rkf50000,x_rkf50000,'-', label="N=50000")
plt.plot(t_rkf100000,x_rkf100000,'-', label="N=100000")
plt.plot(t_rkf1000,x_rkf1000,'-', label="N=1000")
plt.plot(t,y_main,'-', label="Dejanska funkcija")
plt.title('Prikaz približkov Runge-Kutta-Fehlberg funkcije v odvisnosti od št korakov, približana')
plt.xlabel("t")
plt.ylabel("T(t)")
plt.axis([5,20, -5,10])
plt.legend()
plt.show()
def Error(Y,value):
    Error=[]
    for i in range(len(Y)):
        Error.append(abs(Y[i]-value[i]))
    return Error

y_main10=main_function(t_rkf10)
y_main20=main_function(t_rkf20)
y_main50=main_function(t_rkf50)
y_main100=main_function(t_rkf100)
y_main300=main_function(t_rkf300)
y_main1000=main_function(t_rkf1000)
y_main5000=main_function(t_rkf5000)
y_main10000=main_function(t_rkf10000)
y_main50000=main_function(t_rkf50000)
y_main100000=main_function(t_rkf100000)
y_main500000=main_function(t_rkf500000)
y_main1000000=main_function(t_rkf1000000)

plt.plot(t_rkf10,Error(x_rkf10,y_main10),'x', label="N=10")
plt.plot(t_rkf20,Error(x_rkf20,y_main20),'-', label="N=20")
#plt.plot(t_rkf50,Error(x_rkf50,y_main50),'.', label="N=50")
plt.plot(t_rkf100,Error(x_rkf100,y_main100),'--', label="N=100")
#plt.plot(t_rkf300,Error(x_rkf300,y_main300),'.', label="N=300")
plt.plot(t_rkf1000,Error(x_rkf1000,y_main1000),'.', label="N=1000")
plt.title('Prikaz približkov za Runge-Kutta-Fehlberg metodo')
plt.xlabel("t")
plt.ylabel("Error(t)")
plt.legend()
plt.show()


#plt.plot(t_rkf1000,Error(x_rkf1000,y_main1000),'x', label="N=1000")
plt.plot(t_rkf5000,Error(x_rkf5000,y_main5000),'x', label="N=5000")
plt.plot(t_rkf10000,Error(x_rkf10000,y_main10000),'--', label="N=10000")
plt.plot(t_rkf50000,Error(x_rkf50000,y_main50000),'.', label="N=50000")
plt.plot(t_rkf100000,Error(x_rkf100000,y_main100000),'-', label="N=100000")
plt.title('Prikaz približkov za Runge-Kutta-Fehlberg metodo, več korakov')
plt.xlabel("t")
plt.ylabel("Error(t)")
plt.legend()
plt.show()

plt.plot(t_rkf50000,Error(x_rkf50000,y_main50000),'x', label="N=50000")
plt.plot(t_rkf100000,Error(x_rkf100000,y_main100000),'--', label="N=100000")
plt.plot(t_rkf500000,Error(x_rkf500000,y_main500000),'.', label="N=500000")
plt.plot(t_rkf1000000,Error(x_rkf1000000,y_main1000000),'-', label="N=1000000")
plt.title('Prikaz približkov za Runge-Kutta-Fehlberg metodo')
plt.xlabel("t")
plt.ylabel("Error(t)")
plt.legend()
plt.show()