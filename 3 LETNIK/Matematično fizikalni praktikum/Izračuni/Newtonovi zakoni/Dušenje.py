from numpy import sin, cos, pi, array
import numpy as np
import scipy.integrate as integrate
import matplotlib
import scipy
import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import time
import os
from diffeq_2 import *
#!/usr/bin/env python

"""A variety of methods to solve first order ordinary differential equations.

AUTHOR:
    Jonathan Senning <jonathan.senning@gordon.edu>
    Gordon College
    Based Octave functions written in the spring of 1999
    Python version: March 2008, October 2008
"""

import numpy

#-----------------------------------------------------------------------------

def euler( f, x0, t ):
    """Euler's method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = euler(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """

    n = len( t )
    x = numpy.array( [x0] * n )
    for i in range( n - 1 ):
        x[i+1] = x[i] + ( t[i+1] - t[i] ) * f( x[i], t[i] )

    return x

#-----------------------------------------------------------------------------

def heun( f, x0, t ):
    """Heun's method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = heun(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """
    n = len( t )
    x = numpy.array( [x0] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        k1 = h * f( x[i], t[i] )
        k2 = h * f( x[i] + k1, t[i+1] )
        x[i+1] = x[i] + ( k1 + k2 ) / 2.0

    return x

#-----------------------------------------------------------------------------

def rk2a( f, x0, t ):
    """Second-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.
       Also known as Midpoint method

    USAGE:
        x = rk2a(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.

    NOTES:
        This version is based on the algorithm presented in "Numerical
        Analysis", 6th Edition, by Burden and Faires, Brooks-Cole, 1997.
    """

    n = len( t )
    x = numpy.array( [ x0 ] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        k1 = h * f( x[i], t[i] ) / 2.0
        x[i+1] = x[i] + h * f( x[i] + k1, t[i] + h / 2.0 )

    return x

#-----------------------------------------------------------------------------

def rk2b( f, x0, t ):
    """Second-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = rk2b(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.

    NOTES:
        This version is based on the algorithm presented in "Numerical
        Mathematics and Computing" 4th Edition, by Cheney and Kincaid,
        Brooks-Cole, 1999.
    """

    n = len( t )
    x = numpy.array( [ x0 ] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        k1 = h * f( x[i], t[i] )
        k2 = h * f( x[i] + k1, t[i+1] )
        x[i+1] = x[i] + ( k1 + k2 ) / 2.0

    return x

#-----------------------------------------------------------------------------

def rku4( f, x0, t ):
    """Fourth-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = rku4(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """

    n = len( t )
    x = numpy.array( [ x0 ] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        k1 = h * f( x[i], t[i] )
        k2 = h * f( x[i] + 0.5 * k1, t[i] + 0.5 * h )
        k3 = h * f( x[i] + 0.5 * k2, t[i] + 0.5 * h )
        k4 = h * f( x[i] + k3, t[i+1] )
        x[i+1] = x[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0

    return x

#-----------------------------------------------------------------------------

def RK4_theta5( f, x0, t ):
    """Fourth-order Runge-Kutta method with error estimate.

    USAGE:
        x, err = RK4_theta5(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
        err   - NumPy array containing estimate of errors at each step.  If
                a system is being solved, err will be an array of arrays.

    NOTES:
        This version is based on the algorithm presented in "Numerical
        Mathematics and Computing" 6th Edition, by Cheney and Kincaid,
        Brooks-Cole, 2008.
    """

    # Coefficients used to compute the independent variable argument of f

    c20  =   2.500000000000000e-01  #  1/4
    c30  =   3.750000000000000e-01  #  3/8
    c40  =   9.230769230769231e-01  #  12/13
    c50  =   1.000000000000000e+00  #  1
    c60  =   5.000000000000000e-01  #  1/2

    # Coefficients used to compute the dependent variable argument of f

    c21 =   2.500000000000000e-01  #  1/4
    c31 =   9.375000000000000e-02  #  3/32
    c32 =   2.812500000000000e-01  #  9/32
    c41 =   8.793809740555303e-01  #  1932/2197
    c42 =  -3.277196176604461e+00  # -7200/2197
    c43 =   3.320892125625853e+00  #  7296/2197
    c51 =   2.032407407407407e+00  #  439/216
    c52 =  -8.000000000000000e+00  # -8
    c53 =   7.173489278752436e+00  #  3680/513
    c54 =  -2.058966861598441e-01  # -845/4104
    c61 =  -2.962962962962963e-01  # -8/27
    c62 =   2.000000000000000e+00  #  2
    c63 =  -1.381676413255361e+00  # -3544/2565
    c64 =   4.529727095516569e-01  #  1859/4104
    c65 =  -2.750000000000000e-01  # -11/40

    # Coefficients used to compute 4th order RK estimate

    a1  =   1.157407407407407e-01  #  25/216
    a2  =   0.000000000000000e-00  #  0
    a3  =   5.489278752436647e-01  #  1408/2565
    a4  =   5.353313840155945e-01  #  2197/4104
    a5  =  -2.000000000000000e-01  # -1/5

    b1  =   1.185185185185185e-01  #  16.0/135.0
    b2  =   0.000000000000000e-00  #  0
    b3  =   5.189863547758284e-01  #  6656.0/12825.0
    b4  =   5.061314903420167e-01  #  28561.0/56430.0
    b5  =  -1.800000000000000e-01  # -9.0/50.0
    b6  =   3.636363636363636e-02  #  2.0/55.0

    n = len( t )
    x = numpy.array( [ x0 ] * n )
    e = numpy.array( [ 0 * x0 ] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        k1 = h * f( x[i], t[i] )
        k2 = h * f( x[i] + c21 * k1, t[i] + c20 * h )
        k3 = h * f( x[i] + c31 * k1 + c32 * k2, t[i] + c30 * h )
        k4 = h * f( x[i] + c41 * k1 + c42 * k2 + c43 * k3, t[i] + c40 * h )
        k5 = h * f( x[i] + c51 * k1 + c52 * k2 + c53 * k3 + c54 * k4, \
                        t[i] + h )
        k6 = h * f( \
            x[i] + c61 * k1 + c62 * k2 + c63 * k3 + c64 * k4 + c65 * k5, \
            t[i] + c60 * h )

        x[i+1] = x[i] + a1 * k1 + a3 * k3 + a4 * k4 + a5 * k5
        x5 = x[i] + b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6

        e[i+1] = abs( x5 - x[i+1] )

    return ( x, e )

#-----------------------------------------------------------------------------

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

    T = numpy.array( [t] )
    X = numpy.array( [x] )

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
        if len( numpy.shape( r ) ) > 0:
            r = max( r )
        if r <= tol:
            t = t + h
            x = x + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
            T = numpy.append( T, t )
            X = numpy.append( X, [x], 0 )

        # Now compute next step size, and make sure that it is not too big or
        # too small.

        h = h * min( max( 0.84 * ( tol / r )**0.25, 0.1 ), 4.0 )

        if h > hmax:
            h = hmax
        elif h < hmin:
            print("Error: stepsize should be smaller than %e." % hmin)
            break

    # endwhile

    return ( T, X )

#-----------------------------------------------------------------------------

def pc4( f, x0, t ):
    """Adams-Bashforth-Moulton 4th order predictor-corrector method

    USAGE:
        x = pc4(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.

    NOTES:
        This function used the Adams-Bashforth-Moulton predictor-corrector
        method to solve the initial value problem

            dx
            -- = f(x,t),     x(t(1)) = x0
            dt

        at the t values stored in the t array (so the interval of solution is
        [t[0], t[N-1]].  The 4th-order Runge-Kutta method is used to generate
        the first three values of the solution.  Notice that it works equally
        well for scalar functions f(x,t) (in the case of a single 1st order
        ODE) or for vector functions f(x,t) (in the case of multiple 1st order
        ODEs).

    """

    n = len( t )
    x = numpy.array( [ x0 ] * n )

    # Start up with 4th order Runge-Kutta (single-step method).  The extra
    # code involving f0, f1, f2, and f3 helps us get ready for the multi-step
    # method to follow in order to minimize the number of function evaluations
    # needed.

    f1 = f2 = f3 = 0
    for i in range( min( 3, n - 1 ) ):
        h = t[i+1] - t[i]
        f0 = f( x[i], t[i] )
        k1 = h * f0
        k2 = h * f( x[i] + 0.5 * k1, t[i] + 0.5 * h )
        k3 = h * f( x[i] + 0.5 * k2, t[i] + 0.5 * h )
        k4 = h * f( x[i] + k3, t[i+1] )
        x[i+1] = x[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0
        f1, f2, f3 = ( f0, f1, f2 )

    # Begin Adams-Bashforth-Moulton steps

    for i in range( 3, n - 1 ):
        h = t[i+1] - t[i]
        f0 = f( x[i], t[i] )
        w = x[i] + h * ( 55.0 * f0 - 59.0 * f1 + 37.0 * f2 - 9.0 * f3 ) / 24.0
        fw = f( w, t[i+1] )
        x[i+1] = x[i] + h * ( 9.0 * fw + 19.0 * f0 - 5.0 * f1 + f2 ) / 24.0
        f1, f2, f3 = ( f0, f1, f2 )

    return x

#-----------------------------------------------------------------------------

def verlet( f, x0, v0, t ):
    """Verlet's 2nd order symplectic method

    USAGE:
        (x,v) = varlet(f, x0, v0, t)

    INPUT:
        f     - function of x and t equal to d^2x/dt^2.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s) of x.  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        v0    - the initial condition(s) of v=dx/dt.  Specifies the value of v when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values for x corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
        v     - NumPy array containing solution values for v=dx/dt corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.

    NOTES:
        This function used the Varlet/Stoermer/Encke (symplectic) method
        method to solve the initial value problem

            dx^2
            -- = f(x),     x(t(1)) = x0  v(t(1)) = v0
            dt^2

        at the t values stored in the t array (so the interval of solution is
        [t[0], t[N-1]].  The 3rd-order Taylor is used to generate
        the first values of the solution.

    """
    n = len( t )
    x = numpy.array( [ x0 ] * n )
    v = numpy.array( [ v0 ] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        iconds0=np.array([x[i],v[i]])
        x[i+1] = x[i] + h * v[i] + (h*h/2) * f(iconds0,t[i])
        iconds1=np.array([x[i+1],v[i]])
        v[i+1] = v[i] + (h/2) * ( f(iconds0,t[i])+f(iconds1,t[i]))

    return numpy.array([x,v])

def pefrl( f, x0, v0, t ):
    """Position Extended Forest-Ruth Like 4th order symplectic method by Omelyan et al.

    USAGE:
        (x,v) = varlet(f, x0, v0, t)

    INPUT:
        f     - function of x and t equal to d^2x/dt^2.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s) of x.  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        v0    - the initial condition(s) of v=dx/dt.  Specifies the value of v when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values for x corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
        v     - NumPy array containing solution values for v=dx/dt corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.

    NOTES:
        This function uses the Omelyan et al (symplectic) method
        method to solve the initial value problem

            dx^2
            -- = f(x),     x(t(1)) = x0  v(t(1)) = v0
            dt^2

        at the t values stored in the t array (so the interval of solution is
        [t[0], t[N-1]].

    """

    xsi=0.1786178958448091
    lam=-0.2123418310626054
    chi=-0.6626458266981849e-1
    n = len( t )
    x = numpy.array( [ x0 ] * n )
    v = numpy.array( [ v0 ] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        y=numpy.copy(x[i])
        w=numpy.copy(v[i])
        iconds0=np.array([x[i],v[i]])
        y += xsi*h*w
        w += (1-2*lam)*(h/2)*f(iconds0,t[i])
        y += chi*h*w
        iconds0=np.array([y,w])
        w += lam*h*f(iconds0,t[i])
        y += (1-2*(chi+xsi))*h*w
        iconds0=np.array([y,w])
        w += lam*h*f(iconds0,t[i])
        y += chi*h*w
        iconds0=np.array([y,w])
        w += (1-2*lam)*(h/2)*f(iconds0,t[i])
        y += xsi*h*w
        x[i+1]=numpy.copy(y)
        v[i+1]=numpy.copy(w)

    return numpy.array([x,v])

#-----------------------------------------------------------------------------
# global variable as frequency

g = 9.81
l = 9.81
omega2 = 2/3
beta=0.5
v=1

#-----------------------------------------------------------------------------
# linear force (e.g. spring pendulum)
def forcel(state,t):
    theta=state[0]
    omega=state[1]
    return -beta*omega-np.sin(theta)+v*np.cos(omega2*t) # sin(y)

#-----------------------------------------------------------------------------
# linear force (e.g. spring pendulum)
def energy(y,v):
    return 0.5*v**2 *l**2 + g*l*(1 - np.cos(y))

#-----------------------------------------------------------------------------
# a simple pendulum y''= F(y) , state = (y,v)
def pendulum(state,t): 
    dydt=np.zeros_like(state)
    dydt[0]=state[1] # x' = v 
    dydt[1]=forcel(state,t)  # v' = F(x)
    return dydt

#-----------------------------------------------------------------------------
# a simple pendulum

def Error(Y,value):
    Error=[]
    for i in range(len(Y)):
        Error.append(abs(Y[i]-value[i]))
    return Error

def analytical_matpend(t, theta0, omega=1.):
    m = numpy.sin(theta0[0]*0.5)
    return (2*numpy.arcsin(m * scipy.special.ellipj(scipy.special.ellipk(m**2) - theta0*t, m**2.)[0]))


#import diffeq

# Velikost za dt=0.1
dt =  0.001
t = np.arange(0.0, 100, dt)
#initial conditions
theta0=1.
vtheta0=0.
#Dejanska vrednost
iconds=np.array([theta0,vtheta0])
main_f=integrate.odeint(pendulum,iconds,t)
Euler_theta=euler(pendulum,iconds,t)
Error_Euler=Euler_theta[:,0]-main_f[:,0]
Heun_theta=heun(pendulum,iconds,t)
Error_Heun_theta=Heun_theta[:,0]-main_f[:,0]
RK4_theta=rku4(pendulum,iconds,t)
Error_RK4_theta=RK4_theta[:,0]-main_f[:,0]
Verle_theta=verlet(forcel,theta0,vtheta0,t)
Error_Velrel_theta=Verle_theta[0,:]-main_f[:,0]
Pefrl_theta=pefrl(forcel,theta0,vtheta0,t)
Error_Pefrl_theta=Pefrl_theta[0,:]-main_f[:,0]
PC4_theta=pc4(pendulum,iconds,t)
Error_PC4_theta=PC4_theta[:,0]-main_f[:,0]

#plt.plot(t,Euler_theta[:,0],'-',color="blue", label="Eulerjeva")
#plt.plot(t,Heun_theta[:,0],'-',color="pink", label="Heunova")
plt.plot(t,RK4_theta[:,0],'-',color="green",label="Runge Kutta 4")
plt.plot(t,main_f[:,0],'-',color="red", label="Dejanska funkcija")
plt.plot(t,Verle_theta[0, :],'-',color="purple", label="Verle")
plt.plot(t,Pefrl_theta[0, :],'-',color="orange", label="Pefrl")
plt.plot(t,PC4_theta[:, 0],'-',color="black", label="PC4")
plt.title('Prikaz približkov za $\u03B8 (t) $, $\omega_0=$' +str(omega2)+ " in dt=" + str(dt))
plt.xlabel("t")
plt.ylabel("$\u03B8 (t)$")
plt.legend()
plt.show()

#energy plots
en_scipy=energy(main_f[:,0],main_f[:,1])
en_euler=energy(Euler_theta[:,0],Euler_theta[:,1])
en_RK4_theta=energy(RK4_theta[:,0],RK4_theta[:,1])
en_pefrl=energy(Pefrl_theta[0,:],Pefrl_theta[1,:])
en_verlet=energy(Verle_theta[0,:],Verle_theta[1,:])
en_true= numpy.array( [ omega2*0.5 ] * len(t) )
en_PC4=energy(PC4_theta[:,0],PC4_theta[:,1])

#plt.plot(t,en_euler,'-',color="blue", label="Eulerjeva")
plt.plot(t,en_RK4_theta,'-',color="green",label="Runge Kutta 4")
plt.plot(t,en_scipy,'-',color="red", label="Main")
plt.plot(t,en_verlet,'-',color="purple", label="Verle")
plt.plot(t,en_pefrl,'-',color="orange", label="Pefrl")
plt.plot(t,en_PC4,'-',color="black", label="PC4")
plt.title('Prikaz Energij za $\u03B8 (t) $, $\omega_0=$' +str(omega2)+ " in dt=" + str(dt))
plt.xlabel("t")
plt.ylabel("$\u03B8 (t)$")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

#plt.plot(t,en_euler,'-',color="blue", label="Eulerjeva")
plt.plot(t,en_RK4_theta,'-',color="green",label="Runge Kutta 4")
plt.plot(t,en_scipy,'-',color="red", label="Main")
#plt.plot(t,en_verlet,'-',color="purple", label="Verle")
#plt.plot(t,en_pefrl,'-',color="orange", label="Pefrl")
plt.plot(t,en_PC4,'-',color="black", label="PC4")
plt.title('Prikaz Energij za $\u03B8 (t) $, $\omega_0=$' +str(omega2)+ " in dt=" + str(dt))
plt.xlabel("t")
plt.ylabel("$\u03B8 (t)$")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

#plt.plot(t,en_euler,'-',color="blue", label="Eulerjeva")
plt.plot(t,en_RK4_theta-en_scipy,'-',color="green",label="Runge Kutta 4")
plt.plot(t,en_verlet-en_scipy,'-',color="purple", label="Verle")
plt.plot(t,en_pefrl-en_scipy,'-',color="orange", label="Pefrl")
plt.plot(t,en_PC4-en_scipy,'-',color="black", label="PC4")
plt.title('Prikaz Napake Energij za $\u03B8 (t) $, $\omega_0=$' +str(omega2)+ " in dt=" + str(dt))
plt.xlabel("t")
plt.ylabel("$\u03B8 (t)$")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
#Veliost dt=0.5
dt =  0.1
t = np.arange(0.0, 1000, dt)
#initial conditions
theta0=0.5
vtheta0=0.
#Dejanska vrednost
iconds=np.array([theta0,vtheta0])
main_f=integrate.odeint(pendulum,iconds,t)
Euler_theta=euler(pendulum,iconds,t)
Heun_theta=heun(pendulum,iconds,t)
Error_Heun_theta=Heun_theta[:,0]-main_f[:,0]
Error_Euler_theta=Euler_theta[:,0]-main_f[:,0]
RK4_theta=rku4(pendulum,iconds,t)
Error_RK4_theta=RK4_theta[:,0]-main_f[:,0]
Verle_theta=verlet(forcel,theta0,vtheta0,t)
Error_Velrel_theta=Verle_theta[0,:]-main_f[:,0]
Pefrl_theta=pefrl(forcel,theta0,vtheta0,t)
Error_Pefrl_theta=Pefrl_theta[0,:]-main_f[:,0]
PC4_theta=pc4(pendulum,iconds,t)
Error_PC4_theta=PC4_theta[:,0]-main_f[:,0]
plt.plot(t,Euler_theta[:,0],'-',color="blue", label="Eulerjeva")
plt.plot(t,RK4_theta[:,0],'-',color="green",label="Runge Kutta 4")
plt.plot(t,Heun_theta[:,0],'-',color="pink", label="Heunova")
plt.plot(t,main_f[:,0],'-',color="red", label="Dejanska funkcija")
plt.plot(t,Verle_theta[0, :],'-',color="purple", label="Verle")
plt.plot(t,Pefrl_theta[0, :],'-',color="orange", label="Pefrl")
plt.plot(t,PC4_theta[:, 0],'-',color="blue", label="PC4")
plt.title('Prikaz približkov za $\u03B8 (t) $, $\omega_0=$' +str(omega2)+ " in dt=" + str(dt))
plt.xlabel("t")
plt.ylabel("$\u03B8 (t)$")
plt.legend()
plt.show()

#plt.plot(t,Error_Euler_theta,'-',color="blue", label="Eulerjeva")
plt.plot(t,Error_RK4_theta,'-',color="green",label="Runge Kutta 4")
#plt.plot(t,Error_Heun_theta,'-',color="pink", label="Heunova")
#plt.plot(t,Error_Velrel_theta,'-',color="purple", label="Verle")
plt.plot(t,Error_Pefrl_theta,'-',color="orange", label="Pefrl")
plt.plot(t,Error_PC4_theta,'-',color="blue", label="PC4")
plt.title('Prikaz napak za $\u03B8 (t) $, $\omega_0=$' +str(omega2)+ " in dt=" + str(dt))
plt.xlabel("t")
plt.ylabel("$\u03B8 (t)$")
plt.legend()
plt.show()


#plt.plot(Euler_theta[:,0],Euler_theta[:,1],'-',color="blue", label="Eulerjeva")
#plt.plot(RK4_theta[:,0],RK4_theta[:,1],'-',color="green",label="Runge Kutta 4")
#plt.plot(Heun_theta[:,0],Heun_theta[:,1],'-',color="pink", label="Heunova")
plt.plot(main_f[:,0],main_f[:,1],'-',color="red", label="Dejanska funkcija")
#plt.plot(Verle_theta[0, :],Verle_theta[1, :],'-',color="purple", label="Verle")
#plt.plot(Pefrl_theta[0, :],Pefrl_theta[1, :],'-',color="orange", label="Pefrl")
#plt.plot(PC4_theta[:, 0],PC4_theta[:, 1],'-',color="black", label="PC4")
plt.title('Fazni portret, $\omega_0=$' +str(omega2)+ " in dt=" + str(dt))
plt.ylabel("$\dot{\u03B8}$")
plt.xlabel("$\u03B8 (t)$")
plt.legend()
plt.show()
print(main_f[:,0])
print(main_f[:,1])

#plt.plot(Euler_theta[:,0]-main_f[:,0],Euler_theta[:,1]-main_f[:,1],'-',color="blue", label="Eulerjeva")
plt.plot(RK4_theta[:,0]-main_f[:,0],RK4_theta[:,1]-main_f[:,1],'-',color="red",label="Runge Kutta 4")
#plt.plot(Heun_theta[:,0]-main_f[:,0],Heun_theta[:,1]-main_f[:,1],'-',color="pink", label="Heunova")
#plt.plot(Verle_theta[0, :]-main_f[:,0],Verle_theta[1, :]-main_f[:,1],'-',color="purple", label="Verle")
plt.plot(Pefrl_theta[0, :]-main_f[:,0],Pefrl_theta[1, :]-main_f[:,1],'-',color="orange", label="Pefrl")
plt.plot(PC4_theta[:, 0]-main_f[:,0],PC4_theta[:, 1]-main_f[:,1],'-',color="blue", label="PC4")
plt.title('Fazni portret napak, $\omega_0=$' +str(omega2)+ " in dt=" + str(dt))
plt.ylabel("$\dot{\u03B8}$")
plt.xlabel("$\u03B8 (t)$")
plt.legend()
plt.show()