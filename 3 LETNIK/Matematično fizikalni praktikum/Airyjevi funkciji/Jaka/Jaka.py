import numpy as np
import scipy.integrate as intg
import scipy.special as spec
import mpmath as mp
import math as m
import time

mp.prec = 52

alph = 1 / (3 ** (2/3) * spec.gamma(2/3))
beta = 1 / (3 ** (1/3) * spec.gamma(1/3))

def u(s):
    if s == 0:
        return 1
    else:
        return (3*s - 0.5) * (3*s - 1.5) * (3*s - 2.5) / (54 * s * (s - 0.5))

def l(x, s):
    if s == 0:
        return 1
    else:
        return u(s) / x

def p(x, s):
    if s == 0:
        return 1
    else:
        return (- u(2 * s) * u(2*s - 1) / x**2)

def q(x, s):
    if s == 0:
        return u(1) / x
    else:
        return (- u(2*s + 1) * u(2*s) / x**2)

def P(x):
    value = p(x, 0)
    x0 = 1
    k = 1
    while value + x0 != value:
        new = p(x, k)
        if abs(new) < 1:
            x0 *= new
            value += x0
        else: return value
        k += 1
    return value

def Q(x):
    value = q(x, 0)
    x0 = value
    k = 1
    while value + x0 != value:
        new = q(x, k)
        if abs(new) < 1:
            x0 *= new
            value += x0
        else: return value
        k += 1
    return value   

def Ai_maclaurin(x, alph=None, beta=None):
    #calculate Ai(0) via definite integral formula
    if alph == None: alph = 1 / (3 ** (2/3) * spec.gamma(2/3))
    if beta == None: beta = 1 / (3 ** (1/3) * spec.gamma(1/3))

    a = 1
    b = 1

    value = alph - beta * x
            

    f = alph
    g = beta * x

    k = 1
        
    while abs(f - g) + value != value:

        f *= x**3 / (3*k * (3*k - 1)) 
        g *= x**3 / (3*k * (3*k + 1))

        value += f - g
        k += 1

    return value

def Bi_maclaurin(x, alph=None, beta=None):
    #calculate Ai(0) via definite integral formula
    if alph == None: alph = 1 / (3 ** (2/3) * spec.gamma(2/3))
    if beta == None: beta = 1 / (3 ** (1/3) * spec.gamma(1/3))

    a = 1
    b = 1

    value = np.sqrt(3) * (alph + beta * x)


    if x < 0:
        f = np.sqrt(3) * alph
        g = np.sqrt(3) *  beta * x

        k = 1
            
        while abs(f + g) + value != value:
            f *= x**3 / (3*k * (3*k - 1)) 
            g *= x**3 / (3*k * (3*k + 1))

                


            value += f + g
            k += 1
    else:
        f = np.sqrt(3) * alph
        g = np.sqrt(3) *  beta * x

        k = 1
            
        while abs(f + g) + value != value:
            f *= x**3 / (3*k * (3*k - 1)) 
            g *= x**3 / (3*k * (3*k + 1))

                


            value += f + g
            k += 1

    return value

def Ai_asymptotic_negative(x):
    ksi = 2 / 3 * (-x)**(1.5)
    if x < 0:
        a0 =  1 / (np.sqrt(np.pi) * (-x)**(1/4))
        pi = a0 * np.cos(ksi - np.pi / 4) * P(ksi)
        qi = a0 * np.sin(ksi - np.pi / 4) * Q(ksi)
    return pi + qi

def Bi_asymptotic_negative(x):
    ksi = 2 / 3 * (-x)**(1.5)
    if x < 0:
        a0 =  1 / (np.sqrt(np.pi) * (-x)**(1/4))
        pi = - a0 * np.sin(ksi - np.pi / 4) * P(ksi)
        qi = a0 * np.cos(ksi - np.pi / 4) * Q(ksi)
    return pi + qi
    
def Ai_asymptotic(x):
    ksi = 2 / 3 * x**(1.5)
    if x > 0:
        a_0 =  np.exp(- ksi) / (2 * np.sqrt(np.pi) * x**(1/4))
        a_new = l(-ksi, 1)
        a = a_0
        value = a_0
        k = 1
        
        while np.abs(a_new) < 1 and a + value != value:
            a *= a_new
            value += a
            
            k += 1

            a_new = l(- ksi, k)
    return value

def Bi_asymptotic(x):
    ksi = 2 / 3 * x**(1.5)
    if x > 0:
        a_0 =  np.exp(ksi) / (np.sqrt(np.pi) * x**(1/4))
        a_new = l(ksi, 1)
        a = a_0
        value = a_0
        k = 1
        
        while a_new < 1 and a + value != value:
            a *= a_new
            value += a
            
            k += 1

            a_new = l(ksi, k)

    if np.isinf(value): return None
    else: return value

def Ai(x):
    if x < 5.2 and x > -6.8:
        return Ai_maclaurin(x, alph=alph, beta=beta)

    elif x > 0:
        return Ai_asymptotic(x)

    else:
        return Ai_asymptotic_negative(x)

def Bi(x):
    if x < 8.5  and x > -6.8:
        return Bi_maclaurin(x, alph=alph, beta=beta)

    elif x > 0:
        return Bi_asymptotic(x)

    else:
        return Bi_asymptotic_negative(x)

x_list = np.linspace(-12, 1, 20000)

import matplotlib.pyplot as plt
plt.yscale("log")


#relative
if False:
    x1_plot = []
    y1_plot = []
    x2_plot = []
    y2_plot = []
    for x in x_list:
        y1 = Ai(x)
        y2 = Bi(x)

        if y1 != None:
            x1_plot.append(x)
            y1_plot.append(np.abs(y1 / mp.airyai(x) - 1))

        if y2 != None:
            x2_plot.append(x)
            y2_plot.append(np.abs(y2 / mp.airybi(x) - 1))

    plt.scatter(x1_plot, y1_plot, marker=".", c="blue", label="Ai(x)")
    plt.scatter(x2_plot, y2_plot, marker=".", c="red", label="Bi(x)")
#absolute
if False:
    x1_plot = []
    y1_plot = []
    x2_plot = []
    y2_plot = []
    for x in x_list:
        y1 = Ai(x)
        y2 = Bi(x)

        if y1 != None:
            x1_plot.append(x)
            y1_plot.append(np.abs(y1 - mp.airyai(x)))

        if y2 != None:
            x2_plot.append(x)
            y2_plot.append(np.abs(y2 - mp.airybi(x)))

    plt.scatter(x1_plot, y1_plot, marker=".", c="blue", label="Ai(x)")
    plt.scatter(x2_plot, y2_plot, marker=".", c="red", label="Bi(x)")

#absolute, only Ai
if False:
    x1_plot = []
    y1_plot = []
    for x in x_list:
        y1 = Ai_asymptotic(x=x)

        if y1 != None:
            x1_plot.append(x)
            y1_plot.append(np.abs(y1 - mp.airyai(x)))

    plt.scatter(x1_plot, y1_plot, marker=".", c="blue", label="Ai(x)")

#relative, only Bi
if False:
    x1_plot = []
    y1_plot = []
    for x in x_list:
        y1 = Bi_asymptotic(x=x)

        if y1 != None:
            x1_plot.append(x)
            y1_plot.append(np.abs(y1 / mp.airybi(x) - 1))

    plt.scatter(x1_plot, y1_plot, marker=".", c="blue", label="Bi(x)")

#relative, only Ai
if False:
    x1_plot = []
    y1_plot = []
    for x in x_list:
        y1 = i_asymptotic(x=x)

        if y1 != None:
            x1_plot.append(x)
            y1_plot.append(np.abs(y1 / mp.airyai(x) - 1))

    plt.scatter(x1_plot, y1_plot, marker=".", c="blue", label="Bi(x)")

#plot Ai
if False:
    x1_plot = []
    y2_plot = []
    for x in x_list:
        y1 = Ai(x)
        if y1 != None:
            x1_plot.append(x)
            y1_plot.append(y)

    for x in x_list:
        y2 = Bi(x)
        if y2 != None:
            x2_plot.append(x)
            y2_plot.append(y)


    plt.scatter(x_plot, y_plot, marker=".", c="blue")

#last a
if False:
    x1_plot = []
    y1_plot = []
    x2_plot = []
    y2_plot = []
    for x in x_list:
        y1 = Ai_asymptotic(x=x)[0]
        y2 = Ai_asymptotic(x=x)[1]

        if y1 != None:
            x1_plot.append(x)
            y1_plot.append(np.abs(y1 - mp.airyai(x)))

        if y2 != None:
            x2_plot.append(x)
            y2_plot.append(y2)


    plt.scatter(x1_plot, y1_plot, marker=".", c="blue", label="napaka")
    plt.scatter(x2_plot, y2_plot, marker=".", c="red", label="zadnji člen")

#relative, only Ai, test
if False:
    x1_plot = []
    y1_plot = []
    y2_plot = []
    for x in x_list:
        y1 = Ai_asymptotic(x)

        if y1 != None:
            x1_plot.append(x)
            y1_plot.append(np.abs(y1 / mp.airyai(x) - 1))

            eps = 2.5 * 2 / 54 / (2/3 * x**(1.5))
            eps2 = 5.5*4.5*3.5*2.5 / 54**2 / 2 / (2/3 * x**(1.5))**2
            y2_plot.append(eps)

    plt.scatter(x1_plot, y1_plot, marker=".", c="blue", label="Ai(x)")
    plt.scatter(x1_plot, y2_plot, marker=".", c="red", label="Bi(x)")


start_time = time.time()
for x in x_list:
    a = Ai(x)
    b = Bi(x)
end_time = time.time()

print((end_time - start_time))



plt.title("Absolutna napaka izračuna Airyjevih funkcij z asimptotsko vrsto")
#plt.title("Relativna napaka izračuna Airyjevih funkcij z asimptotsko vrsto")

plt.grid()
plt.legend()
plt.show()
plt.close()
