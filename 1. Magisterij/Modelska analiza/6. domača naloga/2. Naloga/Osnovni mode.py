import numpy as np 
import os
import pandas as pd
import gdal
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib import cm
from scipy.optimize import curve_fit

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(THIS_FOLDER + "\\ledvice.dat",dtype=float)
x,y=data.T

a = 13753

def f1(t, a, b):
   return a*np.exp(b*t)

def f2(t, a, b, F):
   return a*np.exp(b*t) + F

def f3(t, a, b, c, d):
   return a*np.exp(b*t) + c*np.exp(d*t) 

def f4(t, a, b, c, d, F):
   return a*np.exp(b*t) + c*np.exp(d*t) + F

def cor(m,n):
    return pcov[m][n]/np.sqrt(pcov[m][m]*pcov[n][n])

t = np.linspace(0, 2200, 1000)
plt.title("Različni modeli z linearno odvisnostjo v eksponentu")
chi = 0
stevec = 0
plt.subplot(1, 2, 1)
popt, pcov = curve_fit(f1, x, y, p0 = [14000, -0.003], method ='lm', sigma = np.sqrt(y))
a,b=popt
chi = 0
for i in range(len(y)):
    chi = chi + ((y[i] - f1(x[i], *popt))/np.sqrt(y[i]))**2
    #print(chi, stevec)
chi = round(chi, 2)
plt.plot(t, f1(t, *popt), color=plt.cm.gist_rainbow(0/4),label=r'fit: $N_1e^{\lambda_1x}$, $\chi^2=%.2f$'%(chi))
plt.errorbar(x,y, np.sqrt(y), fmt ='xk', capsize=3, label='meritve', color=plt.cm.gist_rainbow(4/4))

cor1 =  [[cor(0,0), cor(0,1)],
         [cor(1,0), cor(1,1)]]
chi=0
# plot the heatmap
popt, pcov = curve_fit(f2, x, y, p0 = [14000, -0.003, 100],method ='lm', sigma = np.sqrt(y))
for i in range(len(y)):
    chi = chi + ((y[i] - f2(x[i], *popt))/np.sqrt(y[i]))**2
    #print(chi, stevec)
chi = round(chi, 2)
plt.plot(t, f2(t, *popt), color=plt.cm.gist_rainbow(1/4),label=r'fit: $N_1e^{\lambda_1x} + F$, $\chi^2=%.2f$' %(chi))


cor2 =  [[cor(0,0), cor(0,1), cor(0,2)],
         [cor(1,0), cor(1,1), cor(1,2)],
         [cor(2,0), cor(2,1), cor(2,2)]]
plt.xlabel('Čas')

plt.ylabel('Detekcija')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
popt, pcov = curve_fit(f3, x, y, p0 = [14000, -0.003, 100, -0.001], method ='lm', sigma = np.sqrt(y))
chi=0
for i in range(len(y)):
    chi = chi + ((y[i] - f3(x[i], *popt))/np.sqrt(y[i]))**2
    #print(chi, stevec)
chi = round(chi, 2)
plt.plot(t, f3(t, *popt), color=plt.cm.gist_rainbow(2/4),label=r'fit: $N_1e^{\lambda_1x} + N_2e^{\lambda_2x}$, $\chi^2=%.2f$'%(chi))



cor3 =  [[cor(0,0), cor(0,1), cor(0,2), cor(0,3)],
         [cor(1,0), cor(1,1), cor(1,2), cor(1,3)],
         [cor(2,0), cor(2,1), cor(2,2), cor(2,3)],
         [cor(3,0), cor(3,1), cor(3,2), cor(3,3)],]

popt, pcov = curve_fit(f4, x, y, p0 = [14000, -0.003, 100, -0.001, 100], method ='lm', sigma = np.sqrt(y))
chi=0
for i in range(len(y)):
    chi = chi + ((y[i] - f4(x[i], *popt))/np.sqrt(y[i]))**2
    #print(chi, stevec)
chi = round(chi, 2)
plt.plot(t, f4(t, *popt), color=plt.cm.gist_rainbow(3/4),label=r'fit: $N_1e^{\lambda_1x} + N_2e^{\lambda_2x} + F$, $\chi^2=%.2f$'%(chi))
print(popt)

plt.errorbar(x,y, np.sqrt(y), fmt ='xk', capsize=3, label='meritve', color=plt.cm.gist_rainbow(4/4))

#plt.plot(t, f(t), color='red')



cor4 =  [[cor(0,0), cor(0,1), cor(0,2), cor(0,3), cor(0,4)],
         [cor(1,0), cor(1,1), cor(1,2), cor(1,3), cor(1,4)],
         [cor(2,0), cor(2,1), cor(2,2), cor(2,3), cor(2,4)],
         [cor(3,0), cor(3,1), cor(3,2), cor(3,3), cor(3,4)],
         [cor(4,0), cor(4,1), cor(4,2), cor(4,3), cor(4,4)]]
plt.xlabel('Čas')

plt.ylabel('Detekcija')
plt.grid()
plt.legend()

plt.show()

t = np.linspace(0, 2200, 1000)
plt.title("Odstopanja modelov z linearno odvisnostjo v eksponentu")
chi = 0
stevec = 0
popt, pcov = curve_fit(f1, x, y, p0 = [14000, -0.003], method ='lm', sigma = np.sqrt(y))
a,b=popt
chi = 0
for i in range(len(y)):
    chi = chi + ((y[i] - f1(x[i], *popt))/np.sqrt(y[i]))**2
    #print(chi, stevec)
chi = round(chi, 2)
plt.plot(x, abs(y-f1(x, *popt)), color=plt.cm.gist_rainbow(0/4),label=r'fit: $N_1e^{\lambda_1x}$, $\chi^2=%.2f$'%(chi))
#plt.errorbar(x,y, np.sqrt(y), fmt ='xk', capsize=3, label='meritve', color=plt.cm.gist_rainbow(4/4))

chi=0
# plot the heatmap
popt, pcov = curve_fit(f2, x, y, p0 = [14000, -0.003, 100],method ='lm', sigma = np.sqrt(y))
for i in range(len(y)):
    chi = chi + ((y[i] - f2(x[i], *popt))/np.sqrt(y[i]))**2
    #print(chi, stevec)
chi = round(chi, 2)
plt.plot(x, abs(y-f2(x, *popt)), color=plt.cm.gist_rainbow(1/4),label=r'fit: $N_1e^{\lambda_1x} + F$, $\chi^2=%.2f$' %(chi))


cor2 =  [[cor(0,0), cor(0,1), cor(0,2)],
         [cor(1,0), cor(1,1), cor(1,2)],
         [cor(2,0), cor(2,1), cor(2,2)]]
chi=0
popt, pcov = curve_fit(f3, x, y, p0 = [14000, -0.003, 100, -0.001], method ='lm', sigma = np.sqrt(y))
for i in range(len(y)):
    chi = chi + ((y[i] - f3(x[i], *popt))/np.sqrt(y[i]))**2
    #print(chi, stevec)
chi = round(chi, 2)
plt.plot(x, abs(y-f3(x, *popt)), color=plt.cm.gist_rainbow(2/4),label=r'fit: $N_1e^{\lambda_1x} + N_2e^{\lambda_2x}$, $\chi^2=%.2f$'%(chi))

chi=0
popt, pcov = curve_fit(f4, x, y, p0 = [14000, -0.003, 100, -0.001, 100], method ='lm', sigma = np.sqrt(y))
for i in range(len(y)):
    chi = chi + ((y[i] - f4(x[i], *popt))/np.sqrt(y[i]))**2
    #print(chi, stevec)
chi = round(chi, 2)
plt.plot(x,abs(y-f4(x, *popt)), color=plt.cm.gist_rainbow(3/4),label=r'fit: $N_1e^{\lambda_1x} + N_2e^{\lambda_2x} + F$, $\chi^2=%.2f$'%(chi))
print(popt)


#plt.plot(t, f(t), color='red')
plt.xlabel('Čas')
plt.ylabel('$y-y_{fit}$')
plt.grid()
plt.legend()
plt.show()


t = np.linspace(0, 2200, 1000)
plt.title("Odstopanja modelov z linearno odvisnostjo v eksponentu, log")
chi = 0
stevec = 0
popt, pcov = curve_fit(f1, x, y, p0 = [14000, -0.003], method ='lm', sigma = np.sqrt(y))
a,b=popt
chi = 0
for i in range(len(y)):
    chi = chi + ((y[i] - f1(x[i], *popt))/np.sqrt(y[i]))**2
    #print(chi, stevec)
chi = round(chi, 2)
plt.plot(x, abs(y-f1(x, *popt)), color=plt.cm.gist_rainbow(0/4),label=r'fit: $N_1e^{\lambda_1x}$, $\chi^2=%.2f$'%(chi))
#plt.errorbar(x,y, np.sqrt(y), fmt ='xk', capsize=3, label='meritve', color=plt.cm.gist_rainbow(4/4))
chi=0
# plot the heatmap
popt, pcov = curve_fit(f2, x, y, p0 = [14000, -0.003, 100],method ='lm', sigma = np.sqrt(y))
for i in range(len(y)):
    chi = chi + ((y[i] - f2(x[i], *popt))/np.sqrt(y[i]))**2
    #print(chi, stevec)
chi = round(chi, 2)
plt.plot(x, abs(y-f2(x, *popt)), color=plt.cm.gist_rainbow(1/4),label=r'fit: $N_1e^{\lambda_1x} + F$, $\chi^2=%.2f$' %(chi))


cor2 =  [[cor(0,0), cor(0,1), cor(0,2)],
         [cor(1,0), cor(1,1), cor(1,2)],
         [cor(2,0), cor(2,1), cor(2,2)]]
chi=0
popt, pcov = curve_fit(f3, x, y, p0 = [14000, -0.003, 100, -0.001], method ='lm', sigma = np.sqrt(y))
for i in range(len(y)):
    chi = chi + ((y[i] - f3(x[i], *popt))/np.sqrt(y[i]))**2
    #print(chi, stevec)
chi = round(chi, 2)
plt.plot(x, abs(y-f3(x, *popt)), color=plt.cm.gist_rainbow(2/4),label=r'fit: $N_1e^{\lambda_1x} + N_2e^{\lambda_2x}$, $\chi^2=%.2f$'%(chi))

chi=0
popt, pcov = curve_fit(f4, x, y, p0 = [14000, -0.003, 100, -0.001, 100], method ='lm', sigma = np.sqrt(y))
for i in range(len(y)):
    chi = chi + ((y[i] - f4(x[i], *popt))/np.sqrt(y[i]))**2
    #print(chi, stevec)
chi = round(chi, 2)
plt.plot(x,abs(y-f4(x, *popt)), color=plt.cm.gist_rainbow(3/4),label=r'fit: $N_1e^{\lambda_1x} + N_2e^{\lambda_2x} + F$, $\chi^2=%.2f$'%(chi))
print(popt)


#plt.plot(t, f(t), color='red')
plt.xlabel('Čas')
plt.ylabel('$y-y_{fit}$')
plt.grid()
plt.yscale("log")
plt.legend()
plt.show()

def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

max=0
min=0
fig, (ax1,ax2) = plt.subplots(2)
for i in range(len(cor1)):
    for j in range(len(cor1)):
        if cor1[i][j]>max:
            max=cor1[i][j]
        if cor2[i][j]<min:
            min=cor1[i][j]
        cor1[i][j] = round(cor1[i][j], 3)
fig.suptitle("Kovariančna matrika")
z1_plot=ax1.imshow(cor1, vmin=min, vmax=max, cmap="winter", aspect='auto')
add_colorbar(z1_plot)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(["$N_1$", r"$\lambda_1$"])
ax1.set_yticklabels(["$N_1$", r"$\lambda_1$"])
for (j,i),label in np.ndenumerate(cor1):
    ax1.text(i,j,label,ha='center',va='center')

max=0
min=0
for i in range(len(cor2)):
    for j in range(len(cor2)):
        if cor2[i][j]>max:
            max=cor2[i][j]
        if cor2[i][j]<min:
            min=cor2[i][j]
        cor2[i][j] = round(cor2[i][j], 3)
im=ax2.imshow(cor2, vmin=min, vmax=max, cmap="winter", aspect='auto')
z2_plot=ax2.imshow(cor2, vmin=min, vmax=max, cmap="winter", aspect='auto')
add_colorbar(z2_plot)
ax2.set_xticks([0, 1,2])
ax2.set_yticks([0, 1,2])
ax2.set_xticklabels(["$N_1$", r"$\lambda_1$", "F"])
ax2.set_yticklabels(["$N_1$", r"$\lambda_1$","F"])
for (j,i),label in np.ndenumerate(cor2):
    ax2.text(i,j,label,ha='center',va='center')
plt.show()


max=0
min=0
fig, (ax1,ax2) = plt.subplots(2)
for i in range(len(cor3)):
    for j in range(len(cor3)):
        if cor3[i][j]>max:
            max=cor3[i][j]
        if cor3[i][j]<min:
            min=cor3[i][j]
        cor3[i][j] = round(cor3[i][j], 3)
fig.suptitle("Kovariančna matrika")
z1_plot=ax1.imshow(cor3, vmin=min, vmax=max, cmap="winter", aspect='auto')
add_colorbar(z1_plot)
ax1.set_xticks([0, 1,2,3])
ax1.set_yticks([0, 1,2,3])
ax1.set_xticklabels(["$N_1$", r"$\lambda_1$", "$N_2$",r"$\lambda_2$"])
ax1.set_yticklabels(["$N_1$", r"$\lambda_1$", "$N_2$",r"$\lambda_2$"])
for (j,i),label in np.ndenumerate(cor3):
    ax1.text(i,j,label,ha='center',va='center')

max=0
min=0
for i in range(len(cor4)):
    for j in range(len(cor4)):
        if cor4[i][j]>max:
            max=cor4[i][j]
        if cor4[i][j]<min:
            min=cor4[i][j]
        cor4[i][j] = round(cor4[i][j], 3)
im=ax2.imshow(cor4, vmin=min, vmax=max, cmap="winter", aspect='auto')
z2_plot=ax2.imshow(cor4, vmin=min, vmax=max, cmap="winter", aspect='auto')
add_colorbar(z2_plot)
ax2.set_xticks([0, 1,2,3,4])
ax2.set_yticks([0, 1,2,3,4])
ax2.set_xticklabels(["$N_1$", r"$\lambda_1$","$N_2$", r"$\lambda_2$","F"])
ax2.set_yticklabels(["$N_1$", r"$\lambda_1$","$N_2$",r"$\lambda_2$","F"])
for (j,i),label in np.ndenumerate(cor4):
    ax2.text(i,j,label,ha='center',va='center')
plt.show()






max=0
min=0
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
for i in range(len(cor1)):
    for j in range(len(cor1)):
        if cor1[i][j]>max:
            max=cor1[i][j]
        if cor2[i][j]<min:
            min=cor1[i][j]
        cor1[i][j] = round(cor1[i][j], 3)
z1_plot=ax1.imshow(cor1, vmin=min, vmax=max, cmap="winter", aspect='auto')
add_colorbar(z1_plot)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(["$N_1$", r"$\lambda_1$"])
ax1.set_yticklabels(["$N_1$", r"$\lambda_1$"])
for (j,i),label in np.ndenumerate(cor1):
    ax1.text(i,j,label,ha='center',va='center')

max=0
min=0
for i in range(len(cor2)):
    for j in range(len(cor2)):
        if cor2[i][j]>max:
            max=cor2[i][j]
        if cor2[i][j]<min:
            min=cor2[i][j]
        cor2[i][j] = round(cor2[i][j], 3)
im=ax2.imshow(cor2, vmin=min, vmax=max, cmap="winter", aspect='auto')
z2_plot=ax2.imshow(cor2, vmin=min, vmax=max, cmap="winter", aspect='auto')
add_colorbar(z2_plot)
ax2.set_xticks([0, 1,2])
ax2.set_yticks([0, 1,2])
ax2.set_xticklabels(["$N_1$", r"$\lambda_1$", "F"])
ax2.set_yticklabels(["$N_1$", r"$\lambda_1$","F"])
for (j,i),label in np.ndenumerate(cor2):
    ax2.text(i,j,label,ha='center',va='center')


max=0
min=0
for i in range(len(cor3)):
    for j in range(len(cor3)):
        if cor3[i][j]>max:
            max=cor3[i][j]
        if cor3[i][j]<min:
            min=cor3[i][j]
        cor3[i][j] = round(cor3[i][j], 3)
plt.suptitle("Kovariančna matrika")
z3_plot=ax3.imshow(cor3, vmin=min, vmax=max, cmap="winter", aspect='auto')
add_colorbar(z3_plot)
ax3.set_xticks([0, 1,2,3])
ax3.set_yticks([0, 1,2,3])
ax3.set_xticklabels(["$N_1$", r"$\lambda_1$", "$N_2$",r"$\lambda_2$"])
ax3.set_yticklabels(["$N_1$", r"$\lambda_1$", "$N_2$",r"$\lambda_2$"])
for (j,i),label in np.ndenumerate(cor3):
    ax3.text(i,j,label,ha='center',va='center')

max=0
min=0
for i in range(len(cor4)):
    for j in range(len(cor4)):
        if cor4[i][j]>max:
            max=cor4[i][j]
        if cor4[i][j]<min:
            min=cor4[i][j]
        cor4[i][j] = round(cor4[i][j], 3)
z4_plot=ax4.imshow(cor4, vmin=min, vmax=max, cmap="winter", aspect='auto')
add_colorbar(z4_plot)
ax4.set_xticks([0, 1,2,3,4])
ax4.set_yticks([0, 1,2,3,4])
ax4.set_xticklabels(["$N_1$", r"$\lambda_1$","$N_2$", r"$\lambda_2$","F"])
ax4.set_yticklabels(["$N_1$", r"$\lambda_1$","$N_2$",r"$\lambda_2$","F"])
for (j,i),label in np.ndenumerate(cor4):
    ax4.text(i,j,label,ha='center',va='center')
ax1.title.set_text(r'$N_1e^{\lambda_1x}$')
ax2.title.set_text(r'$N_1e^{\lambda_1x} + F$')
ax3.title.set_text(r'$N_1e^{\lambda_1x}+N_2e^{\lambda_2x} $')
ax4.title.set_text(r'$N_1e^{\lambda_1x} +N_2e^{\lambda_2x}+ F$')
plt.show()

def parameter_maker(x,y,sigma):
    popt, pcov = curve_fit(f1, x, y, method ='lm', sigma = sigma, absolute_sigma= True)
    return popt,pcov
plt.title(r"Smiselnost števila podatkov za $N_1e^{\lambda_1x}$")
plt.errorbar(x, y, np.sqrt(y), label= "Meritve", color= "b",  fmt='.k', capsize=3)
i=0
sigma=np.sqrt(y)
for i in range(len(x)):
    if (i%3==0 and i>2) or i==len(x)-1:
        popt, pcov = curve_fit(f1, x[0:i+1],y[0:i+1], p0 = [14000, -0.003], method ='lm', sigma = np.sqrt(y[0:i+1]))
        chi = 0
        stevec = 0
        for j in y:
            chi = chi + ((j - f1(x[stevec], *popt))/sigma[stevec])**2
    #print(chi, stevec)
            stevec = stevec + 1
        chi = round(chi, 2)
        plt.plot(t, f1(t, *popt), color=plt.cm.gist_rainbow(i/len(x)),label=r"Št podatkov $%.1f$, $\chi=%.2f$" %(i,chi) )
plt.xlabel('Čas')
plt.ylabel('Detekcija')
plt.grid()
plt.legend()
plt.show()


plt.title(r"Smiselnost števila podatkov za $N_1e^{\lambda_1x} +N_2e^{\lambda_2x}+ F$")
plt.errorbar(x, y, np.sqrt(y), label= "Meritve", color= "b",  fmt='.k', capsize=3)
i=0
sigma=np.sqrt(y)
for i in range(len(x)):
    if (i%3==0 and i>8) or i==len(x)-1:
        print(i)
        popt, pcov = curve_fit(f4, x[0:i+1],y[0:i+1], p0 = [14000, -0.003, 100, -0.001, 100], method ='lm', sigma = np.sqrt(y[0:i+1]))
        chi = 0
        stevec = 0
        for j in y:
            chi = chi + ((j - f4(x[stevec], *popt))/sigma[stevec])**2
    #print(chi, stevec)
            stevec = stevec + 1
        chi = round(chi, 2)
        plt.plot(t, f4(t, *popt), color=plt.cm.gist_rainbow(i/len(x)),label=r"Št podatkov $%.1f$, $\chi=%.2f$" %(i,chi) )
plt.xlabel('Čas')
plt.ylabel('Detekcija')
plt.grid()
plt.legend()
plt.show()