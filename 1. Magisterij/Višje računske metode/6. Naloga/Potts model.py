import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit,njit, objmode
import random
import matplotlib.cm as cm
import time
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from func import Potts, energy, magnetization, energy_c, Potts_mag,Potts_el

N=10
J=1
k=0.5
q=3
n=10**7

start_plot="No"
if start_plot=="Yes":
    fig = plt.figure()
    grid_new, grid_start, E1_list, iterations_list=Potts(q,N,n,J,k)
    #print(grid_new)
    #print(grid_start)
    fig.suptitle(r"Parametri $J=%.2f, k=%.2f, q=%i, n=10^{%i}$"%(J,k,q,np.log10(n)))
    ax1 = fig.add_subplot(121)
    # Bilinear interpolation - this will look blurry
    ax1.imshow(grid_start, cmap=cm.rainbow)
    
    ax2 = fig.add_subplot(122)
    # 'nearest' interpolation - faithful but blocky
    im=ax2.imshow(grid_new, cmap=cm.rainbow)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    plt.show()

Matrix_plot="Yes"
if Matrix_plot=="Yes":
    import matplotlib.gridspec as gridspec
    fig=plt.figure(figsize = (8,8))
    q_list=[2,3,4,5]
    kB_T_list=[0.1,0.5,1,2]
    gs1 = gridspec.GridSpec(int(len(q_list)), int(len(kB_T_list)))
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
    index=0
    fig.suptitle(r"Prikaz razvoja stanja $J=%.2f, n=10^{%i}, N=%i$"%(J,np.log10(n),N))
    fig3 = plt.figure()
    fig3.suptitle(r"Prikaz razvoja energije od iteracij $J=%.2f, N=%i$"%(J, N))
    for i in tqdm(range((len(q_list)))):
        jndex=0
        ax2 = fig3.add_subplot(2,2,1+i)
        for j in tqdm(range(len(kB_T_list))):
            ax1 = fig.add_subplot(gs1[index])
            grid_new, grid_start, E1_list, iterations_list=Potts(q_list[i],N,n,J,kB_T_list[j])
            im=ax1.imshow(grid_new, cmap=cm.rainbow)
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            if j==0:
                ax1.set_yticklabels(["$q=%i$" %q_list[i]])
                ax1.set_yticks([N/2])
            if i==0:
                ax1.set_title(r"$\beta=%.2f$" %kB_T_list[j])
            #if j==len(kB_T_list)-1:
                #divider = make_axes_locatable(ax1)
                #cax = divider.append_axes("right", size="2%", pad=0.05)
                #plt.colorbar(im, cax=cax)   

            plt.axis('on')
            index+=1
            ax2.plot(iterations_list,E1_list, label=r"$\beta=%.2f$" %kB_T_list[j], color=plt.cm.rainbow(jndex/(len(q_list))))
            jndex+=1
        ax2.set_title("$q=%i$" %q_list[i])
        ax2.grid()
        ax2.set_xlabel("N")
        ax2.set_ylabel(r"$\langle E \rangle$")
        ax2.set_xscale("log")
        ax2.legend()
    plt.show()

Energy_plot="No"
if Energy_plot=="Yes":
    import matplotlib.gridspec as gridspec
    fig=plt.figure(figsize = (8,8))
    n=10**5
    q_list=[10,20,50,100]
    kB_T_list=[0.1,0.5,1,2]
    q=1
    gs1 = gridspec.GridSpec(int(len(kB_T_list)), int(len(q_list)))
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
    index=0
    fig.suptitle(r"Prikaz razvoja stanja $J=%.2f, n=10^{%i}, N=%i$"%(J,np.log10(n),N))
    fig3 = plt.figure()
    fig3.suptitle(r"Prikaz razvoja energije od velikosti matrike $J=%.2f, q=%i$"%(J, q))
    for i in tqdm(range((len(kB_T_list)))):
        jndex=0
        ax2 = fig3.add_subplot(2,2,1+i)
        for j in tqdm(range(len(q_list))):
            ax1 = fig.add_subplot(gs1[index])
            grid_new, grid_start, E1_list, iterations_list=Potts(q,q_list[j],n,J,kB_T_list[i])
            im=ax1.imshow(grid_new, cmap=cm.rainbow)
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            if i==0:
                ax1.set_yticklabels([r"$N=%i$" %q_list[j]])
                ax1.set_yticks([N/2])
            if j==0:
                ax1.set_title(r"$\beta=%.2f$" %kB_T_list[i])
            #if j==len(kB_T_list)-1:
                #divider = make_axes_locatable(ax1)
                #cax = divider.append_axes("right", size="2%", pad=0.05)
                #plt.colorbar(im, cax=cax)   

            plt.axis('on')
            index+=1
            ax2.plot(iterations_list,E1_list, label=r"$N=%i$" %q_list[j], color=plt.cm.rainbow(jndex/(len(q_list))))
            jndex+=1
        ax2.set_title(r"$\beta=%.2f$" %kB_T_list[i])
        if i==0:
            ax2.legend()
        ax2.grid()
        ax2.set_xlabel("N")
        ax2.set_ylabel(r"$\langle E \rangle$")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
    plt.show()

Matrix_plot_Mag="No"
if Matrix_plot_Mag=="Yes":
    import matplotlib.gridspec as gridspec
    fig=plt.figure(figsize = (8,8))
    q_list=[2,3,4]
    #q_list=[2,3,4]
    kB_T_list=[0.1,0.5,1,2]
    gs1 = gridspec.GridSpec(int(len(q_list)), int(len(kB_T_list)))
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
    index=0
    fig.suptitle(r"Prikaz razvoja stanja $J=%.2f, n=10^{%i}, N=%i$"%(J,np.log10(n),N))
    fig3 = plt.figure()
    fig3.suptitle(r"Prikaz razvoja magnetizacije od iteracij $J=%.2f, N=%i$"%(J, N))
    for i in tqdm(range((len(q_list)))):
        jndex=0
        ax2 = fig3.add_subplot(2,2,1+i)
        for j in tqdm(range(len(kB_T_list))):
            ax1 = fig.add_subplot(gs1[index])
            grid_new, grid_start, E1_list, M1_list,iterations_list,iterations_list_M=Potts_mag(q_list[i],N,n,J,kB_T_list[j])
            im=ax1.imshow(grid_new, cmap=cm.rainbow)
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            if j==0:
                ax1.set_yticklabels(["$q=%i$" %q_list[i]])
                ax1.set_yticks([N/2])
            if i==0:
                ax1.set_title(r"$\beta=%.2f$" %kB_T_list[j])
            #if j==len(kB_T_list)-1:
                #divider = make_axes_locatable(ax1)
                #cax = divider.append_axes("right", size="2%", pad=0.05)
                #plt.colorbar(im, cax=cax)   

            plt.axis('on')
            index+=1
            ax2.plot(iterations_list_M,M1_list, label=r"$\beta=%.2f$" %kB_T_list[j], color=plt.cm.rainbow(jndex/(len(q_list))))
            jndex+=1
        ax2.set_title("$q=%i$" %q_list[i])
        ax2.grid()
        ax2.set_xlabel("N")
        ax2.set_ylabel(r"$\langle E \rangle$")
        ax2.set_xscale("log")
        ax2.legend()
    plt.show()

#plt.title("Prikaz grafa kritične točke od izbire $q$")
#plt.plot(np.arange(1,101),np.array((math.log(np.arange(1,101)**(1/2)+1)/(J))))
#plt.xlabel("q")
#plt.ylabel(r"$\beta_c$")
#plt.grid()
#plt.show
#
#kB_T_list=np.linspace(0.1,5,100)
##q_list=[2,3,4,5,6]
#q_list=[2,3,4]
#Energy_kt_list=[]
#Magnetization_kt_list=[]
#index=0
#N_list=[10,50,100,200]
#n=10**5
#fig = plt.figure()
#fig.suptitle(r"Prikaz odvisnosti količin od $\beta$, $J=%.2f, n=10^{%i}$ in $q=2$"%(J,np.log10(n)))
#ax1 = fig.add_subplot(211)
#ax2 = fig.add_subplot(212)
#q=2
#fig1 = plt.figure()
#fig1.suptitle(r"Prikaz odvisnosti količin od $\beta$, $J=%.2f, n=10^{%i}$ in $q=2$"%(J,np.log10(n)))
#ax11 = fig1.add_subplot(211)
#ax21 = fig1.add_subplot(212)
#for N in N_list:
#    Energy_kt_list=[]
#    Magnetization_kt_list=[]
#    chi_list=[]
#    c_kt_list=[]
#    for k in tqdm(kB_T_list):
#        grid_new, grid_start, E1_list, iterations_list=Potts(q,N,n,J,k)
#        E=energy(grid_new,J,N)
#        Energy_kt_list.append(E/N**2)
#        M=magnetization(grid_new,J,q,N).real
#        chi=abs(magnetization(grid_new**2,J,q,N) - magnetization(grid_new,J,q,N)**2)*k/N
#        chi_list.append(chi)
#        E_matrix=energy_c(N,J,grid_new)
#        c=abs(np.average(E_matrix**2) - np.average(E_matrix)**2)*k**2
#        c_kt_list.append(c)
#        Magnetization_kt_list.append(abs(M))
#    ax1.axvline(x=(math.log(q**(1/2)+1)/(J)), color=plt.cm.rainbow(index/(len(q_list))))
#    ax1.scatter(np.array(kB_T_list),Energy_kt_list/max(np.array(Energy_kt_list)), color=plt.cm.rainbow(index/(len(q_list))), label="N=%i" %N)
#    ax2.axvline(x=math.log(q**(1/2)+1)/(J), color=plt.cm.rainbow(index/(len(q_list))), label=r'Fazni prehod, $\beta=\frac{\log(1+%i^{1/2})}{J}$' %q)
#    ax2.scatter(np.array(kB_T_list),Magnetization_kt_list/max(np.array(Magnetization_kt_list)), color=plt.cm.rainbow(index/(len(q_list))))
#    ax11.axvline(x=(math.log(q**(1/2)+1)/(J)), color=plt.cm.rainbow(index/(len(q_list))))
#    ax11.scatter(np.array(kB_T_list),chi_list/max(np.array(chi_list)), color=plt.cm.rainbow(index/(len(q_list))), label="N=%i" %N)
#    ax21.axvline(x=math.log(q**(1/2)+1)/(J), color=plt.cm.rainbow(index/(len(q_list))), label=r'Fazni prehod, $\beta=\frac{\log(1+%i^{1/2})}{J}$' %q)
#    ax21.scatter(np.array(kB_T_list),c_kt_list/max(np.array(c_kt_list)), color=plt.cm.rainbow(index/(len(q_list))))
#    index+=1
#ax1.grid()
##ax1.set_xlabel("$k_BT$")
#ax1.set_ylabel(r"$\langle E \rangle_{rel}$")
##ines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
##ines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#ax1.set_xticks([])
## Finally, the legend (that maybe you'll customize differently)
##ig.legend(lines, labels, loc='center', ncol=len(q_list))
#ax1.legend()
#ax2.grid()
#ax2.set_xlabel(r"$\beta$")
#ax2.set_ylabel(r"$\langle M \rangle_{rel}$")
#
#
#ax11.grid()
##ax1.set_xlabel("$k_BT$")
#ax11.set_ylabel(r"$\chi_{rel}$")
##lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
##lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#ax11.set_xticks([])
## Finally, the legend (that maybe you'll customize differently)
##fig1.legend(lines, labels, loc='center', ncol=len(q_list))
#ax11.legend()
#ax21.grid()
#ax21.legend()
#ax21.set_xlabel(r"$\beta$")
#ax21.set_ylabel(r"$c_{v,rel}$")
#plt.show()

kB_T_list=np.linspace(0.1,5,100)
#q_list=[2,3,4,5,6]
q_list=[2,3,4]
Energy_kt_list=[]
Magnetization_kt_list=[]
index=0

fig = plt.figure()
fig.suptitle(r"Prikaz odvisnosti količin od $\beta$, $J=%.2f, n=10^{%i}$"%(J,np.log10(n)))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

fig1 = plt.figure()
fig1.suptitle(r"Prikaz odvisnosti količin od $\beta$, $J=%.2f, n=10^{%i}$"%(J,np.log10(n)))
ax11 = fig1.add_subplot(211)
ax21 = fig1.add_subplot(212)
for q in q_list:
    Energy_kt_list=[]
    Magnetization_kt_list=[]
    chi_list=[]
    c_kt_list=[]
    for k in tqdm(kB_T_list):
        grid_new, grid_start, E1_list, iterations_list=Potts(q,N,n,J,k)
        E=energy(grid_new,J,N)
        Energy_kt_list.append(E/N**2)
        M=magnetization(grid_new,J,q,N).real
        chi=abs(magnetization(grid_new**2,J,q,N) - magnetization(grid_new,J,q,N)**2)*k/N
        chi_list.append(chi)
        E_matrix=energy_c(N,J,grid_new)
        c=abs(np.average(E_matrix**2) - np.average(E_matrix)**2)*k**2
        c_kt_list.append(c)
        print(M)
        Magnetization_kt_list.append(abs(M))
    #ax1.axvline(x=(math.log(q**(1/2)+1)/(J)), color=plt.cm.rainbow(index/(len(q_list))), label=r'Fazni prehod, $\beta=\frac{\log(1+%i^{1/2})}{2*J}$' %q)
    ax1.scatter(np.array(kB_T_list),Energy_kt_list, color=plt.cm.rainbow(index/(len(q_list))), label="q=%i" %q)
    #ax2.axvline(x=math.log(q**(1/2)+1)/(2*J), color=plt.cm.rainbow(index/(len(q_list))), label=r'Fazni prehod, $\beta=\frac{\log(1+%i^{1/2})}{2*J}$' %q)
    ax2.scatter(np.array(kB_T_list),Magnetization_kt_list, color=plt.cm.rainbow(index/(len(q_list))))
    #ax11.axvline(x=(math.log(q**(1/2)+1)/(J)), color=plt.cm.rainbow(index/(len(q_list))), label=r'Fazni prehod, $\beta=\frac{\log(1+%i^{1/2})}{2*J}$' %q)
    ax11.scatter(np.array(kB_T_list),chi_list, color=plt.cm.rainbow(index/(len(q_list))), label="q=%i" %q)
    #ax2.axvline(x=math.log(q**(1/2)+1)/(2*J), color=plt.cm.rainbow(index/(len(q_list))), label=r'Fazni prehod, $\beta=\frac{\log(1+%i^{1/2})}{2*J}$' %q)
    ax21.scatter(np.array(kB_T_list),c_kt_list, color=plt.cm.rainbow(index/(len(q_list))))
    index+=1
ax1.grid()
#ax1.set_xlabel("$k_BT$")
ax1.set_ylabel(r"$\langle E \rangle$")
#ines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
#ines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax1.set_xticks([])
# Finally, the legend (that maybe you'll customize differently)
#ig.legend(lines, labels, loc='center', ncol=len(q_list))
ax1.legend()
ax2.grid()
ax2.set_xlabel(r"$\beta$")
ax2.set_ylabel(r"$\langle M \rangle$")


ax11.grid()
#ax1.set_xlabel("$k_BT$")
ax11.set_ylabel(r"$\chi$")
#lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
#lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax11.set_xticks([])
# Finally, the legend (that maybe you'll customize differently)
#fig1.legend(lines, labels, loc='center', ncol=len(q_list))
ax11.legend()
ax21.grid()
ax21.set_xlabel(r"$\beta$")
ax21.set_ylabel(r"$c_v$")
plt.show()


kB_T_list=np.linspace(0.1,2,70)
q_list=[2,3,4,5,6]
Energy_kt_list=[]
Magnetization_kt_list=[]
index=0
for q in q_list:
    Energy_kt_list=[]
    Magnetization_kt_list=[]
    for k in tqdm(kB_T_list):
        grid_new, grid_start, E1_list, iterations_list=Potts(q,N,n,J,k)
        E=energy(grid_new,J,N)
        Energy_kt_list.append(E/N**2)
        M=magnetization(grid_new,J,q,N).real
        Magnetization_kt_list.append(abs(M))
    plt.title(r"Prikaz odvisnosti povprečne energije od $\beta$, $J=%.2f, k=%.2f, q=%i, n=10^{%i}$"%(J,k,q,np.log10(n)))
    plt.axvline(x=math.log(q**(1/2)+1)/(2*J), color=plt.cm.rainbow(index/(len(q_list))), label=r'Fazni prehod, $\beta=\frac{\log(1+%i^{1/2})}{2*J}$' %q)
    plt.scatter(1/kB_T_list,Energy_kt_list, color=plt.cm.rainbow(index/(len(q_list))), label="q=%i" %q)
    index+=1
plt.legend()
plt.grid()
plt.xlabel(r"$\beta$")
plt.ylabel(r"$\langle E \rangle$")
plt.show()





