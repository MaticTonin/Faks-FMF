from multiprocessing import Pool
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.random import rand
from numba import jit,njit
import scipy
from scipy.stats import norm
from tqdm import tqdm
from func import Evaluation, acceptance_rate
reps = int(1e+2)
steps=int(1e+5)
m_ar = [200, 100, 50, 25, 10]
M=100
init_state = np.random.normal(size=M)
beta = 1
lam=0
eps=1
steps=10000
eps_line=np.logspace(-4,0,500)
beta_list=[0.5,1,5,10]
if __name__ == '__main__':
    result_objs = []
    initai_state=[]
    E=[]
    H=[]
    V=[]
    index=0
    acceptance_beta_plot="No"
    if acceptance_beta_plot=="Yes":
        fig1, ax1 = plt.subplots()
        fig1.suptitle(r"Število sprejetih potez od $\varepsilon$")

        fig2 = plt.figure() # create the canvas for plotting
        fig2.suptitle(r"Število sprejetih potez od korakov")
        for beta in beta_list:
            acc_line=[]
            acc_avg_line=[]
            ax2 = fig2.add_subplot(2,2,index+1)
            ax2.set_title(r"$\beta=%.2f$" %(beta))
            for eps in tqdm(eps_line):
                result = acceptance_rate(init_state,M,beta,eps,lam, steps)
                acc,acc_avg=result
                acc_line.append(acc)
                acc_avg_line.append(acc_avg)
            acc_avg_line2=np.array(acc_avg_line)
            acc_avg_line=np.mean(np.array(acc_avg_line).T, axis=0)
            ax1.plot(eps_line,acc_avg_line, label=r"$\beta=%.2f$" %(beta), color=plt.cm.coolwarm(index/len(beta_list)))
            for j in range(len(acc_avg_line)):
                if eps_line[j]%0.01<0.0001:
                    ax2.plot(np.linspace(0,len(acc_avg_line2[j]),len(acc_avg_line2[j])),acc_avg_line2[j], label=r"$\varepsilon=%.5f$" %(eps_line[j]), color=plt.cm.coolwarm(j/len(acc_avg_line)))
            index+=1
            ax2.legend()
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.grid()
            ax2.set_xlabel(r"$N$")
            ax2.set_ylabel(r"Št sprejetih")
        ax1.axhline(0.5, color='black', linestyle='--', label="Sprejetih=0.5")
        ax1.legend()
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.grid()
        ax1.set_xlabel(r"$\varepsilon$")
        ax1.set_ylabel(r"Št sprejetih")
        plt.show()

    epsilon_plot="No"
    if epsilon_plot=="Yes":
        result_objs = []
        initai_state=[]
        E=[]
        H=[]
        V=[]
        eps_line=[0.001,0.01,0.1,0.5,1]
        index=0

        fig2 = plt.figure() # create the canvas for plotting
        fig2.suptitle(r"Število sprejetih potez od korakov",fontsize="20")
        plt.rcParams.update({'font.size': 18})
        for beta in beta_list:
            acc_line=[]
            acc_avg_line=[]
            ax2 = fig2.add_subplot(2,2,index+1)
            ax2.set_title(r"$\beta=%.2f$" %(beta))
            for eps in tqdm(eps_line):
                result = acceptance_rate(init_state,M,beta,eps,lam, steps)
                acc,acc_avg=result
                acc_line.append(acc)
                acc_avg_line.append(acc_avg)
            acc_avg_line2=np.array(acc_avg_line)
            acc_avg_line=np.mean(np.array(acc_avg_line).T, axis=0)
            for j in range(len(acc_avg_line)):
                ax2.plot(np.linspace(0,len(acc_avg_line2[j]),len(acc_avg_line2[j])),acc_avg_line2[j], label=r"$\varepsilon=%.3f$" %(eps_line[j]), color=plt.cm.coolwarm(j/len(acc_avg_line)))
            ax2.axhline(0.5, color='black', linestyle='--', label="Sprejetih=0.5")
            if index==0:
                ax2.legend(fontsize="16")
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.grid()
            ax2.set_xlabel(r"$N$", fontsize="16")
            ax2.set_ylabel(r"Št sprejetih", fontsize="16")
            index+=1
        plt.show()

#init_state, dE, H_V=Quantum_monte_carlo(init_state,beta,reps,lam, eps,M)
if __name__ == '__main__':
    steps_plot="No"
    if steps_plot=="Yes":
        result_objs = []
        initai_state=[]
        E=[]
        H=[]
        V=[]
        steps_list=[int(1e+2),int(1e+3),int(1e+4)]
        #steps_list=[int(1e+6),int(1e+5),int(1e+4),int(1e+3),int(1e+2)]
        args = [[np.random.normal(size=M),beta,reps,lam, eps, steps,M] for steps in steps_list]
        with Pool(processes=os.cpu_count() - 2) as pool:
            result = pool.starmap(Evaluation, args)
            init_state_finish,E_finish,H_finish,V_finish,Energy_list,H_list,V_list,epsilon=np.array(result).T
        fig1, ax1 = plt.subplots()
        plt.rcParams.update({'font.size': 18})
        fig1.suptitle(r"Prikaz spremembe energije za več ponavljanj istega števila korakov pri $\beta=%.2f$" %beta)
        fig2 = plt.figure()
        plt.rcParams.update({'font.size': 18})
        fig2.suptitle(r"Prikaz spremembe energije za več ponavljanj istega števila korakov pri $\beta=%.2f$" %beta)
        index=0
        for E in E_finish:
            ax1.plot(np.linspace(0,reps+1,reps+1),E, label=r"$n=%i$" %(steps_list[index]), color=plt.cm.coolwarm(index/len(steps_list)))
            index+=1
        index=0
        ax2 = fig2.add_subplot(4,1,1)
        ax3 = fig2.add_subplot(4,1,2)
        ax4 = fig2.add_subplot(4,1,3)
        ax5 = fig2.add_subplot(4,1,4)
        for i in range(len(E_finish)):
            ax2.plot(np.linspace(0,(len(epsilon[i]))*steps_list[i]/100,len(epsilon[i])),Energy_list[i], label=r"$n=%i$" %(steps_list[index]), color=plt.cm.coolwarm(index/len(steps_list)))
            ax2.axvline((len(epsilon[i]))*steps_list[i]/100, linestyle='--', color=plt.cm.coolwarm(index/len(steps_list)))
            ax3.plot(np.linspace(0,(len(epsilon[i]))*steps_list[i]/100,len(epsilon[i])),H_list[i], label=r"$n=%i$" %(steps_list[index]), color=plt.cm.coolwarm(index/len(steps_list)))
            ax3.axvline((len(epsilon[i]))*steps_list[i]/100, linestyle='--', color=plt.cm.coolwarm(index/len(steps_list)))
            ax4.plot(np.linspace(0,(len(epsilon[i]))*steps_list[i]/100,len(epsilon[i])),V_list[i], label=r"$n=%i$" %(steps_list[index]), color=plt.cm.coolwarm(index/len(steps_list)))
            ax4.axvline((len(epsilon[i]))*steps_list[i]/100, linestyle='--', color=plt.cm.coolwarm(index/len(steps_list)))
            ax5.plot(np.linspace(0,(len(epsilon[i]))*steps_list[i]/100,len(epsilon[i])),epsilon[i], label=r"$n=%i$" %(steps_list[index]), color=plt.cm.coolwarm(index/len(steps_list)))
            ax5.axvline((len(epsilon[i]))*steps_list[i]/100, linestyle='--', color=plt.cm.coolwarm(index/len(steps_list)))
            index+=1
        ax1.legend()
        ax2.legend()
        ax2.set_xscale("log")
        ax2.set_xlabel(r"$N$", fontsize="16")
        ax2.set_ylabel(r"$\langle E \rangle$", fontsize="16")
        ax2.grid()
        ax3.set_xscale("log")
        ax3.set_xlabel(r"$N$", fontsize="16")
        ax3.set_ylabel(r"$\langle H \rangle$", fontsize="16")
        ax3.grid()
        ax4.set_xscale("log")
        ax4.set_xlabel(r"$N$", fontsize="16")
        ax4.set_ylabel(r"$\langle V \rangle$", fontsize="16")
        ax4.grid()
        ax5.set_xscale("log")
        ax5.set_xlabel(r"$N$", fontsize="16")
        ax5.set_ylabel(r"$\varepsilon$", fontsize="16")
        ax5.grid()
        #ax1.set_xscale("log")
        ax1.grid()
        ax1.set_xlabel(r"$reps$")
        ax1.set_ylabel(r"$\langle E \rangle$")
        plt.show()
    beta_plot="Yes"
    if beta_plot=="Yes":
        lam=0
        steps=int(1e+6)
        reps = int(1e+1)
        beta_list=np.logspace(-1,2,10)
        M_list=[25]
        fig1 =plt.figure()
        fig1.suptitle(r"Prikaz povprečnih količin v odvisnosti od $\beta$ za $n=10^{%i}, \lambda=%i$" %(np.log10(steps),lam))
        ax1=fig1.add_subplot(3,1,1)
        ax2=fig1.add_subplot(3,1,2)
        ax3=fig1.add_subplot(3,1,3)
        for j in range(len(M_list)):
            M=M_list[j]
            E_mean=[]
            E_std=[]
            H_mean=[]
            H_std=[]
            V_mean=[]
            V_std=[]
            #args = [[np.random.normal(size=M),beta,reps,lam, eps, steps,M] for beta in beta_list]
            #with Pool(processes=os.cpu_count() - 10) as pool:
            #    for result in pool.starmap(Evaluation, args):
            #       init_state_finish,E_finish,H_finish,V_finish,Energy_list,H_list,V_list,epsilon=np.array(result)
            for i in tqdm(range(len(beta_list))):
                init_state_finish,E_finish,H_finish,V_finish,Energy_list,H_list,V_list,epsilon=Evaluation(np.random.normal(size=M),beta_list[i],reps,lam, eps, steps,M)
                E_mean.append(np.mean(np.array(E_finish)[int(reps/2):reps]))
                E_std.append(np.std(np.array(E_finish)[int(reps/2):reps]))
                H_mean.append(np.mean(np.array(H_finish)[int(reps/2):reps]))
                H_std.append(np.std(np.array(H_finish)[int(reps/2):reps]))
                V_mean.append(np.mean(np.array(V_finish)[int(reps/2):reps]))
                V_std.append(np.std(np.array(V_finish)[int(reps/2):reps]))

                #E_mean.append(np.mean(np.array(E_finish[i])[int(reps/2):reps]))
                #E_std.append(np.std(np.array(E_finish[i])[int(reps/2):reps]))
                #H_mean.append(np.mean(np.array(H_finish[i])[int(reps/2):reps]))
                #H_std.append(np.std(np.array(H_finish[i])[int(reps/2):reps]))
                #V_mean.append(np.mean(np.array(V_finish[i])[int(reps/2):reps]))
                #V_std.append(np.std(np.array(V_finish[i])[int(reps/2):reps]))
            ax1.plot(beta_list,E_mean, color=plt.cm.coolwarm(j/len(M_list)), label="$M=%i$" %M)
            ax1.axvline()
            ax2.plot(beta_list,H_mean, color=plt.cm.coolwarm(j/len(M_list)), label="$M=%i$" %M)
            ax3.plot(beta_list,V_mean, color=plt.cm.coolwarm(j/len(M_list)), label="$M=%i$" %M)
        ax1.set_xlabel(r"$\beta$")
        ax1.set_ylabel(r"$\langle E \rangle$")
        ax2.set_xlabel(r"$\beta$")
        ax2.set_ylabel(r"$\langle H \rangle$")
        ax3.set_xlabel(r"$\beta$")
        ax3.set_ylabel(r"$\langle V \rangle$")
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        ax3.set_xscale("log")
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.legend()
        plt.show()



