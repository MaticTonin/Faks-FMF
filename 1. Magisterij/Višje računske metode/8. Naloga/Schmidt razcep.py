import numpy as np
from numpy.linalg import eigh, norm, svd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from func import *
from numba import jit,njit

# OD KJE IZHAJA MATRIKA https://arxiv.org/pdf/hep-th/9810032.pdf
n_list=[4,6,8,10]
index=0
example=0
n=6

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,1,2)
entropy_period=[]
entropy_open=[]
example=1

from tqdm import tqdm
for n in tqdm(n_list):
    if example==1:
        fig.suptitle(r"Prikaz lastnih vrednosti po Smidtovem razcepu; razpolovljen prostor")
        ax3.set_title(r"Prikaz entropije po Smidtovem razcepu; razpolovljen prostor")
    if example==0:
        fig.suptitle(r"Prikaz lastnih vrednosti po Smidtovem razcepu; simetričen prostor")
        ax3.set_title(r"Prikaz entropije po Smidtovem razcepu; simetričen prostor")
    schmidt=schmidt_decomposition(n, False,example)
    ax1.set_title("Odprt robni pogoj")
    ax1.plot(schmidt, marker="o", label="$n=%i$" %n, color=plt.cm.nipy_spectral(index/len(n_list)))
    entropy_open.append(entropy(schmidt))
    schmidt=schmidt_decomposition(n, True,example)
    ax2.set_title("Periodičen robni pogoj")
    ax2.plot(schmidt, marker="o", label="$n=%i$" %n, color=plt.cm.nipy_spectral(index/len(n_list)))
    entropy_period.append(entropy(schmidt))
    index+=1

if example==1:
    ax3.plot(n_list, entropy_open, "-x",label=r"Open",color="Red")
    ax3.plot(n_list,entropy_period, "-x",label=r"Period",color="Blue")
if example==0:
    popt, _ = curve_fit(line, n_list, entropy_period)
    a_p, b_p = popt
    y_line_p = line(np.array(n_list), a_p, b_p)
    popt, _ = curve_fit(line, n_list, entropy_open)
    a_o, b_o = popt
    y_line_o = line(np.array(n_list), a_o, b_o)
    ax3.plot(n_list, entropy_open, "-x",label=r"Open, $y=%.2f \cdot n + %.2f$" %(a_o,b_o),color="Red")
    ax3.plot(n_list,entropy_period, "-x",label=r"Period $y=%.2f \cdot n + %.2f$" %(a_p,b_p),color="Blue")
    ax3.plot(n_list, y_line_p, '--', color='black')
    ax3.plot(n_list, y_line_o, '--', color='black')
ax1.set_ylabel(r"$\lambda_i$")
ax1.set_xlabel("i")
ax2.set_ylabel(r"$\lambda_i$")
ax2.set_xlabel("i")
ax2.grid()
ax1.grid()
ax1.legend()
ax2.legend()

ax3.grid()
ax3.set_xlabel("n")
ax3.set_ylabel("S(n)")
ax3.legend()
plt.show()


fig =plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
fig.suptitle(r"Prikaz odvisnosti izračuna S(n) od izbire preloma prostora", fontsize=18)

fig1 =plt.figure()
ax11 = fig1.add_subplot(1,2,1)
ax21 = fig1.add_subplot(1,2,2)
fig1.suptitle(r"Prikaz odvisnosti S(n) od izbire preloma prostora, odprt pogoj", fontsize=18)
jndex=0
n_list=[2,4,6,8,10]
for n in tqdm(n_list):
    entr_svd=[]
    entr_mpa=[]
    schmidt_mpa=[]
    schmidt_svd=[]
    state = np.random.normal(0., 1., 2**n) + 1.j * np.random.normal(0., 1., 2**n)
    state = state / norm(state)
    h=create_hamitolian(n,True)
    state = eigh(h)[1][:,0]
    space = np.array([i for i in range(n)])
    for i in range(1, n):
        space_A = space[:i]
        space_B = space[i:]
        psi = make_psi_matrix(state, space_A, space_B)
        schmidt = svd(psi, full_matrices=False, compute_uv=False)
        schmidt_svd.append(schmidt)
        entr_svd.append(entropy(schmidt))
    index=0
    ax1.set_title("Periodičen")
    entr_svd = [0]+entr_svd+[0]
    ax1.plot(np.linspace(0, 1, num=len(entr_svd)),entr_svd, label="n=%i" %n, color=plt.cm.nipy_spectral(jndex/len(n_list)))
    ax1.legend()
    ax1.set_xlabel("j/N", fontsize=18)
    ax1.set_ylabel(r"$S(n)$", fontsize=18)

    entr_svd=[]
    entr_mpa=[]
    schmidt_mpa=[]
    schmidt_svd=[]
    h=create_hamitolian(n,False)
    state = eigh(h)[1][:,0]
    space = np.array([i for i in range(n)])
    for i in range(1, n):
        space_A = space[:i]
        space_B = space[i:]
        psi = make_psi_matrix(state, space_A, space_B)
        schmidt = svd(psi, full_matrices=False, compute_uv=False)
        schmidt_svd.append(schmidt)
        entr_svd.append(entropy(schmidt))
    index=0
    ax2.set_title("Odprt")
    entr_svd = [0]+entr_svd+[0]
    ax2.plot(np.linspace(0, 1, num=len(entr_svd)),entr_svd, label="n=%i" %n, color=plt.cm.nipy_spectral(jndex/len(n_list)))
    ax2.legend()
    ax2.set_xlabel("j/N", fontsize=18)
    ax2.set_ylabel(r"$S(n)$", fontsize=18)
    ax11.set_title("Lihe vrednosti j", fontsize=18)
    ax11.plot(np.linspace(0, 1, num=len(entr_svd[1::2])),entr_svd[1::2], label="n=%i" %n, color=plt.cm.nipy_spectral(jndex/len(n_list)))
    ax11.legend()
    ax11.set_xlabel("j/N", fontsize=18)
    ax11.set_ylabel(r"$S(n)$", fontsize=18)
    ax21.set_title("Sode vrednosti j", fontsize=18)
    ax21.plot(np.linspace(0, 1, num=len(entr_svd[::2])),entr_svd[::2], label="n=%i" %n, color=plt.cm.nipy_spectral(jndex/len(n_list)))
    ax21.legend()
    ax21.set_xlabel("j/N", fontsize=18)
    ax21.set_ylabel(r"$S(n)$", fontsize=18)
    jndex+=1
ax1.grid()
ax2.grid()
ax11.grid()
ax21.grid()
plt.show()




entr_svd=[]
entr_mpa=[]
schmidt_mpa=[]
schmidt_svd=[]
fig =plt.figure()
fig.suptitle("Prikaz lastnih vrednosti v odvisnosti od metode", fontsize=18)

fig2 =plt.figure()
fig2.suptitle("Prikaz entropije v odvisnosti od metode", fontsize=18)
ax1=fig2.add_subplot(2,2,1)
ax2=fig2.add_subplot(2,2,2)
ax3=fig2.add_subplot(2,1,2)

fig3 =plt.figure()
fig3.suptitle("Prikaz maksimuma entropije od n", fontsize=18)
ax4=fig3.add_subplot(1,1,1)
pl_index=0
n_list=[2,4,6,8,10,12,14,16,18]
jndex=0
middle_mpa=[]
for n in tqdm(n_list):
    entr_svd=[]
    entr_mpa=[]
    schmidt_mpa=[]
    schmidt_svd=[]
    ax = fig.add_subplot(len(n_list),3,pl_index+1)
    if pl_index==0:
        ax.set_title("Metoda MPA", fontsize=18)
    #h=create_hamitolian(n,True)
    #state = eigh(h)[1][:,0]
    mpa = MPA(state)
    schmidt_mpa=mpa[0]
    index=0
    for s in schmidt_mpa:
        entr_mpa.append(entropy(s))
        ax.plot(s, "o-",color=plt.cm.nipy_spectral(index/len(schmidt_mpa)))
        index+=1
    ax.plot([],[],label="n=%i" %n)
    ax.set_xlabel("k", fontsize=18)
    ax.set_ylabel(r"$\lambda_k$", fontsize=18)
    ax.grid()
    ax.legend()
    pl_index+=1
    space = np.array([i for i in range(n)])
    ax = fig.add_subplot(len(n_list),3,pl_index+1)
    if pl_index==1:
        ax.set_title("Metoda SVD", fontsize=18)
    for i in range(1, n):
        space_A = space[:i]
        space_B = space[i:]
        psi = make_psi_matrix(state, space_A, space_B)
        schmidt = svd(psi, full_matrices=False, compute_uv=False)
        schmidt_svd.append(schmidt)
        entr_svd.append(entropy(schmidt))
    index=0
    for s in schmidt_svd:
        ax.plot(s, "o-",label="j=%i" %index,color=plt.cm.nipy_spectral(index/len(schmidt_mpa)))
        index+=1
    ax.set_xlabel("k", fontsize=18)
    ax.set_ylabel(r"$\lambda_k$", fontsize=18)
    ax.grid()
    pl_index+=1
    ax = fig.add_subplot(len(n_list),3,pl_index+1)
    if pl_index==2:
        ax.set_title("Razlika med njima", fontsize=18)
    index=0
    for i in range(len(schmidt_svd)):
        ax.plot(np.divide(np.absolute(schmidt_svd[i]-schmidt_mpa[i]),schmidt_svd[i]), "o-", color=plt.cm.nipy_spectral(index/len(schmidt_mpa)))
        ax.plot([],[],label="j=%i" %index, color=plt.cm.nipy_spectral(index/len(schmidt_mpa)))
        index+=1
    ax.set_xlabel("k", fontsize=18)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\frac{|\lambda_{SVD}-\lambda_{MPA}|}{\lambda_{SVD}}$", fontsize=18)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    entr_svd = [0]+entr_svd+[0]
    entr_mpa = [0]+entr_mpa+[0]
    middle_mpa.append(entr_mpa[len(entr_mpa)//2])
    ax1.set_title("Metoda SVD", fontsize=18)
    ax1.plot(np.linspace(0, 1, num=len(entr_svd)),entr_svd, label="n=%i" %n, color=plt.cm.nipy_spectral(jndex/len(n_list)))
    ax1.legend()
    ax1.set_xlabel("j/N", fontsize=18)
    ax1.set_ylabel("S(N)", fontsize=18)
    ax2.set_title("Metoda MPA", fontsize=18)
    ax2.plot(np.linspace(0, 1, num=len(entr_mpa)),entr_mpa, color=plt.cm.nipy_spectral(jndex/len(n_list)))
    ax2.set_xlabel("j/N", fontsize=18)
    ax2.set_ylabel("S(N)", fontsize=18)
    ax3.set_title("Razlika med njima", fontsize=18)
    ax3.plot(np.linspace(0, 1, num=len(entr_mpa)),abs(np.array(entr_mpa)-np.array(entr_svd))/np.array(entr_mpa), color=plt.cm.nipy_spectral(jndex/len(n_list)))
    ax3.set_xlabel("j/N", fontsize=18)
    ax3.set_ylabel(r"$\Delta S(N)$", fontsize=18)
    ax3.set_yscale("log")
    pl_index+=1
    jndex+=1


popt, _ = curve_fit(x_2, n_list, middle_mpa)
a_o, b_o, c_o = popt
y_line_o = x_2(np.array(n_list), a_o, b_o, c_o)
ax4.plot(n_list,middle_mpa,"-x", color="red")
ax4.plot(n_list,y_line_o, "--", label=r"Fit $%.5f x^2+ %.2f x+ %.2f$" %(a_o,b_o,c_o), color="black")
popt, _ = curve_fit(x_2_x, n_list, middle_mpa)
a_o, c_o = popt
y_line_o = x_2_x(np.array(n_list), a_o, c_o)
ax4.plot(n_list,y_line_o, "--", label=r"Fit $%.5f x^2+ %.2f$" %(a_o,c_o), color="blue")
popt, _ = curve_fit(x_2_x_c, n_list, middle_mpa)
a_o = popt
y_line_o = x_2_x_c(np.array(n_list), a_o)
ax4.plot(n_list,y_line_o, "--", label=r"Fit $%.5f x^2$" %(a_o), color="green")
ax4.set_xlabel("n", fontsize=18)
ax4.set_ylabel(r"$S(N)_{max}$", fontsize=18)
ax4.legend()
ax4.grid()
ax3.grid()
ax2.grid()
ax1.grid()
plt.show()

