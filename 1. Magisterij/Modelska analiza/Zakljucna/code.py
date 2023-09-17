from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import winsound
import itertools
import math
import itertools
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.linalg as la
import spectrum as spc
from spectrum import *
import pandas as pd
import math
from math import floor
def distance(p1, p2):
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2)))

def total_distance(path, vertices):
    total = 0
    for i in range(len(path) - 1):
        total += distance(vertices[path[i]], vertices[path[i + 1]])
    total += distance(vertices[path[-1]], vertices[path[0]])  # Closing the loop
    return total

def thompson_bruteforce(vertices):
    num_vertices = len(vertices)
    min_distance = float('inf')
    min_path = []

    for path in itertools.permutations(range(num_vertices)):
        path_distance = total_distance(path, vertices)
        if path_distance < min_distance:
            min_distance = path_distance
            min_path = path

    return min_path, min_distance

@jit(nopython=True)
def weird_division(n, d):
    return n / d if d else 0

@jit(nopython=True)
def r_gen(phi, theta):
    sphere_vector = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    if phi > 7 * np.pi / 4 or phi < np.pi / 4:
        if theta < np.pi / 2:
            t1 = weird_division(1, sphere_vector[0])
            t2 = weird_division(1, sphere_vector[2])
            t = np.min(np.array([t1, t2]))
            r = t * sphere_vector
        if theta > np.pi / 2:
            t1 = weird_division(1, sphere_vector[0])
            t2 = weird_division(-1, sphere_vector[2])
            t = np.min(np.array([t1, t2]))
            r = t * sphere_vector

    if np.pi / 4 < phi < 3 * np.pi / 4:
        if theta < np.pi / 2:
            t1 = weird_division(1, sphere_vector[1])
            t2 = weird_division(1, sphere_vector[2])
            t = np.min(np.array([t1, t2]))
            r = t * sphere_vector
        if theta > np.pi / 2:
            t1 = weird_division(1, sphere_vector[1])
            t2 = weird_division(-1, sphere_vector[2])
            t = np.min(np.array([t1, t2]))
            r = t * sphere_vector

    if 3 * np.pi / 4 < phi < 5 * np.pi / 4:
        if theta < np.pi / 2:
            t1 = weird_division(-1, sphere_vector[0])
            t2 = weird_division(1, sphere_vector[2])
            t = np.min(np.array([t1, t2]))
            r = t * sphere_vector
        if theta > np.pi / 2:
            t1 = weird_division(-1, sphere_vector[0])
            t2 = weird_division(-1, sphere_vector[2])
            t = np.min(np.array([t1, t2]))
            r = t * sphere_vector

    if 5 * np.pi / 4 < phi < 7 * np.pi / 4:
        if theta < np.pi / 2:
            t1 = weird_division(-1, sphere_vector[1])
            t2 = weird_division(1, sphere_vector[2])
            t = np.min(np.array([t1, t2]))
            r = t * sphere_vector
        if theta > np.pi / 2:
            t1 = weird_division(-1, sphere_vector[1])
            t2 = weird_division(-1, sphere_vector[2])
            t = np.min(np.array([t1, t2]))
            r = t * sphere_vector
    return r


@jit(nopython=True)
def energy(vectors):
    N = int(len(vectors) / 2)
    energy = 0
    for i in range(N):
        for j in range(N):
            if j > i:
                distance = np.linalg.norm(r_gen(vectors[2 * i], vectors[2 * i + 1]) - r_gen(vectors[2 * j], vectors[2 * j + 1]))
                energy += weird_division(1, distance)
    return energy


@jit(nopython=True)
def initial_conditions(N):
    vector0 = []
    for i in range(N):
        vector0.append(2 * np.pi * np.random.rand())
        vector0.append(np.pi * np.random.rand())
    return np.array(vector0)


# a = 1
# print(r_gen(1.481855793623531, 0.030466666775523787))
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
N_list =[1,2,3,4,5]
#N_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
#N_list = np.linspace(10,160,30)
index = 0
fig = plt.figure(figsize=(9,8))
fig10 = plt.figure(figsize=(9,8))
fig2 = plt.figure(figsize=(9,8))
ax1 = fig2.add_subplot(1, 1, 1)
fig3 = plt.figure(figsize=(9,8))
ax3 = fig3.add_subplot(1, 1, 1)
fig4 = plt.figure(figsize=(9,8))
ax4 = fig4.add_subplot(1, 1, 1)
fig5 = plt.figure(figsize=(9,8))
ax5 = fig5.add_subplot(1, 1, 1)
saved_energy = []
N=1
fig.suptitle(f"Prikaz nabojev od {1}-{4}")
fig10.suptitle(f"Prikaz nabojev na matriki od {1}-{4}")
color_index = 0
time_taken = []
box_size = 50
import time

iterations = 5 
index = 0
N_list = [600]
for N in N_list:
    sum_energy = 0
    phi_sol_array = []
    theta_sol_array = []
    time_array = []
    if index % 4 == 0 and index !=0:
        fig10.savefig(THIS_FOLDER +f"\Slike\Test {N}-{N+index-1}.png")
        fig.savefig(THIS_FOLDER +f"\Slike\Square {N}-{N+index-1}.png")
        plt.close(fig)
        plt.close(fig10)
        fig = plt.figure(figsize=(9,8))
        fig.suptitle(f"Prikaz nabojev od {N}-{N+index-1}")
        fig10 = plt.figure(figsize=(9,8))
        fig10.suptitle(f"Prikaz nabojev na matriki od {N}-{N+index-1}")
        index = 0
    ax10 = fig10.add_subplot(2, 2, index+1)
    for j in range(iterations):
        print(j)
        t0 = time.time()
        vector0 = initial_conditions(N)
        bnds = [] 
        for i in range(N):
            bnds.append([0, 2 * np.pi])
            bnds.append([0, np.pi])
        solution = minimize(energy, vector0, method='BFGS', bounds=bnds, options={"maxiter": 2 * N * 1000, "maxfev": 2 * N * 1000}).x
        phi_sol = np.array([solution[2 * i] for i in range(N)])
        #phi_sol_array.append(phi_sol)
        theta_sol = np.array([solution[2 * i + 1] for i in range(N)])
        #theta_sol_array.append(theta_sol)
        sum_energy += energy(solution)
        cartesian_sol = np.array([r_gen(phi_sol[i], theta_sol[i]) for i in range(N)])
        #np.savetxt(THIS_FOLDER +"\data_N{}iter{}.txt".format(N, 0), cartesian_sol)
        xdata = cartesian_sol.T[0] / 2 + 1 / 2
        ydata = cartesian_sol.T[1] / 2 + 1 / 2
        zdata = cartesian_sol.T[2] / 2 + 1 / 2
        for i in range(len(xdata)):
            theta_sol_array.append(xdata[i])
            phi_sol_array.append(ydata[i])
        color_index += 1
    #phi_sol_array = np.mean(phi_sol_array,axis=0)
    #theta_sol_array = np.mean(theta_sol_array,axis=0)
    #cartesian_sol = np.array([r_gen(phi_sol_array[i], theta_sol_array[i]) for i in range(N)])
    
    #if N % 4 == 0:
    #    ax3.hist(phi_sol_array, label =f"N = {N}",bins= np.linspace(0,2*np.pi, 30),edgecolor='black', color=plt.cm.coolwarm(color_index/len(N_list)), alpha =0.4)
    #    ax4.hist(theta_sol_array, label =f"N = {N}",bins= np.linspace(0,np.pi, 30),edgecolor='black',color=plt.cm.coolwarm(color_index/len(N_list)), alpha =0.4)
    #else:
    #    ax3.hist(phi_sol_array, edgecolor='black',bins= np.linspace(0,2*np.pi, 30), color=plt.cm.coolwarm(color_index/len(N_list)), alpha =0.4)
    #    ax4.hist(theta_sol_array,edgecolor='black',bins= np.linspace(0,np.pi, 30),color=plt.cm.coolwarm(color_index/len(N_list)), alpha =0.4)
    sum_energy = sum_energy / iterations
    saved_energy.append(sum_energy/N)
    xdata = phi_sol_array
    ydata = theta_sol_array
    #xdata = cartesian_sol.T[0] / 2 + 1 / 2
    #ydata = cartesian_sol.T[1] / 2 + 1 / 2
    #zdata = cartesian_sol.T[2] / 2 + 1 / 2
    fig10.suptitle(f"Prikaz nabojev na matriki od {N}")
    counts, xedges, yedges, im = ax10.hist2d(xdata,ydata, bins = 5, cmap="Greens")
    plt.colorbar(im, ax=ax10)
    ax10 = fig10.add_subplot(2, 2, 2)
    counts, xedges, yedges, im = ax10.hist2d(xdata,ydata, bins = 10, cmap="Greens")
    plt.colorbar(im, ax=ax10)
    ax10 = fig10.add_subplot(2, 2, 3)
    counts, xedges, yedges, im = ax10.hist2d(xdata,ydata, bins = 20, cmap="Greens")
    plt.colorbar(im, ax=ax10)
    ax10 = fig10.add_subplot(2, 2, 4)
    counts, xedges, yedges, im = ax10.hist2d(xdata,ydata, bins = 40, cmap="Greens")
    plt.colorbar(im, ax=ax10)
    plt.show()
    index += 1
fig10.savefig(THIS_FOLDER +f"\Slike\Test.png")
for N in N_list:
    print(N)
    N = int(N)
    if index % 4 == 0 and index !=0:
        fig10.savefig(THIS_FOLDER +f"\Slike\Matrix {N}-{N+index-1}.png")
        fig.savefig(THIS_FOLDER +f"\Slike\Square {N}-{N+index-1}.png")
        plt.close(fig)
        plt.close(fig10)
        fig = plt.figure(figsize=(9,8))
        fig.suptitle(f"Prikaz nabojev od {N}-{N+index-1}")
        fig10 = plt.figure(figsize=(9,8))
        fig10.suptitle(f"Prikaz nabojev na matriki od {N}-{N+index-1}")
        index = 0
    ax = fig.add_subplot(2, 2, index+1, projection='3d')
    ax10 = fig10.add_subplot(2, 2, index+1)
    vector0 = initial_conditions(N)
    bnds = []
    for i in range(N):
        bnds.append([0, 2 * np.pi])
        bnds.append([0, np.pi])
    solution = minimize(energy, vector0, method='SLSQP', bounds=bnds, options={"maxiter": 2 * N * 1000, "maxfev": 2 * N * 1000}).x
    phi_sol = np.array([solution[2 * i] for i in range(N)])
    theta_sol = np.array([solution[2 * i + 1] for i in range(N)])
    cartesian_sol = np.array([r_gen(phi_sol[i], theta_sol[i]) for i in range(N)])

    # plotting
    xdata = cartesian_sol.T[0] / 2 + 1 / 2
    ydata = cartesian_sol.T[1] / 2 + 1 / 2
    zdata = cartesian_sol.T[2] / 2 + 1 / 2
    counts, xedges, yedges, im = ax10.hist2d(xdata,ydata, bins = 50, cmap="coolwarm")
    plt.colorbar(im, ax=ax10)

    axes = [1, 1, 1]
    data = np.ones(axes, dtype=np.bool_)
    alpha = 0.4
    colors = np.empty(axes + [4], dtype=np.float32)
    colors[:] = [1, 1, 1, alpha]  # red

    ax.set_title("N = {}".format(N))
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_zlabel('$z$', fontsize=14)
    #ax.set_xticks([-1, 0, 1])
    #ax.set_yticks([-1, 0, 1])
    #ax.set_zticks([-1, 0, 1])
    #ax.set_box_aspect([1, 1, 1])
    ax.scatter3D(xdata, ydata, zdata, c="blue", alpha = 0.7)
    ax.voxels(data, facecolors=colors)
    index +=1

    iterations = 1
    sum_energy = 0
    phi_sol_array = []
    theta_sol_array = []
    time_array = []
    for j in range(iterations):
        t0 = time.time()
        vector0 = initial_conditions(N)
        bnds = []
        for i in range(N):
            bnds.append([0, 2 * np.pi])
            bnds.append([0, np.pi])
        solution = minimize(energy, vector0, method='BFGS', bounds=bnds, options={"maxiter": 2 * N * 1000, "maxfev": 2 * N * 1000}).x
        phi_sol = np.array([solution[2 * i] for i in range(N)])
        phi_sol_array.append(phi_sol)
        theta_sol = np.array([solution[2 * i + 1] for i in range(N)])
        theta_sol_array.append(theta_sol)
        sum_energy += energy(solution)
        cartesian_sol = np.array([r_gen(phi_sol[i], theta_sol[i]) for i in range(N)])
        #np.savetxt(THIS_FOLDER +"\data_N{}iter{}.txt".format(N, 0), cartesian_sol)
        time_array.append(time.time() - t0)
    time_taken.append(np.mean(time_array))
    phi_sol_array = np.mean(phi_sol_array,axis=0)
    theta_sol_array = np.mean(theta_sol_array,axis=0)
    cartesian_sol = np.array([r_gen(phi_sol_array[i], theta_sol_array[i]) for i in range(N)])
    if N % 4 == 0:
        ax3.hist(phi_sol_array, label =f"N = {N}",bins= np.linspace(0,2*np.pi, 30),edgecolor='black', color=plt.cm.coolwarm(color_index/len(N_list)), alpha =0.4)
        ax4.hist(theta_sol_array, label =f"N = {N}",bins= np.linspace(0,np.pi, 30),edgecolor='black',color=plt.cm.coolwarm(color_index/len(N_list)), alpha =0.4)
    else:
        ax3.hist(phi_sol_array, edgecolor='black',bins= np.linspace(0,2*np.pi, 30), color=plt.cm.coolwarm(color_index/len(N_list)), alpha =0.4)
        ax4.hist(theta_sol_array,edgecolor='black',bins= np.linspace(0,np.pi, 30),color=plt.cm.coolwarm(color_index/len(N_list)), alpha =0.4)
    sum_energy = sum_energy / iterations
    saved_energy.append(sum_energy/N)
    color_index += 1
plt.close(fig)
ax3.grid()
ax3.legend()
ax4.grid()
ax4.legend()
fig3.suptitle(r"Prikaz histograma porazdelitve po kotu $\phi$")
fig4.suptitle(r"Prikaz histograma porazdelitve po kotu $\theta$")
ax3.set_xlabel(r"$\phi$")
ax3.set_ylabel(r"Number")
ax4.set_xlabel(r"$\theta$")
ax4.set_ylabel(r"Number")
fig2.suptitle("Prikaz odvisnoti energije od N")
ax1.grid()
ax1.set_xlabel("N")
ax1.set_ylabel("Energija/N")

#ax1.plot(N_list, saved_energy)
#a, b = np.polyfit(N_list, saved_energy, 1)
#ax1.plot(N_list, a*N_list+b, label = r"%.2f N%.2f" %(a,b))
#ax1.legend()

ax5.set_xlabel(r"N")
ax5.set_ylabel(r"Potreben čas [s]")
ax5.plot(N_list,time_taken)
ax5.grid()
fig5.suptitle("Prikaz potrebnega časa za izdelavo grafov")
plt.show()