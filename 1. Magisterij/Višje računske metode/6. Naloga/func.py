import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit,njit, objmode
import random
import matplotlib.cm as cm
import time
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

def sosed(s1,s2,J):
    if s1==s2:
        return -J
    else: 
          return 0
    

@jit(nopython = True)
def Potts(q,N,iterations,J,k):
    grid=np.zeros((N,N))
    number=0
    for i in range(N):
        for j in range(N):
            number=np.random.randint(1, high=q+1)
            grid[i,j]=number
    grid_start=grid.copy()
    grid_new=np.ones((N,N))
    spr=0

    E1_list=[]
    iterations_list=[]
    E=0
    for i in range(iterations):
        x=random.randint(0,N-1)
        y=random.randint(0,N-1)
        grid_new=grid.copy()
        old_value=grid[x][y]
        grid_new[x][y]=np.random.randint(1, high=q+1)
        dE=0
        if x-1 >=0:
            if grid_new[x][y]==grid_new[x-1][y] and grid[x][y]!=grid[x-1][y]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x-1][y] and grid[x][y]==grid[x-1][y]:
                    dE+=-J*(-1)
                    spr+=1
        if x+1 < N:
            if grid_new[x][y]==grid_new[x+1][y] and grid[x][y]!=grid[x+1][y]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x+1][y] and grid[x][y]==grid[x+1][y]:
                    dE+=-J*(-1)
                    spr+=1
        if y-1 >=0:
            if grid_new[x][y]==grid_new[x][y-1] and grid[x][y]!=grid[x][y-1]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x][y-1] and grid[x][y]==grid[x][y-1]:
                    dE+=-J*(-1)
                    spr+=1
        if y+1 < N:
            if grid_new[x][y]==grid_new[x][y+1] and grid[x][y]!=grid[x][y+1]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x][y+1] and grid[x][y]==grid[x][y+1]:
                    dE+=-J*(-1)
                    spr+=1
        E+=dE       
        if dE > 0:
            zeta = random.random()
            if zeta > np.exp(-dE*k):
                grid_new[x][y] = old_value
                E-=dE
        if i>10**3 and i%1000==0:
            E1_list.append(E/N)
            iterations_list.append(i)

        grid=grid_new.copy()
    return grid_new, grid_start, E1_list, iterations_list


@jit(nopython = True)

def Potts_2(grid,q,N,iterations,J,k):
    grid_start=grid.copy()
    spr=0

    #iterations_list=np.zeros(N)
    #E1_list=np.zeros(N)
    E1_list=[]
    iterations_list=[]
    E=0
    for i in range(iterations):
        x=random.randint(0,N-1)
        y=random.randint(0,N-1)
        grid_new=grid.copy()
        old_value=grid[x][y]
        grid_new[x][y]=np.random.randint(1, high=q+1)
        dE=0
        if x-1 >=0:
            if grid_new[x][y]==grid_new[x-1][y] and grid[x][y]!=grid[x-1][y]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x-1][y] and grid[x][y]==grid[x-1][y]:
                    dE+=-J*(-1)
                    spr+=1
        if x+1 < N:
            if grid_new[x][y]==grid_new[x+1][y] and grid[x][y]!=grid[x+1][y]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x+1][y] and grid[x][y]==grid[x+1][y]:
                    dE+=-J*(-1)
                    spr+=1
        if y-1 >=0:
            if grid_new[x][y]==grid_new[x][y-1] and grid[x][y]!=grid[x][y-1]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x][y-1] and grid[x][y]==grid[x][y-1]:
                    dE+=-J*(-1)
                    spr+=1
        if y+1 < N:
            if grid_new[x][y]==grid_new[x][y+1] and grid[x][y]!=grid[x][y+1]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x][y+1] and grid[x][y]==grid[x][y+1]:
                    dE+=-J*(-1)
                    spr+=1
        E+=dE       
        if dE > 0:
            zeta = random.random()
            if zeta > np.exp(-dE*k):
                grid_new[x][y] = old_value
        if i>10**3 and i%1000==0:
            E1_list.append(E/N)
            iterations_list.append(i)

        grid=grid_new.copy()
    return grid_new, grid_start, E1_list, iterations_list


@jit(nopython = True)
def Potts_el(q,N,iterations,J,k):
    grid=np.random.randint(1, high=q+1, size=(N,N))
    grid_start=grid.copy()
    spr=0
    iterations_list=0
    iterations_list_M=[]
    E1_list=[]
    M1_list=0
    E=0
    for i in range(iterations):
        print(i)
        grid_new=grid.copy()
        x=random.randint(0,N)
        y=random.randint(0,N)
        old_value=grid[x][y]
        grid_new[x][y]=np.random.randint(1, high=q+1)
        dE=0
        if x-1 >=0:
            if grid_new[x][y]==grid_new[x-1][y] and grid[x][y]!=grid[x-1][y]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x-1][y] and grid[x][y]==grid[x-1][y]:
                    dE+=-J*(-1)
                    spr+=1
        if x+1 < N:
            if grid_new[x][y]==grid_new[x+1][y] and grid[x][y]!=grid[x+1][y]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x+1][y] and grid[x][y]==grid[x+1][y]:
                    dE+=-J*(-1)
                    spr+=1
        if y-1 >=0:
            if grid_new[x][y]==grid_new[x][y-1] and grid[x][y]!=grid[x][y-1]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x][y-1] and grid[x][y]==grid[x][y-1]:
                    dE+=-J*(-1)
                    spr+=1
        if y+1 < N:
            if grid_new[x][y]==grid_new[x][y+1] and grid[x][y]!=grid[x][y+1]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x][y+1] and grid[x][y]==grid[x][y+1]:
                    dE+=-J*(-1)
                    spr+=1
        E+=dE       
        if dE > 0:
            zeta = random.random()
            if zeta > np.exp(-dE*k):
                grid_new[x][y] = old_value
                E-=dE
        time_one=0
        M=0
        if i>10**4 and i%100==0:
            print(E)
            E=energy(grid,J,N)
            E1_list.append(E/N)
            iterations_list_M.append(i)
        grid=grid_new.copy()
    return grid_new, grid_start, E1_list,M1_list, iterations_list,iterations_list_M

@jit(nopython = True)
def Potts_mag(q,N,iterations,J,k):
    grid=np.zeros((N,N))
    grid_start=grid.copy()
    grid_new=np.zeros((N,N))
    spr=0
    iterations_list=[]
    iterations_list_M=[]
    E1_list=0
    M1_list=[]
    E=0
    for i in range(iterations):
        grid_new=grid.copy()
        x=random.randint(0,N)
        y=random.randint(0,N)
        old_value=grid[x][y]
        grid_new[x][y]=np.random.randint(1, high=q+1)
        dE=0
        if x-1 >=0:
            if grid_new[x][y]==grid_new[x-1][y] and grid[x][y]!=grid[x-1][y]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x-1][y] and grid[x][y]==grid[x-1][y]:
                    dE+=-J*(-1)
                    spr+=1
        if x+1 < N:
            if grid_new[x][y]==grid_new[x+1][y] and grid[x][y]!=grid[x+1][y]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x+1][y] and grid[x][y]==grid[x+1][y]:
                    dE+=-J*(-1)
                    spr+=1
        if y-1 >=0:
            if grid_new[x][y]==grid_new[x][y-1] and grid[x][y]!=grid[x][y-1]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x][y-1] and grid[x][y]==grid[x][y-1]:
                    dE+=-J*(-1)
                    spr+=1
        if y+1 < N:
            if grid_new[x][y]==grid_new[x][y+1] and grid[x][y]!=grid[x][y+1]:
                    dE+=-J*(1)
                    spr+=1
            if grid_new[x][y]!=grid_new[x][y+1] and grid[x][y]==grid[x][y+1]:
                    dE+=-J*(-1)
                    spr+=1
        E+=dE       
        if dE > 0:
            zeta = random.random()
            if zeta > np.exp(-dE*k):
                grid_new[x][y] = old_value
                E-=dE
        time_one=0
        #E1_list.append(E/N)
        M=0
        if i>10**4 and i%100==0:
            for x in range(N):
                for y in range(N):
                    M+=np.exp(2*np.pi*1j*(grid[x,y]-1)/q)
            M1_list.append(abs(M))
            iterations_list_M.append(i)
        iterations_list.append(i)
        grid=grid_new.copy()
    return grid_new, grid_start, E1_list,M1_list, iterations_list,iterations_list_M

@jit(nopython = True)
def energy(grid,J,N):
    E=0
    for x in range(N):
        for y in range(N):
            if x-1 >=0:
                if grid[x][y]==grid[x-1][y]:
                        E+=-J*(1)
            if x+1 < N:
                if grid[x][y]==grid[x+1][y]:
                        E+=-J*(1)
            if y-1 >=0:
                if grid[x][y]==grid[x][y-1]:
                        E+=-J*(1)
            if y+1 < N:
                if grid[x][y]==grid[x][y+1]:
                        E+=-J*(1)
    return E
              
def magnetization(grid,J,q,N):
    M=0
    for x in range(N):
        for y in range(N):
            M+=np.exp(2*np.pi*1j*(grid[x,y]-1)/q)
    return M

def energy_c(N,J,grid):
    N_grid = grid.shape[0]
    S=grid
    E = np.zeros((N_grid,N_grid))
    for x in range(N_grid):
        for y in range(N_grid):
            if x-1 >=0:
                if grid[x][y]==grid[x-1][y]:
                        E[x,y]+=-J*(1)
            elif x+1 < N:
                if grid[x][y]==grid[x+1][y]:
                        E[x,y]+=-J*(1)
            if y-1 >=0:
                if grid[x][y]==grid[x][y-1]:
                        E[x,y]+=-J*(1)
            if y+1 < N:
                if grid[x][y]==grid[x][y+1]:
                        E[x,y]+=-J*(1)
    return E
