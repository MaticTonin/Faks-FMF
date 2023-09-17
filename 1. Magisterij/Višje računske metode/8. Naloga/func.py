import numpy as np
from numpy.linalg import eigh, norm, svd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def create_hamitolian(n,period):
    delta=1
    h_2=delta*np.diag([1.,-1.,-1.,1.])
    h_2[1,2], h_2[2,1]=2.,2.
    H= np.zeros((2**n,2**n), dtype="complex128")
    for j in range(1,n):
        id_l=np.eye(2 ** (j - 1))
        id_r=np.eye(2 ** (n- j - 1))
        kron_left=np.kron(id_l,h_2)
        H+=np.kron(kron_left, id_r)
    if period==True:
        pauli_x= np.array([[0,1],[1,0]])
        pauli_y= np.array([[0,-1j],[1j,0]])
        pauli_z= np.array([[delta,0],[0,-delta]])
        kron_x=np.kron(pauli_x, np.eye(2**(n-2)))
        kron_y=np.kron(pauli_y, np.eye(2**(n-2)))
        kron_z=np.kron(pauli_z, np.eye(2**(n-2)))
        H+=np.kron(kron_x, pauli_x) +np.kron(kron_y, pauli_y) + np.kron(kron_z, pauli_z)
    return H


def make_psi_matrix(state, state_A, state_B):
    n_a = 2**(len(state_A))
    n_b = 2**(len(state_B))
    psi = np.empty((n_a, n_b), dtype=np.complex128)
    n_basis = len(state)

    for i in range(n_basis):

        #Binarna forma celote
        i_binary = bin(i)[2:]
        i_binary = (int(np.log2(n_basis)) - len(i_binary)) * '0' + i_binary #Doda potrebne nule spredaj

        #Pobere stanja, ki so v posamezni bazi v binarni obliki
        idx_a = '0b'
        for index_A in state_A:
            idx_a += i_binary[index_A]

        idx_b = '0b'
        for index_B in state_B:
            idx_b += i_binary[index_B]
        
        #Transformira stanje nazaj v index v celotnem stanju Psi
        idx_a = int(idx_a, 2)
        idx_b = int(idx_b, 2)
        #Določi v prostoru vrednost stanja, ki pripada tej bazi
        psi[idx_a, idx_b] = state[i]

    return psi
def entropy(schmidt):
    return np.sum(-np.abs(schmidt)**2 * np.log(np.abs(schmidt)**2))

def schmidt_decomposition(n, period, example):
    h=create_hamitolian(n,period)
    #h=heisenberg(n, delta=1., periodic=True)
    state = eigh(h)[1][:,0]
    space = np.array([i for i in range(n)])
    if example==0:
        space_A= space[::2]
        space_B=space[1::2]
    if example==1:
        space_A= space[int(n/2)-1:]
        space_B=space[:int(n/2)-1]
    psi_matrix=make_psi_matrix(state, space_A, space_B)
    #print("Creating svg for n=%i" %n)
    schmidt = svd(psi_matrix, full_matrices=False, compute_uv=False)
    return schmidt

def schmidt_decomposition_matrix(n, period, example):
    h=create_hamitolian(n,period)
    #h=heisenberg(n, delta=1., periodic=True)
    state = eigh(h)[1][:,0]
    space = np.array([i for i in range(n)])
    if example==0:
        space_A= space[::2]
        space_B=space[1::2]
    if example==1:
        space_A= space[int(n/2)-1:]
        space_B=space[:int(n/2)-1]
    psi_matrix=make_psi_matrix(state, space_A, space_B)
    #print("Creating svg for n=%i" %n)
    u, schmidt, v = svd(psi_matrix, full_matrices=False)
    return u,schmidt,v


def line(x, a, b):
 return a * x + b

def x_2(x, a, b, c):
 return a * x**2 + b* x + c

def x_2_x(x, a, c):
 return a * x**2  + c

def x_2_x_c(x, a):
 return a * x**2 
def get_a_from_u(u, k1s2, k2):
    a = np.empty((2, int(k1s2/2), k2), dtype=np.complex128)
    nqbit = int(np.log2(k1s2))
    #print(nqbit)
    for i in range(k1s2):
        bin_k1s2 = format(i, "0"+str(nqbit)+"b")
        row = int(bin_k1s2[-1], 2)
        #print(bin_k1s2)
        col = int(bin_k1s2[:-1], 2)
        a[row, col, :] = u[i, :]
    return a

def MPA(state):
    n=int(np.log2(len(state)))
    A_matrix, B_matrix, V_matrix=[],[],[]
    schmidt= []

    space = np.array([i for i in range(0,n)])
    psi =make_psi_matrix(state, space[:1],space[1:])
    #Step 1
    u, sch_temp,vh=svd(psi, full_matrices=False)
    #Ven vržemo vse ničelne vrednosti
    nonzero_sch = np.append(np.where(sch_temp == 0.)[0], len(sch_temp))[0]
    A_matrix.append(u[:,:nonzero_sch])
    B_matrix.append(A_matrix[-1])
    schmidt.append(sch_temp[:nonzero_sch])
    V_matrix.append(vh[:nonzero_sch])
    psi=np.einsum("i,ij->ij",sch_temp,vh)
    #Middle step
    for i in range(2, n):
        psi=psi.flatten()
        n_u=int(np.log2(u.shape[1]))+1
        psi=make_psi_matrix(psi, space[:n_u], space[n_u:n-i+n_u])
        u, sch_temp, vh = svd(psi, full_matrices=False)
        #Ven vržemo vse neničelne vrednosti
        nonzero_sch=len(sch_temp)
        #nonzero_sch = np.append(np.where(sch_temp == 0.)[0], len(sch_temp))[0]
        #u = u[:, :nonzero_sch]
        k1s2, k2 = u.shape
        A_matrix.append(get_a_from_u(u, k1s2, k2))
        B_matrix.append(np.einsum('i,jik->jik', np.power(schmidt[-1], -1.), A_matrix[-1]))
        schmidt.append(sch_temp[:nonzero_sch])
        V_matrix.append(vh[:nonzero_sch])
        psi = np.einsum('i,ij->ij', sch_temp, vh)

    A_matrix.append(np.transpose(psi))
    B_matrix.append(np.einsum('i,ji->ji', schmidt[-1]**(-1.), A_matrix[-1]))
    return schmidt, A_matrix, B_matrix, V_matrix
