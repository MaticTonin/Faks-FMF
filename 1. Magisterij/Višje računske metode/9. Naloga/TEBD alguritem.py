import numpy as np
from numpy.linalg import eigh, norm, svd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import matplotlib
from tqdm import tqdm
plt.rcParams.update({'font.size': 16})


def create_hamitolian(n, period=True):
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


def heisenberg_mpa(nqbit, delta=1.):
    pauli_x = np.array([[0, 1],[1, 0]], dtype=np.complex_)
    pauli_y = np.array([[0, -1j],[1j, 0]], dtype=np.complex_)
    pauli_z = np.array([[1, 0],[0, -1]], dtype=np.complex_)
    
    operators = []
    for i in range(nqbit-1):
        h_dict = {}
        h_dict[i] = pauli_x
        h_dict[i+1] = pauli_x
        operators.append(h_dict)
        h_dict = {}
        h_dict[i] = pauli_y
        h_dict[i+1] = pauli_y
        operators.append(h_dict)
        h_dict = {}
        h_dict[i] = np.sqrt(delta)*pauli_z
        h_dict[i+1] = np.sqrt(delta)*pauli_z
        operators.append(h_dict)
    return operators

def entropy(schmidt):
    return np.sum(-np.abs(schmidt)**2 * np.log(np.abs(schmidt)**2))


def make_psi_matrix(state, state_A, state_B):
    n_a = 2**(len(state_A))
    n_b = 2**(len(state_B))
    psi = np.empty((n_a, n_b), dtype=np.complex_)
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
    B_matrix.append(np.expand_dims(A_matrix[-1], axis=1))
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
        nonzero_sch = np.append(np.where(sch_temp == 0.)[0], len(sch_temp))[0]
        u = u[:, :nonzero_sch]

        k1s2, k2 = u.shape
        A_matrix.append(get_a_from_u(u, k1s2, k2))
        B_matrix.append(np.einsum('i,jik->jik', np.power(schmidt[-1], -1.), A_matrix[-1]))
        schmidt.append(sch_temp[:nonzero_sch])
        V_matrix.append(vh[:nonzero_sch])
        psi = np.einsum('i,ij->ij', sch_temp, vh)

    A_matrix.append(np.transpose(psi))
    B_matrix.append(np.einsum('i,ji->ji', schmidt[-1]**(-1.), A_matrix[-1]))
    B_matrix[-1] = np.expand_dims(B_matrix[-1], axis=2)
    return schmidt, A_matrix, B_matrix, V_matrix



def u_gate(z, delta=1.):
    u = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    a = np.exp(delta*z, dtype=np.complex128)
    b = np.multiply( np.cosh(2.*z, dtype=np.complex128), np.exp((-1.)*delta*z, dtype=np.complex128))
    c = np.multiply( np.sinh(2.*z, dtype=np.complex128), np.exp((-1.)*delta*z, dtype=np.complex128))
    u[0,0,0,0] = a
    u[1,1,1,1] = a
    u[0, 1, 0, 1] = b
    u[0, 1, 1, 0] = c
    u[1, 0, 0, 1] = c
    u[1, 0, 1, 0] = b
    return u


def avg_energy(E, beta_list):
    avg_ene = []
    for beta in beta_list:
        temp = np.exp(-beta*E)
        avg_ene.append(np.sum(E*temp)/np.sum(temp))
    return np.array(avg_ene)


def ent_entropy(schmidt):
    return np.sum(-np.abs(schmidt)**2. * np.log(np.abs(schmidt)**2.))


def matrix_element(a_mat1, a_mat2, operator_dict):
    n = len(a_mat1)
    indices = operator_dict.keys()
    # L - left vector
    if 0 in indices:
        sp = np.einsum("si,st,tj->ij", np.conjugate(a_mat1[0]), operator_dict[0], a_mat2[0])
    else:
        sp = np.einsum("sk,sl->kl", np.conjugate(a_mat1[0]), a_mat2[0])

    for i in range(1, n-1):
        if i in indices:
            op = operator_dict[i]
            t = np.einsum("sz,skh,zlm->klhm", op, np.conjugate(a_mat1[i]), a_mat2[i])
        else:
            t = np.einsum("skh,slm->klhm", np.conjugate(a_mat1[i]), a_mat2[i])
        sp = np.einsum("ij,ijkl->kl", sp, t)
    
    # R - right vector
    if n-1 in indices:
        r = np.einsum("si,st,tj->ij", np.conjugate(a_mat1[-1]), operator_dict[n-1], a_mat2[-1])
    else:
        r = np.einsum("sk,sl->kl", np.conjugate(a_mat1[-1]), a_mat2[-1]) 
    sp = np.einsum("ij,ij", sp, r)
    return sp



def from_b_to_a(b_matrices, lam):
    a_matrices = [b_matrices[0]]
    for i in range(1, len(b_matrices)):
        a_matrices.append(np.einsum('i,sij->sij', lam[i-1], b_matrices[i]))
    a_matrices[0] = a_matrices[0].reshape(a_matrices[0].shape[::2])
    a_matrices[-1] = a_matrices[-1].reshape(a_matrices[-1].shape[:-1])
    return a_matrices





class TEBD:
    def __init__(self, psi, delta=1.):
        self.init_psi = psi
        self.init_lam, self.init_a_ar, self.init_b_ar = MPA(psi)[:-1]
        self.n = len(self.init_b_ar)
        self.lam, self.a_ar, self.b_ar = self.init_lam, self.init_a_ar, self.init_b_ar 
        self.delta = delta

    def __brake_b(self, u, idx):
        double_b = np.einsum("sij,tjk->stik", self.b_ar[idx]*self.lam[idx], self.b_ar[idx+1] )
        double_b = np.einsum("stij,ijkl->stkl", u, double_b)
        if idx == 0.:
            if idx+1 == self.n-1:
                q = double_b
            else:      
                q = np.einsum("stij,j->isjt", double_b, self.lam[idx+1])
        elif idx+1 == self.n-1:
            q = np.einsum("i,stij->isjt", self.lam[idx-1], double_b)
        else:
            q = np.einsum("i,stij,j->isjt", self.lam[idx-1], double_b, self.lam[idx+1])
        q_shape = q.shape
        q = q.reshape((q_shape[0]*q_shape[1], q_shape[2]*q_shape[3]))
        
        u, schmidt, vh = svd(q, full_matrices=False)
        schmidt /= norm(schmidt)
        nonzero_sch = np.append(np.where(schmidt == 0.)[0], len(schmidt))[0]
        
        u = u[:, :nonzero_sch]
        u_shape = u.shape
        u = u.reshape((u_shape[0] // 2, 2, u_shape[1]))
        if idx == 0.:
            self.b_ar[idx] = np.swapaxes(u, 0, 1)
        else:
            self.b_ar[idx] = np.einsum('i,ijk->jik', np.power(self.lam[idx-1], -1.), u)

        vh = vh[:nonzero_sch]
        vh_shape = vh.shape
        vh = vh.reshape((vh_shape[0], vh_shape[1] // 2, 2))
        if idx+1 == self.n-1:
            self.b_ar[idx+1] = np.transpose(vh, [2, 0, 1])
        else:            
            self.b_ar[idx+1] = np.einsum('ijk,j->kij', vh, np.power(self.lam[idx+1], -1.) )
        self.lam[idx] = schmidt[:nonzero_sch]

    def single_step(self, z, ts, m_max=None):
        for i in range(len(ts[0])):
            u = u_gate(ts[0][i]*z, delta=self.delta)
            for j in range(self.n // 2):
                idx = 2*j
                self.__brake_b(u, idx)
            u = u_gate(ts[1][i]*z, delta=self.delta)
            for j in range( (self.n // 2) - 1):
                idx = 2*j+1
                self.__brake_b(u, idx)
        self.a_ar = from_b_to_a(self.b_ar, self.lam)
        # normalization = sp_transfer(self.a_ar, self.a_ar)
        # for a in self.a_ar:
        #     a = a/normalization
        # self.b_ar = from_a_to_b(self.a_ar, self.lam)
# e0 plot
beta_list_max = 10.
delta = 1.
periodic = False
sym_crt =  +0.1259921049894873E+01
#ts = sym_methods["forest_ruth"]
ts=np.array([
                [(1./(2.*(2. - sym_crt))), ((1. - sym_crt)/(2.*(2. - sym_crt))), ((1. - sym_crt)/(2.*(2. - sym_crt))), (1./(2.*(2. - sym_crt))) ],
                [(1./(2. - sym_crt)), (-sym_crt/(2. - sym_crt)), (1./(2. - sym_crt)), 0.]])
n_list = [4, 6]

num = 1000
colors = cm.rainbow(np.linspace(0, 1, len(n_list)))

beta_list = np.linspace(0., beta_list_max, num=num)
db = beta_list[1]-beta_list[0]



fig, (ax,ax3) = plt.subplots(1, 2)
for n, c in zip(n_list, colors):
    if n <= 10:
        h = create_hamitolian(n, periodic)
        Eig_E = eigh(h)[0]
        ax.hlines(Eig_E[0], beta_list[0], beta_list[-1], color=c, linestyle="--", alpha=0.7)
        avg_ene = avg_energy(Eig_E, beta_list)
        ax.plot(beta_list, avg_ene, color=c, alpha=0.7, linestyle=":")

    state = np.random.normal(0., 1., 2**n) + 1.j * np.random.normal(0., 1., 2**n)
    state = state / norm(state)
    e = []
    h_mpa = heisenberg_mpa(n, delta)
    propagator = TEBD(state)
    for b in beta_list:
        propagator.single_step(-db, ts)

        e_sum = 0.
        for op in h_mpa:
            e_sum += matrix_element(propagator.a_ar, propagator.a_ar, op)
        e.append(e_sum)
    ax.plot(beta_list, e, color=c)
    ax3.plot(beta_list[int(len(e)*3/4):], abs((e[int(len(e)*3/4):]-Eig_E[0])/Eig_E[0]), color=c, label="N="+str(n)+r", E_0=$%.2f$" %(Eig_E[0]))
fig.suptitle(r"Prikaz spremembe $E_0$ od $\beta$")
ax.set_ylabel(r'$E_0$')
ax.set_xlabel(r'$\beta$')
ax3.set_ylabel(r"$\frac{|E_0-E_0(\beta)|}{E_0}$")
ax3.set_xlabel(r"$\beta$")
ax3.legend()
ax2 = ax.twinx()
ax2.plot([],[], color="Black", label=r"$E_0(\beta)$", linestyle=":")
ax2.hlines([],[],[], color="Black", label=r"$E_0$", linestyle="--")
ax2.plot([],[], color="Black", label="TEBD metoda")
ax2.get_yaxis().set_visible(False)

ax.legend(loc=1)
ax2.legend(loc=4)
ax.grid()
ax3.set_yscale("log")
ax3.grid()
plt.show()



n_list = [4, 6, 8]

num = 1000
num_list=[50,75,100,150,200,250,500,750,1000,2500]
colors = cm.rainbow(np.linspace(0, 1, len(num_list)))
colors1 = cm.rainbow(np.linspace(0, 1, len(num_list)))
fig =plt.figure()
fig.suptitle(r"Prikaz spremembe $E_0$ od $\beta$ z parametrom $d\beta$")

fig2 =plt.figure()
index=1
ax4 = fig2.add_subplot(1,1,1)
fig2.suptitle(r"Prikaz časovne zahtevnosti programa")
import time
jndex=0
start = time.time()
print("hello")

for n in tqdm(n_list):
    ax = fig.add_subplot(len(n_list),2,index)
    ax3 = fig.add_subplot(len(n_list),2,index+1)
    time_list=[]
    db_list=[]
    for num,c in zip(num_list, colors):
        beta_list = np.linspace(0., beta_list_max, num=num)
        db = beta_list[1]-beta_list[0]
        h = create_hamitolian(n, periodic)
        Eig_E = eigh(h)[0]
        ax.hlines(Eig_E[0], beta_list[0], beta_list[-1], color=c, linestyle="--", alpha=0.7)
        avg_ene = avg_energy(Eig_E, beta_list)
        ax.plot(beta_list, avg_ene, color=c, alpha=0.7, linestyle=":")
        state = np.random.normal(0., 1., 2**n) + 1.j * np.random.normal(0., 1., 2**n)
        state = state / norm(state)
        e = []
        h_mpa = heisenberg_mpa(n, delta)
        start = time.time()
        propagator = TEBD(state)
        for b in beta_list:
            propagator.single_step(-db, ts)

            e_sum = 0.
            for op in h_mpa:
                e_sum += matrix_element(propagator.a_ar, propagator.a_ar, op)
            e.append(e_sum)
        end = time.time()
        db_list.append(db)
        time_list.append(abs(end-start))
        ax.plot(beta_list, e, color=c)
        ax3.plot(beta_list[int(len(e)*3/4):], abs((e[int(len(e)*3/4):]-Eig_E[0])/Eig_E[0]), color=c, label=r"$d\beta=%.3f$" %db)
    if index==1:
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax4.plot(db_list, time_list, color=plt.cm.rainbow(jndex/len(n_list)), label=r"$n=%i$" %n)
    jndex+=1
    ax.set_title(r"$n=%i$" %n)
    ax3.set_title(r"$n=%i$" %n)
    ax.set_ylabel(r'$E_0$')
    ax.set_xlabel(r'$\beta$')
    ax3.set_ylabel(r"$\frac{|E_0-E_0(\beta)|}{E_0}$")
    ax3.set_xlabel(r"$\beta$")
    if index==5:
        ax2 = ax3.twinx()
        ax2.plot([],[], color="Black", label=r"$E_0(\beta)$", linestyle=":")
        ax2.hlines([],[],[], color="Black", label=r"$E_0$", linestyle="--")
        ax2.plot([],[], color="Black", label="TEBD metoda")
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.get_yaxis().set_visible(False)
    index+=2
    ax.grid()
    ax3.set_yscale("log")
    ax3.grid()
ax4.set_xlabel(r"$d\beta$")
ax4.set_ylabel("time[s]")
ax4.grid()
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.legend()
plt.show()