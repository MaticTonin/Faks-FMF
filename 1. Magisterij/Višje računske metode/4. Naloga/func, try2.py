import numpy as np
from numba import jit,njit
from multiprocessing import Pool


def U_matrix(z):
    return np.exp(-z)*np.array([[np.exp(2*z),0,0,0],[0,np.cosh(2*z),np.sinh(2*z),0],[0,np.sinh(2*z),np.cosh(2*z),0], [0,0,0,np.exp(2*z)]], dtype=np.complex128)

def new_psi(n):
    psi = np.random.randn(2**n)/(2**n)
    norm=np.sum(psi**2)
    psi/=np.sqrt(norm)
    psi=psi.astype(np.complex128)

    return psi
@njit
def get_b_index(i,j):
    i=i//2**j 
    return i%2
 

@njit
def change_nth_digit(x,d,n):
    d0=get_b_index(x,n)
    if d0==d: return x
    bn=2**n
    x-=d0*bn
    x+=d*bn

    return x

DEC_INT=np.array([[int(f'0b{i}{j}',2) for j in range(2)] for i in range(2)])

@njit
def calc_psi_prime(psi, U, j_mat, N):
    psi_new= np.zeros(2**N, dtype=np.complex128)
    jp1=(j_mat+1)%N
    for i in range(2**N):
        bj=get_b_index(i, j_mat)
        bjp1=get_b_index(i,jp1)
        row=bj*2+bjp1
        for j in range(2):
            for k in range(2):
                new_ind=change_nth_digit(i,j,j_mat)
                new_ind=change_nth_digit(new_ind,k,jp1)
                psi_new[i]+= U[row, DEC_INT[j,k]]*psi[new_ind]
    return psi_new

@njit
def H_psi(psi, j_mat, N):
    psi_new=np.zeros(2**N,dtype=np.complex128)
    for i in range(2**N):
        bj=get_b_index(i,j_mat)
        jp1=(j_mat+1)%N
        bjp1=get_b_index(i,jp1)
        new_ind=i+2**(jp1)-2**j_mat
        psi_new[i]=(-1)**(bj-bjp1)*psi[i]+2*abs(bj-bjp1)*psi[new_ind]
    return psi_new

def H_generator(psi,n):
    psi_new=np.zeros(2**n, dtype=np.complex128)
    for j in range(n):
        jp1=(j+1)%n

        for i in range(len(psi)):
            bj=get_b_index(i,j)
            bjp1=get_b_index(i,jp1)
            bdiff = bj - bjp1
            psi_new[i] += (-1)**bdiff*psi[i]

            if bdiff !=0:
                new_ind = change_nth_digit(i,bjp1,j)
                new_ind = change_nth_digit(i,bj,jp1)
                psi_new += 2*psi[new_ind]
    return psi_new


def exp_2j(psi,z,n):
    U=U_matrix(z)
    for j in range(int(n/2)):
        psi=calc_psi_prime(psi,U,2*j,n)
    #print(np.dot(psi_new, np.conj(psi_new)))
    return psi


def exp_2j_1(psi,z,n):
    U=U_matrix(z)
    for j in range(int(n/2)):
        psi=calc_psi_prime(psi,U,2*j+1,n)
    return psi



def S_2(psi,z,n):
    psi=exp_2j(psi,z/2,n)
    psi=exp_2j_1(psi,z,n)
    psi=exp_2j(psi,z/2,n)
    return psi



def S_4(psi, z,n):
    x_0=-2**(1/3)/(2-2**(1/3))
    x_1=1/(2-2**(1/3))
    psi=S_2(psi,x_1*z,n)
    psi=S_2(psi,x_0*z,n)
    psi=S_2(psi,x_1*z,n)
    return psi


@njit
def method(psi0,n,z,func,K):
    for i in range(K):
        psi=func(psi0,z,n)
        psi0=psi.copy()
    return psi
@njit
def simulation(n,tau,S_4,K):
    psi0=new_psi(n)
    psi_beta=method(psi0,n,-1j*tau,S_4,K)
    return psi_beta

def Z_function(psi,beta,n,tau):
    K=int(beta/2/tau)
    beta=np.zeros(K)
    Z=np.zeros(K)
    beta[0]=0
    Z[0]=np.sum(np.abs(psi)**2)
    for i in range(K):
        psi = S_4(-tau,psi,n) # for real integration tau = -1j*tau but for imaginary tau = -(1j*(-1j*tau)) = -tau
        beta[i+1] = beta[i] + tau
        Z[i+1] = np.sum(abs(psi)**2).real
    return beta*2,Z

def calculate_Z(generator, beta, n ,tau):
    psi_random=[new_psi(n) for i in range((generator))]
    with Pool(14) as pool:
        resoult = pool.starmap(Z_function,[[p,beta,n,tau] for p in psi_random])
    
    beta=resoult[0][0]
    Z = np.array([resoult[i][1] for i in range(generator)]) # stack all averages in one array
    Z = np.average(Z,axis=0) # get Z by averaging over the columns

    return beta, Z
def simulate(psi,t_max,tau,N):

    nsteps = int(np.abs(t_max/tau))
    final_step = t_max - nsteps*tau

    for _ in range(nsteps):
        psi = S_4(psi,-1j*tau,N)
        #print(np.sum(np.abs(psi)**2))
    
    if np.abs(final_step)>0: psi = S_4(psi,final_step,N)
    
    return psi

from tqdm import tqdm
import matplotlib.pyplot as plt
#if __name__ == "__main__":
#    N=6
#    beta = np.linspace(0.1,2,10)
#    Z = np.zeros(beta.shape[0])
#    for i in range(len(beta)):
#        psis = [new_psi(N) for i in range(64)]
#        with Pool(14) as pool:
#            res = pool.starmap(simulate,[[p,-1j*beta[i]/2,-1j*1e-3,N] for p in psis])
#        Z[i] = np.sum([np.abs(r)**2 for r in res])/len(psis)
#    plt.plot(beta,-np.log(Z)/beta)
#    plt.show()
#    plt.close()

n_list=[2,3,4,6]
K=20
tau=0.01
beta_list=np.linspace(0.2,4,20)
# beta=np.exp(temp)
from multiprocessing.pool import Pool
fig,ax=plt.subplots()
fig1,ax1=plt.subplots()
if __name__ == '__main__':
    generator=40
    index=0
    beta=4
    for n in tqdm(n_list):
        psi_0_list=[]
        Z_list=[]
        for i in range(generator):
                psi_0_list.append(new_psi(n))
        psi_0_list=np.array(psi_0_list)
        jndex=0
        for psi in psi_0_list:
            K=int((beta/tau)/2)
            beta_list=np.zeros(K+1)
            Z_generator=np.zeros(K+1)
            beta_list[0]=0
            Z_generator[0]=np.sum(np.abs(psi)**2)
            for i in range(K):
                psi = S_4(psi,-tau,n)
                beta_list[i+1] = beta_list[i] + tau
                Z=np.sum(abs(psi)**2)
                Z_generator[i+1] = Z
            if jndex==0:
                beta_plot=beta_list*2
                jndex+=1
            Z_list.append(Z_generator)
        Z_list=np.array(Z_list)
        Z_plot=np.average(Z_list,axis=0)
        plt.plot(beta_plot[1:],-np.log(Z_plot[1:])/beta_plot[1:], label="$%i$" %n, color=plt.cm.coolwarm(index/len(n_list)))
        plt.xlabel('$\\beta$')
        plt.ylabel('F')
        index+=1
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()



n_list=[2,6,8]
K=20
tau=0.001
beta_list=np.linspace(0.2,2,20)
# beta=np.exp(temp)
from multiprocessing.pool import Pool
#if __name__ == '__main__':
#    for n in tqdm(n_list):
#        toplotna=[]
#        Z_list=[]
#        psi0=new_psi(n)
#        generator=int(50)
#        for beta in tqdm(beta_list):
#            t=np.arange(0,beta,tau)
#            K=len(t)
#            index=0
#            psi_matrix=[]
#            Z=0
#            for i in range(generator):
#                psi_beta=simulation(n,-(tau)*1j/2,S_4,K)
#            #with Pool(6) as pool:
#                #for psi_beta in pool.starmap(simulation,[[n,(tau)*1j/2,S_4,K] for i in range(generator)]):
#                #print(np.dot(np.conj(psi_beta), psi_beta), Z)
#                Z+=(np.dot(np.conj(psi_beta), psi_beta))
#            Z=Z/generator
#            Z_list.append(Z)
#            #psi_matrix_sum=sum(psi_matrix)
#            #print(abs(psi_matrix[0])**2)
#            toplotna.append(-1/beta*np.log(Z))
#        #plt.plot(beta_list,Z_list, label="Z, $%i$" %n)
#        plt.plot(beta_list,toplotna,'-', label="$%i$" %n)
#    plt.xlabel(r"$\beta$")
#    plt.ylabel("F")
#    plt.legend()
#    plt.grid()
#    plt.show()