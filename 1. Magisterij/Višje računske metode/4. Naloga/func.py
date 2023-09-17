import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#Vsota po i, j (pazi sode lihe)
def U_matrix(psi,z,i_state, j_pairs, n_base):
    psi_dummy=np.zeros(len(psi))
    binary=np.binary_repr(i_state,n_base) #Binarna oblika i-tega stanja od PSI
    #Delovanje operatorja U na stanje te oblike, matrično množenje na roke
    if binary[j_pairs:j_pairs+2] == "00":
        psi_dummy[i_state] = psi[i_state] * np.exp(z)

    if binary[j_pairs:j_pairs+2] == "10":
        i_binary = binary[:j_pairs] + "01" + binary[j_pairs+2:] #Dummy index
        psi_dummy[i_state] = np.exp(-1*z) * (np.sinh(2*z) * psi[i_state] + np.cosh(2*z) * psi[int(i_binary,2)])

    if binary[j_pairs:j_pairs+2] == "01":
        i_binary = binary[:j_pairs] + "10" + binary[j_pairs+2:] #Dummy to get index of state
        psi_dummy[i_state] = np.exp(-1*z) * (np.cosh(2*z) * psi[i_state] + np.sinh(2*z) * psi[int(i_binary,2)])

    if binary[j_pairs:j_pairs+2] == "11":
        psi_dummy[i_state] = psi[i_state] * np.exp(z)
    
    return psi_dummy

def U_edge_matrix(psi,z,i_state, j_pairs, n_base):
    psi_dummy=np.zeros(len(psi))
    binary=np.binary_repr(i_state,n_base) #Binarna oblika i-tega stanja od PSI
    #Delovanje operatorja U na stanje te oblike, matrično množenje na roke
    if binary[-1] + binary[0]== "00":
        psi_dummy[i_state] = psi[i_state] * np.exp(z)

    if binary[-1] + binary[0] == "10":
        i_binary ="1"+ binary[1:-1] + "0" 
        psi_dummy[i_state] = np.exp(-1*z) * (np.sinh(z) * psi[i_state] + np.cosh(z) * psi[int(i_binary,2)])

    if binary[-1] + binary[0] == "01":
        i_binary ="0"+ binary[1:-1] + "1"  #Dummy to get index of state
        psi_dummy[i_state] = np.exp(-1*z) * (np.cosh(z) * psi[i_state] + np.sinh(z) * psi[int(i_binary,2)])

    if binary[-1] + binary[0] == "11":
        psi_dummy[i_state] = psi[i_state] * np.exp(z)
    
    return psi_dummy


def exp_2j(psi,z,n):
    psi_new=np.zeros(len(psi))
    for j in range(n):
        if (j%2==0):
            for i in range(len(psi)):
                psi_dummy=U_matrix(psi,z,i, j, n)
                psi_new+=psi_dummy
            #psi=psi_new.copy()
    return psi_new

def exp_2j_1(psi,z,n):
    psi_new=np.zeros(len(psi))
    for j in range(n):
        if (j%2==1):
            if j==n-1:
                for i in range(len(psi)):
                    psi_dummy=U_edge_matrix(psi,z,i, j, n)
                    psi_new+=psi_dummy
                #psi=psi_new.copy()

            for i in range(len(psi)):
                psi_dummy=U_matrix(psi,z,i, j, n)
                psi_new+=psi_dummy
            #psi=psi_new.copy()
    return psi_new

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

def method(psi0,n,z,func,K):
    for i in range(K):
        psi=func(psi0,z,n)
        psi0=psi.copy()
    return psi


def new_psi(n):
    psi = np.random.randn(2**n)/(2**n)
    norm=np.sum(psi**2)
    psi/=np.sqrt(norm)
    psi=psi.astype(np.complex128)
    return psi
    
def simulation(n,tau,S_2,K):
    psi0=new_psi(n)
    psi_beta=method(psi0,n,tau,S_2,K)
    return psi_beta



n_list=[3]

tau=0.01
beta_list=np.linspace(0.1,2,10)
# beta=np.exp(temp)
from multiprocessing.pool import Pool
if __name__ == '__main__':
    for n in tqdm(n_list):
        toplotna=[]
        Z_list=[]
        psi0=new_psi(n)
        if n==2:
            generator=int(2)
        else:
            generator=int(30)
        for beta in tqdm(beta_list):
            t=np.arange(0,beta,tau)
            K=len(t)
            index=0
            psi_matrix=[]
            Z=0
            for i in tqdm(range(generator)):
                psi_beta=simulation(n,(tau)*1j/2,S_4,K)
            #with Pool(8) as pool:
                #for psi_beta in pool.starmap(simulation,[[psi0,n,t[1]-t[0],S_4,K] for i in range(generator)]):
                #print(np.dot(np.conj(psi_beta), psi_beta), Z)
                Z+=(np.dot(psi_beta, np.conj(psi_beta)))
            Z=Z/generator
            Z_list.append(Z)
            #psi_matrix_sum=sum(psi_matrix)
            #print(abs(psi_matrix[0])**2)
            toplotna.append(-1/beta*np.log(Z))
        #plt.plot(beta_list,Z_list, label="Z, $%i$" %n)
        plt.plot(beta_list,toplotna,'-', label="$%i$" %n)
    plt.legend()
    plt.show()