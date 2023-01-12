from re import X
from turtle import color
import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
# ali naplota E ali c
plot = 0
# 0 ... c
# 1 ... E




J = +1 #feromagneti
#J= -1 #antiferomagneti

#število iteracij
MC = 10**6

#zunanje magnetno polje
Hlist = 0

#dimenzija mreže
N = 50

#začetna porazdelitev
random.seed(4612213169)
mreza = np.zeros((N,N)) 
for i in range(N):                          
    for j in range(N):
        mreza[i][j] = random.randint(0,1)*2-1

mreza0 = mreza

kT_list = np.linspace(0.1,5,10)
Hlist = [0, 0.1, 0.5, 1]
no = 0
Hst = -1
for kT in tqdm(kT_list):
    Hst = Hst + 1
    #začetna energija
    E = 0
    for i in range(N):
        for j in range(N): 

            if i-1 < 0:
                E = E - J*mreza[i][j]*mreza[N-1][j]
            else:
                E = E - J*mreza[i][j]*mreza[i-1][j]

            if i+1 >= N:
                E = E - J*mreza[i][j]*mreza[0][j]
            else:
                E = E - J*mreza[i][j]*mreza[i+1][j]

            if j-1 < 0:
                E = E - J*mreza[i][j]*mreza[i][N-1]
            else:
                E = E - J*mreza[i][j]*mreza[i][j-1]

            if j+1 >= N:
                E = E - J*mreza[i][j]*mreza[i][0]
            else:
                E = E - J*mreza[i][j]*mreza[i][j+1]

            E = E - H*mreza[i][j]
    print(E)
    E0 = E
    #temperatura
    kT_list = np.linspace(0.1,5,20)
    #kT_list=[0,1,10]
    #kT_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3, 4, 5, 6, 7 ,8 ,9 ,10]
    eavr = np.zeros(len(kT_list))
    eavr2 = np.zeros(len(kT_list))
    #(kTc = 2,27J)
    ime = 0
    #
    Elist = np.zeros(MC)
    
    g = 0
    for H in tqdm(Hlist):
        random.seed(4612213169)
        mreza = np.zeros((N,N)) 
        for i in range(N):                          
            for j in range(N):
                mreza[i][j] = random.randint(0,1)*2-1

        mreza0 = mreza

        
        #začetna energija
        E = 0
        for i in range(N):
            for j in range(N): 

                if i-1 < 0:
                    E = E - J*mreza[i][j]*mreza[N-1][j]
                else:
                    E = E - J*mreza[i][j]*mreza[i-1][j]

                if i+1 >= N:
                    E = E - J*mreza[i][j]*mreza[0][j]
                else:
                    E = E - J*mreza[i][j]*mreza[i+1][j]

                if j-1 < 0:
                    E = E - J*mreza[i][j]*mreza[i][N-1]
                else:
                    E = E - J*mreza[i][j]*mreza[i][j-1]

                if j+1 >= N:
                    E = E - J*mreza[i][j]*mreza[i][0]
                else:
                    E = E - J*mreza[i][j]*mreza[i][j+1]

                E = E - H*mreza[i][j]

        for i in range(MC):
        
            #zamenjamo random spin
            random.seed()
            x = random.randint(0,N-1)
            y = random.randint(0,N-1)
            mreza[x][y] = -mreza[x][y]
            #sprememba energije zaradi flippa
            dE = 0
            if x-1 < 0:
                dE = dE - 2*J*mreza[x][y]*mreza[N-1][y]
            else:
                dE = dE - 2*J*mreza[x][y]*mreza[x-1][y]

            if x+1 >= N:
                dE = dE - 2*J*mreza[x][y]*mreza[0][y]
            else:
                dE = dE - 2*J*mreza[x][y]*mreza[x+1][y]

            if y-1 < 0:
                dE = dE - 2*J*mreza[x][y]*mreza[x][N-1]
            else:
                dE = dE - 2*J*mreza[x][y]*mreza[x][y-1]

            if y+1 >= N:
                dE = dE - 2*J*mreza[x][y]*mreza[x][0]
            else:
                dE = dE - 2*J*mreza[x][y]*mreza[x][y+1]

            dE = dE - 2*H*mreza[x][y]
            E = E + dE 

            #pogoj ali zavrnemo spremembo
            if dE > 0:
                e = random.random()
                if e > np.exp(-dE/kT):
                    mreza[x][y] = - mreza[x][y]
                    E = E - dE    
            Elist[i] = E

        st = 600000
        n = 0
        esum = 0
        eprod = 0
        while st < MC:
            esum = esum + Elist[st]
            eprod = eprod + (Elist[st])**2

            n = n + 1 
            st = st + 5000

        eavr[g] = esum/n
        eavr2[g] = eprod/n 
        g = g + 1

        #narišemo graf
    #    plt.text(1,-1,'kT = '+str(kT))
    #    plt.plot(Elist/N/N, color = barva[no], label = 'T = '+str(kT))
    #    #no = no + 1
    #plt.xlabel('i MC')
    #plt.ylabel(r'E/$N^2$')
    #plt.legend()
    #plt.show()

    c = np.zeros(len(eavr))
    
    for i in range(len(eavr)):
        c[i] =  (eavr2[i] - eavr[i]**2)/N/kT_list[i]**2

    if plot == 0:
        plt.title("Prikaz specifične toplote v odvisnosti od $H$")
        plt.plot(kT_list, c, label= 'H='+str(H), color =plt.cm.rainbow(no/len(Hlist)))
    if plot == 1:
        plt.title("Prikaz povprečne energije v odvisnosti od $H$")
        plt.plot(kT_list, eavr/N/N, label= 'H='+str(H), color = plt.cm.rainbow(no/len(Hlist)))
    no = no + 1

if plot == 0:
    plt.ylabel(r'c')
if plot == 1:
    plt.ylabel(r'$\frac{<E>}{N^2}$')
plt.legend()
plt.xlabel('$k_BT$')
plt.axvline(x=2.27, color = 'k', ls = ':')
plt.grid()
plt.show()
plt.clf()

#for i in range(len(Hlist)):
#    plt.plot(kT_list, c[i][:], label= 'H='+str(Hlist[i]), color = barva[i])
#
#plt.legend()
#plt.xlabel('kT')
#plt.ylabel(r'c')
#plt.axvline(x=2.27, color = 'k', ls = ':')
#plt.grid()
#
#
#plt.savefig('cT'+str(ime))
#plt.clf()
    


        #plt.savefig(ime)
