import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
n = 0
N = 10**4
nu = 1/2
L = 1
T = 0
R = 0
s = 0
odboji = 0 
nu_list = np.logspace(-2,0.3, 100)
št_sipanj_list = np.zeros((N,len(nu_list)))
j = -1
T_line=[]
R_line=[]
odboji_line=[]
theta_list = np.zeros((N,len(nu_list)))

for nu in tqdm(nu_list):
    T = 0
    R = 0
    j = j + 1 
    odboji = 0
    for i in range(int(N)):
        pot = 0
        sipal = 0
        prvi=True
        while pot < L and pot >= 0:
            sipal+= 1
            if prvi == True:
                pot-=nu*np.log(np.random.rand())
                prvi = False
            else:
                theta=2 * np.random.rand() - 1
                pot+=theta*(np.random.randint(0,2)*2-1)*(-nu*np.log(np.random.rand()))
        theta=np.arccos(theta)*180/np.pi
        if sipal < 100:
            št_sipanj_list[i][j] = sipal
        else:
            št_sipanj_list[i][j] = 100
        if pot > L:
            T =  T + 1 
        elif pot < 0:
            R = R + 1
        odboji+=sipal
        theta_list[i][j]=theta
    odboji = odboji/N
    T_line.append(T/N)
    R_line.append(R/N)
    odboji_line.append(odboji)

T_line_0=[]
R_line_0=[]
odboji_line_0=[]
št_sipanj_list = np.zeros((N,len(nu_list)))
j = -1

for nu in tqdm(nu_list):
    T = 0
    R = 0
    j = j + 1 
    odboji = 0
    for i in range(int(N)):
        pot = 0
        sipal = 0
        prvi=True
        while pot < L and pot >= 0:
            sipal+= 1
            if prvi == True:
                pot-=nu*np.log(np.random.rand())
                prvi = False
            else:
                pot+=(np.random.randint(0,2)*2-1)*(-nu*np.log(np.random.rand()))
        if sipal < 100:
            št_sipanj_list[i][j] = sipal
        else:
            št_sipanj_list[i][j] = 100
        if pot > L:
            T =  T + 1 
        elif pot < 0:
            R = R + 1
        odboji+=sipal
    odboji = odboji/N
    T_line_0.append(T/N)
    R_line_0.append(R/N)
    odboji_line_0.append(odboji)


plt.title(r"Prikaz histograma porazdelitve delcev od kota")
plt.hist(theta_list[:,0], color = plt.cm.gnuplot(3/3), alpha=1, bins=100,label = r'$\nu$ = '+str(nu_list[0]))
plt.hist(theta_list[:,10], color = plt.cm.gnuplot(1/3),alpha=0.5,bins=100, label = r'$\nu$ = '+str(nu_list[10]))
plt.hist(theta_list[:,19], color = plt.cm.gnuplot(2/3),alpha=0.5,bins=100, label = r'$\nu$ = '+str(nu_list[19]))
plt.legend()
plt.grid()
plt.xlabel(r'$\theta[°]$')
plt.show()

plt.title(r"Prikaz histograma porazdelitve delcev od števila sipanj za visoke $\nu$")
plt.hist(št_sipanj_list[:,0], color = plt.cm.gnuplot(3/3), alpha=1, bins=20,label = r'$\nu$ = '+str(nu_list[0]))
plt.hist(št_sipanj_list[:,10], color = plt.cm.gnuplot(1/3),alpha=0.5,bins=50, label = r'$\nu$ = '+str(nu_list[10]))
plt.legend()
plt.grid()
plt.xlabel('Št sipanj')
plt.show()

plt.title(r"Prikaz histograma porazdelitve delcev od števila sipanj za visoke $\nu$")
plt.hist(št_sipanj_list[:,0], color = plt.cm.gnuplot(3/3), alpha=1, bins=10,label = r'$\nu$ = '+str(nu_list[0]))
plt.hist(št_sipanj_list[:,10], color = plt.cm.gnuplot(1/3),alpha=0.5,bins=10,label = r'$\nu$ = '+str(nu_list[10]))
plt.legend()
plt.grid()
plt.xlabel('Št sipanj')
plt.show()



plt.title("Prikaz prepustnosti in odbojnosti pri $%i$" %(N))
plt.plot(nu_list,T_line,"--", label="T izotropno", color="blue")
plt.plot(nu_list,R_line,"--", label="R izotropno", color="red")
plt.plot(nu_list,T_line_0, label="T normal", color="black")
plt.plot(nu_list,R_line_0, label="R normal", color="green")
plt.legend()
plt.xlabel(r"$\nu$")
plt.ylabel("T/R")
plt.grid()
plt.show()


plt.title("Prikaz števila odbojev pri $%i$" %(N))
plt.axvline(x = 0.01, color = 'gray')
plt.text(0.12, max(odboji_line)/2, "Nizke", color='black')
plt.axvspan(0.01, 0.1, alpha=0.2, color='black')
plt.axvline(x = 0.1, color = 'gray')
plt.text(0.5, max(odboji_line)/2, "Srednje", color='blue')
plt.axvspan(0.1, 1, alpha=0.2, color='blue')
plt.axvline(x = 1, color = 'b')
plt.text(1.5, max(odboji_line)/2, "Visoke", color='red')
plt.axvspan(1, 2, alpha=0.2, color='red')
plt.axvline(x = 2, color = 'red')
plt.scatter(nu_list,odboji_line, color="blue")
plt.xlabel(r"$\nu$")
plt.ylabel(r"$N_{odboji}$")
plt.grid()
plt.show()

plt.title("Prikaz števila odbojev pri $%i$ za oba primera" %(N))
plt.axvline(x = 0.01, color = 'gray')
plt.text(0.12, max(odboji_line)/2, "Nizke", color='black')
plt.axvspan(0.01, 0.1, alpha=0.2, color='black')
plt.axvline(x = 0.1, color = 'gray')
plt.text(0.5, max(odboji_line)/2, "Srednje", color='blue')
plt.axvspan(0.1, 1, alpha=0.2, color='blue')
plt.axvline(x = 1, color = 'b')
plt.text(1.5, max(odboji_line)/2, "Visoke", color='red')
plt.axvspan(1, 2, alpha=0.2, color='red')
plt.axvline(x = 2, color = 'red')
plt.plot(nu_list,odboji_line, color="blue", label="Izotropno")
plt.plot(nu_list,odboji_line_0, color="black", label="Normal")
plt.legend()
plt.xlabel(r"$\nu$")
plt.ylabel(r"$N_{odboji}$")
plt.grid()
plt.show()

odboji = 0 
nu_list = np.logspace(-2,1, 50)
št_sipanj_list = np.zeros((N,len(nu_list)))
j = -1
T_line=[]
R_line=[]
odboji_line=[]
nu=1/2
L_list=np.logspace(-2,1, 50)
for L in L_list:
    T = 0
    R = 0
    j = j + 1 
    odboji = 0
    for i in range(N):
        pot = 0
        sipal = 0
        prvi=True
        while pot < L and pot >= 0:
            sipal+= 1
            if prvi == True:
                pot-=nu*np.log(np.random.rand())
                prvi = False
            else:
                pot+=(np.random.randint(0,2)*2-1)*(-nu*np.log(np.random.rand()))
        if sipal < 100:
            št_sipanj_list[i][j] = sipal
        else:
            št_sipanj_list[i][j] = 100
        if pot > L:
            T =  T + 1 
        elif pot < 0:
            R = R + 1
        odboji+=sipal
    odboji = odboji/N
    T_line.append(T/N)
    R_line.append(R/N)
    odboji_line.append(odboji)
plt.title(r"Prikaz prepustnosti in odbojnosti pri $%i$ in $\nu=%.2f$" %(N,nu))
plt.scatter(L_list,T_line, label="Prepustnost T", color="blue")
plt.scatter(L_list,R_line, label="Odbojnost R", color="red")
plt.legend()
plt.xlabel(r"$L$")
plt.ylabel("T/R")
plt.grid()
plt.show()


plt.title(r"Prikaz števila odbojev pri $%i$ in $\nu=%.2f$" %(N,nu))
plt.scatter(L_list,odboji_line, color="blue")
plt.legend()
plt.xlabel(r"$L$")
plt.ylabel(r"$N_{odboji}$")
plt.grid()
plt.show()



N_list=np.linspace(10**1,10**5,20)
L_list=np.logspace(-2,1, 20)
R_matrix=np.zeros((len(L_list),len(nu_list)))
T_matrix=np.zeros((len(L_list),len(nu_list)))
jndex=0
Odboji_matrix=np.zeros((len(L_list),len(nu_list)))
for k in tqdm(L_list): 
    index=0
    j = -1
    for nu in nu_list:
        T = 0
        R = 0
        j = j + 1 
        odboji = 0
        for i in range(N):
            pot = 0
            sipal = 0
            prvi=True
            while pot < k and pot >= 0:
                sipal+= 1
                if prvi == True:
                    pot-=nu*np.log(np.random.rand())
                    prvi = False
                else:
                    pot+=(np.random.randint(0,2)*2-1)*(-nu*np.log(np.random.rand()))
            if sipal < 100:
                št_sipanj_list[i][j] = sipal
            else:
                št_sipanj_list[i][j] = 100
            if pot > k:
                T =  T + 1 
            elif pot < 0:
                R = R + 1
            odboji+=sipal
        odboji = odboji/N
        R_matrix[jndex,index]=R/i
        T_matrix[jndex,index]=T/i
        Odboji_matrix[jndex,index]=odboji
        index+=1
    jndex+=1
X, Y = np.meshgrid(nu_list,L_list)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz odbojnosti od $\nu$ in debeline $L$")
surf = ax.plot_surface(X, Y, R_matrix, cmap="coolwarm",
                       linewidth=0, antialiased=False)
ax.set_ylabel("L")
ax.set_xlabel(r"$\nu$")
ax.set_zlabel("R")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz prepustnosti od $\nu$ in debeline $L$")
surf = ax.plot_surface(X, Y, T_matrix, cmap="coolwarm",
                       linewidth=0, antialiased=False)
ax.set_ylabel("L")
ax.set_xlabel(r"$\nu$")
ax.set_zlabel("T")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz števila sipanj od $\nu$ in N")
surf = ax.plot_surface(X, Y, Odboji_matrix, cmap="coolwarm",
                       linewidth=0, antialiased=False)
ax.set_ylabel("L")
ax.set_xlabel(r"$\nu$")
ax.set_zlabel("Št")
plt.show()

N_list=np.linspace(10**3,10**4,10)
nu_list = np.logspace(-2,0, 10)
R_matrix=np.zeros((len(N_list),len(nu_list)))
T_matrix=np.zeros((len(N_list),len(nu_list)))
L_list=np.linspace(0.5,2.5,20)
jndex=0
Odboji_matrix=np.zeros((len(N_list),len(nu_list)))
for k in tqdm(N_list): 
    št_sipanj_list = np.zeros((int(k),len(nu_list)))
    index=0
    j = -1
    for nu in nu_list:
        T = 0
        R = 0
        j = j + 1 
        odboji = 0
        for i in range(int(k)):
            pot = 0
            sipal = 0
            prvi=True
            while pot < L and pot >= 0:
                sipal+= 1
                if prvi == True:
                    pot-=nu*np.log(np.random.rand())
                    prvi = False
                else:
                    pot+=(np.random.randint(0,2)*2-1)*(-nu*np.log(np.random.rand()))
            if sipal < 100:
                št_sipanj_list[i][j] = sipal
            else:
                št_sipanj_list[i][j] = 100
            if pot > L:
                T =  T + 1 
            elif pot < 0:
                R = R + 1
            odboji+=sipal
        odboji = odboji/k
        R_matrix[jndex,index]=R/i
        T_matrix[jndex,index]=T/i
        Odboji_matrix[jndex,index]=odboji
        index+=1
    jndex+=1

X, Y = np.meshgrid(nu_list,N_list)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz odbojnosti od $\nu$ in N")
surf = ax.plot_surface(X, Y, R_matrix, cmap="coolwarm",
                       linewidth=0, antialiased=False)
ax.set_ylabel("N")
ax.set_xlabel(r"$\nu$")
ax.set_zlabel("R")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz prepustnosti od $\nu$ in N")
surf = ax.plot_surface(X, Y, T_matrix, cmap="coolwarm",
                       linewidth=0, antialiased=False)
ax.set_ylabel("N")
ax.set_xlabel(r"$\nu$")
ax.set_zlabel("T")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r"Prikaz števila sipanj od $\nu$ in N")
surf = ax.plot_surface(X, Y, Odboji_matrix, cmap="coolwarm",
                       linewidth=0, antialiased=False)
ax.set_ylabel("N")
ax.set_xlabel(r"$\nu$")
ax.set_zlabel("Št")
plt.show()
plt.hist(št_sipanj_list[:,0], color = 'b', bins=100,label = r'$\lambda$ = '+str(nu_list[0]))
plt.legend()
plt.xlabel('Število sipanj')
plt.ylabel('Pogostost')
plt.show()

plt.hist(št_sipanj_list[:,3], color = 'r', bins=100,label = r'$\lambda$ = '+str(nu_list[3]))
plt.legend()
plt.xlabel('Število sipanj')
plt.ylabel('Pogostost')
plt.show()

plt.hist(št_sipanj_list[:,6], color = 'k',bins=10, label = r'$\lambda$ = '+str(nu_list[6]))
plt.legend()
plt.xlabel('Število sipanj')
plt.ylabel('Pogostost')
plt.show()

plt.hist(št_sipanj_list[:,9], color = 'g',bins=5, label = r'$\lambda$ = '+str(nu_list[9]))
plt.legend()
plt.xlabel('Število sipanj')
plt.ylabel('Pogostost')
plt.show()

plt.legend()
plt.xlabel('Število sipanj')
plt.ylabel('Pogostost')

#plt.grid(which="both", ls = '--')

plt.show()

