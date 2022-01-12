import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import mpmath as mp
mp.dps = 50

N = 100             #dimenzija matrike A oz. H

def H(N):                   #v resnici je to naša matrika A, imena nisem spreminjal zaradi recikliranja kode
    H = np.zeros((N, N))
    i = 0
    while i < N:
        h = np.zeros(N)
        j = 0
        while j < N:
            if i == j:
                h[j] = -2
            elif abs(i - j) == 1:
                h[j] = 1
            else:
                h[j] = 0
            j += 1
        H[i] = h
        i += 1

    return H

H = H(N)
eig1, eig2 = np.linalg.eigh(H)
eig2 = -np.transpose(eig2)          #ker je d^2 f / dx^2 = -E*f
eig1 = -eig1                        #sedaj so lastne vrednosti razvrščene od največje energije do najmanše (in njim pripadajoči lastni vektorji prav tako)

#plottanje
a = 2       #širina jame
n = 2
x0 = np.linspace(-a/2, a/2, N, endpoint=True)
y0 = eig2[len(eig2)-n]      #n-ti lastni vektor, psi(x)

h = x0[1] - x0[0]       #dolžina koraka
E = np.zeros(N)
for i in range(N):
    E[i] = eig1[i]/h**2
print(E)                #lastne energije, razvrščene od največje do najmanjše

#normirajmo lastno funkcijo (da bo vrh pri 1, za a=2 je to celo prava normalizacija kot jo poznamo iz analitične rešitve)
y0 = y0/max(y0)

plt.plot(x0, y0, color="blue")
plt.xlabel("x")
plt.ylabel("$\psi_n (x)$")
plt.title("Lastna funkcija za n="+str(n))
plt.show()
plt.close()