import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt

def ft(t_arr, h_arr, real=True, zero_pading = 0):
    """calculate discrete fourier transform of equidistant set"""
    #for real functions return cos/sin
    #works only if t in t_arr are equidistant
    delta = t_arr[1] - t_arr[0]
    if zero_pading != 0:
        for i in range(zero_pading):
            t_arr = np.append(t_arr, t_arr[-1] + delta)
            h_arr = np.append(h_arr, 0)
    N = len(t_arr)
    
    if N % 2 == 0:
        k_set = np.arange(-N / 2, N / 2)
    else:
        k_set = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1)

    fk_set = k_set / (delta * N)
    H = np.array([])

    for f in fk_set:
        h = 0
        for i in range(N):
            h += h_arr[i] * np.exp(-2j * np.pi * f * t_arr[i])
        H = np.append(H, h)

    if real:
        return fk_set[int(N / 2):], 2 * np.real(H[int(N / 2):]), - 2 * np.imag(H[int(N / 2):])
    else:
        return fk_set, H

#generate data for transform


t = np.linspace(-20, 20, 1000, endpoint=False)
#y = 15 * np.cos(2*np.pi * t * 0.5) +  np.cos(2*np.pi * t * 3)

gauss = lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
const = lambda x: 1
y = gauss(t)
print(y)
#y = [const(i) for i in t]

#k, c, s = ft(t, y, zero_pading=0, real=True)
k, H = ft(t, y, zero_pading=0, real=False)
print(k)


#plt.plot(k, c, marker=".")
plt.plot(k, np.real(H), marker=".")
plt.plot(k, np.sqrt(2*np.pi) * gauss(2*np.pi*k))

plt.show()
plt.close()

y = np.sin(2*np.pi*t)
print(y)
#y = [const(i) for i in t]

#k, c, s = ft(t, y, zero_pading=0, real=True)
k, H = ft(t, y, zero_pading=0, real=False)
print(k)


#plt.plot(k, c, marker=".")
plt.plot(k, np.real(H), marker=".")

plt.show()
plt.close()