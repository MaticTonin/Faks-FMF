import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.fftpack import fft, fftfreq
from scipy.signal import windows
from scipy.signal import find_peaks
from scipy.signal import peak_widths
import os

#število sample točko
#N = 512 deli 1
#N = 256 deli 2
#N = 128 deli 4
#N = 64  deli 8
deli = 1
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
N = int(512/deli)

cm = plt.get_cmap('rainbow')
filename="val3.dat"
#branje datoteke
def podatki(filename,N):
    f=open(THIS_FOLDER +"\\"+filename,"r")

    lines=f.readlines()
    val2=[]

    #naredimo array iz N meritev
    t=0

    for y in lines:
        if t >= N:
            break
        val2 = np.append(val2, float(y.split('\n')[0]))
        t = t + 1  
    f.close()

    #sample spacing
    x = np.linspace(0,len(val2), len(val2))
    return val2, x
#FFT
N_list=[512,256,126,64]
index=1
plt.suptitle("Prikaz signala "+str(filename))
for i in N_list:
    plt.subplot(2, 2, int(index))
    val2,x=podatki(filename,i)
    val_fft = fft(val2)
    x_fft = fftfreq(i, 1/512)[:i//2]
    plt.title("N="+str(i))
    plt.plot(x, val2, '-', mfc='r')
    index+=1
plt.xlabel("N")
plt.show()



for i in N_list:
    val2,x=podatki(filename,i)
    val_fft = fft(val2)
    x_fft = fftfreq(i, 1/512)[:i//2]
    bart = windows.bartlett(i)
    blck = windows.blackman(i)
    cos = windows.cosine(i)
    #flat = windows.flattop(i)
    hamm = windows.hamming(i)
    han = windows.hann(i)

    okna = [bart, blck, cos, hamm, han]
    ime_okna = ['Bartlett', 'Blackman', 'Cosine', 'Hamming', 'Hann']
    count = 0
    plt.suptitle("Prikaz signala "+str(filename)+", N="+str(i))
    plt.subplot(2, 2, 1)
    plt.title("Signal")
    plt.plot(x, val2, '-', mfc='r')
    plt.grid()
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.title("Realni del FT") 
    plt.plot(x_fft, 2.0/i * val_fft[0:i//2].real, '-', mfc='r')
    plt.grid()
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.title("Imaginarni del FT") 
    plt.plot(x_fft, 2.0/i * val_fft[0:i//2].imag, '-', mfc='r')
    plt.grid()
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.title("Absolutna vrednost FT") 
    plt.plot(x_fft, 2.0/i * np.abs(val_fft[0:i//2])**2, '-', mfc='r')

    plt.grid()
    plt.legend()
    plt.show()


    plt.title("Prikaz uporabljenih oken")
    for j in range(len(okna)):
        count = count + 1
        plt.plot(x, okna[j], color = cm(1.*count/(len(okna)+1)), label=ime_okna[j])
    plt.grid()
    plt.legend()
    plt.show()

    vrh, _ = find_peaks(2.0/i * np.abs(val_fft[0:i//2])**2, 0.6)
    deli=int(512/i)
    for j in range(len(vrh)):
        plt.axvline(x = vrh[j]*(deli), color = 'k', ls = '--')
    count = 0
    plt.xlabel('t')


    plt.yscale('log')
    plt.xlabel(r'$\nu$')
    plt.ylabel('PSD')
    plt.title("Prikaz "+str(filename)+" po transformaciji z različnimi okni pri N="+str(i))
    plt.plot(x_fft, 2.0/i * np.abs(val_fft[0:i//2])**2,color = cm(1.*count/(len(okna)+1)), label='Brez okna')

    for j in range(len(okna)):
        count = count + 1
        plt.plot(x_fft, 2.0/i * np.abs(fft(val2*okna[j])[0:i//2])**2, color = cm(1.*count/(len(okna)+1)), label=ime_okna[j])
    plt.grid()
    plt.legend()
    plt.show()
    

