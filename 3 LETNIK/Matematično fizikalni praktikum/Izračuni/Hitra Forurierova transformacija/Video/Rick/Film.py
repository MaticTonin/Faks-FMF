from numpy import fft
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.ticker import LogLocator
import time
import os

fsampling = 44100
fc = fsampling/2
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(THIS_FOLDER, "Rick.wav.txt")
fnamepic = os.path.join(THIS_FOLDER, "Back")
file = open(fname, "r")

xpremik = 1800
xinterval = 1800
#xinterval ne sme biti manjši od xpremik/2
#je tudi nesmiselno da bi bil
def Podaljšanje(f):  #število podaljšanj
    f=np.asarray(f,dtype=float)
    N=len(f)
    h=np.zeros(N*2)
    for i in range(N):
            h[i]=f[i]
    return h

def Autocorelation1(Inverse): #naredi autokorelacijo
    N=len(Inverse)
    Auto=[]
    F=Inverse
    #F=np.fft.ifftshift(Inverse)
    for i in range(N):
        Auto.append((1/N-i)*F[i])
    return Auto
i = 0
j = 0
f = np.zeros(2*xinterval)
k1 = xinterval
k2 = xinterval
ind=1
start1 = time.time()
for line in file:
	if i < 2*xinterval:
		f[j] = float(line.split('\n')[0])

	elif j == 2*xinterval: 
		x = np.linspace(i - 2*xinterval, i - 1, 2*xinterval)
		freq = np.linspace(-fc, fc, 2*xinterval)
	
		#plt.figure()
		#plt.plot(x, f)

		fft = np.fft.ifftshift(np.fft.fft(f))
		#plt.figure()
		#plt.plot(freq, (fft).real)
		#plt.figure()
		#plt.plot(freq, (fft).imag)
		plt.figure()
		#plt.plot(freq, np.absolute(fft))
		plt.title('Rick and Morty '+ str(ind) + " 44100 Hz")
		plt.bar(freq[math.ceil(2*xinterval/2):], (np.absolute(fft))[math.ceil(2*xinterval/2):], width = fc/xinterval)
		plt.xlabel("Frekvenca")
		plt.ylabel("Amplituda")
		plt.xlim(-200,7000)
		plt.ylim(0,5000000)
		#plt.figure()
		#plt.semilogx(freq[math.ceil(2*xinterval/2):], np.absolute(fft)[math.ceil(2*xinterval/2):])
	
		#spekter nazaj shiftamo pred ifft
		#ifft = np.fft.ifft(np.fft.fftshift(fft))
		#plt.figure()
		#plt.plot(x, (ifft).real)
		#plt.figure()
		#plt.plot(x, (ifft).imag)
		plt.savefig(fnamepic+"Rick and Morty " + str(ind) + ".png")
		
		k1 = k2
		k2 = k1 + xpremik
		razlika = (k1 + xinterval) - (k2 - xinterval)
		j = razlika 
		f = np.roll(f, razlika)
		f[j] = float(line.split('\n')[0])
		ind+=1
		print(ind)
	else:
		f[j] = float(line.split('\n')[0])			
		
	i += 1
	j += 1
	


end1 = time.time()
print(end1-start1)