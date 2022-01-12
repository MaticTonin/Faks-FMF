from scipy.io.wavfile import read
import numpy as np

print("Ime .wav datoteke: ")
ime = input()

a = read(ime + ".wav")
b = np.array(a[1],dtype=float)

file = open(ime + ".txt","w") 

for x in b:
	file.write(str(x[1]) + "\n")

file.close() 