from scipy.io.wavfile import read
import numpy as np
import os

print("Ime .wav datoteke: ")
ime = input()
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
ime = os.path.join(THIS_FOLDER, str(ime) + ".wav")

a = read(ime)
b = np.array(a[1],dtype=float)

file = open(ime + ".txt","w") 

for x in b:
	file.write(str(x[1]) + "\n")

file.close() 