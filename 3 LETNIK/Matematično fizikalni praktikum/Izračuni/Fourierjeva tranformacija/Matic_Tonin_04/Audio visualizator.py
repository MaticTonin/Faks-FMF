import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
import time
import pygame 
from tkinter import TclError
import os
from pygame._sdl2 import get_num_audio_devices, get_audio_device_name #Get playback device names
from pygame import mixer


from scipy.io.wavfile import read
import numpy as np
mixer.init()
print([get_audio_device_name(x, 0).decode() for x in range(get_num_audio_devices(0))])
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
mixer.init(devicename='Avdio za zaslon Intel(R) - HDMI 1 (Avdio za zaslon Intel(R))')
#fname = os.path.join(THIS_FOLDER,"Undertale - Megalovania.mp3")
fname = os.path.join(THIS_FOLDER,"Undertale OST 001 - Once Upon A Time.wav")
def play_mp3(mp3filename):
    mixer.init()
    mixer.music.load(mp3filename)
    mixer.music.play()
#play_mp3(fname)

 

CHUNK=1024 * 4
FORMAT = pyaudio.paInt16
CHANNELS =1
RATE = 44100
# create matplotlib figure and axes
fig, (ax,ax2) = plt.subplots(2, figsize=(15, 7))

# pyaudio class instance
p = pyaudio.PyAudio()

# stream object to get data from microphone
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# variable for plotting
x = np.arange(0, 2 * CHUNK, 2)
x_fft =np.linspace(0,RATE, CHUNK)
# create a line object with random data
line, = ax.plot(x, np.random.rand(CHUNK), '-', lw=2)
line_fft, =ax2.plot(x_fft,np.random.rand(CHUNK), "-",lw=2)
# basic formatting for the axes
ax.set_title('AUDIO WAVEFORM')
ax.set_xlabel('samples')
ax.set_ylabel('volume')
ax.set_ylim(0, 250)
ax.set_xlim(0, 2 * CHUNK)
plt.setp(ax, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])

ax2.set_xlim(20, RATE/ 4)
ax2.set_ylim(0, 1.2)
# show the plot
plt.show(block=False)

print('stream started')

# for measuring frame rate
frame_count = 0
start_time = time.time()

while True:
    
    # binary data
    data = stream.read(CHUNK)  
    
    # convert data to integers, make np array, then offset it by 127
    data_int=struct.unpack(str(2 * CHUNK) + 'B', data)
    
    # create np array and offset by 128
    data_np = np.array(data_int, dtype='b')[::2] + 128
    
    line.set_ydata(data_np)

    y_fft =np.fft.fft(data_int)
    line_fft.set_ydata(np.abs(y_fft[0:CHUNK]) *2 / (256* CHUNK))
    
    # update figure canvas
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        frame_count += 1
    except TclError:
        
        # calculate average frame rate
        frame_rate = frame_count / (time.time() - start_time)
        
        print('stream stopped')
        print('average frame rate = {:.0f} FPS'.format(frame_rate))
        break

a = read(fname)
b = np.array(a[1],dtype=float)

file = open(fname + ".txt","w") 

for x in b:
	file.write(str(x[1]) + "\n")

file.close()
