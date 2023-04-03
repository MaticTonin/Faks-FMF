from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import scipy.signal as sig
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import scipy.optimize as opt
import os
from tqdm import tqdm
#from numba import jit,njit

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
def zero_padding(im, N):
    M = len(im)
    new_im = np.zeros((M + 2*N, M + 2*N))
    new_im[N:-N, N:-N] = im

    return new_im

RMS = "x"
kernel_num = 3
kernel_name="kernel{}.pgm".format(kernel_num)
picture_name="lena_k{}_n{}.pgm".format(kernel_num, RMS)
kernel =  THIS_FOLDER +"\\lena_slike\\"+kernel_name
picture = THIS_FOLDER +"\\lena_slike\\"+picture_name

def hann(N):
    arr = np.array([[0.5 * (1 - np.cos(2*np.pi * i / N)) * 0.5 * (1 - np.cos(2*np.pi * j / N))for i in range(N)] for j in range(N)])
    return arr

def gauss(N, sigma):
    arr = np.array([[np.exp(-(i - N/2) ** 2 / (2*sigma**2 * (N/2)**2)) * np.exp(-(j - N/2) ** 2 / (2*sigma**2 * (N/2)**2))for i in range(N)] for j in range(N)])
    return arr

def welch(N):
    arr = np.array([[(1 - ((i - N/2) / N * 2)**2) * (1 - ((j - N/2) / N * 2)**2) for i in range(N)] for j in range(N)])
    return arr
#@jit(nopython=True)
def notch_filter(data, x0, y0, sigma):
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i, j] *= (1 - np.exp(-((i-x0)**2 + (j-y0)**2) / (2*sigma**2)))


def square(x, y):
    if x <= 226:
        return True
    elif x >= 286:
        return True
    elif y <= 228:
        return True
    elif y >= 285:
        return True
    
    else:
        return False

def kernels(kernel, kernel_name):
    ker = np.loadtxt(kernel, skiprows=3)
    plt.subplot(1,2,1)
    plt.suptitle("Prikaz slike " +str(kernel_name))
    plt.title("XY prostor")
    plt.imshow(ker, cmap="gnuplot", norm = col.Normalize(0, 256))
    ker /= np.sum([np.sum(i) for i in ker]) / 255
    ker = fft.fftshift(ker)
    norm = "ortho"
    KER = fft.fft2(ker, norm=norm)
    KER = fft.fftshift(KER)
    plt.subplot(1,2,2)
    plt.title("Frekvenčni prostor")
    plt.imshow(abs(KER)**2, cmap="gnuplot")
    plt.show()

def creator(kernel, picture,index,gs1,RMS, kernel_num,only_pictures, gauss_yes, filter_yes):

    data = np.loadtxt(picture, skiprows=3)

    ker = np.loadtxt(kernel, skiprows=3) 
    ker /= np.sum([np.sum(i) for i in ker]) / 255

    data = np.array([data[512*i:512*(i+1)] for i in range(512)])
    if only_pictures!="Yes":
        data *= gauss(512, 0.5)
        if gauss_yes=="Yes":
            norm = "ortho"
            DATA = fft.fft2(data, norm=norm)
            DATA = fft.fftshift(DATA)
            data = np.real(fft.ifft2(DATA , norm=norm))
        if gauss_yes!="Yes":
            zeros = 30

            #data = zero_padding(data, zeros)
            #ker = zero_padding(ker, zeros) 


            #data = fft.fftshift(data)
            ker = fft.fftshift(ker)

            N = 0**2 * np.ones((512, 512))

            norm = "ortho"

            DATA = fft.fft2(data, norm=norm)
            KER = fft.fft2(ker, norm=norm)

            neighborhood_size = 3
            threshold = 1000

            filter_data = np.abs(fft.fftshift(DATA))**2

            data_max = filters.maximum_filter(filter_data, neighborhood_size)
            maxima = (filter_data == data_max)
            data_min = filters.minimum_filter(filter_data, neighborhood_size)
            diff = ((data_max - data_min) > threshold)
            maxima[diff == 0] = 0

            labeled, num_objects = ndimage.label(maxima)
            slices = ndimage.find_objects(labeled)
            x, y = [], []

            for dy,dx in slices:
                x_center = (dx.start + dx.stop - 1)/2
                y_center = (dy.start + dy.stop - 1)/2

                if square(x_center, y_center):
                    x.append(x_center)
                    y.append(y_center)

            DATA = fft.fftshift(DATA)

            if filter_yes=="Yes":
                for i in range(len(x)):
                #print(DATA[int(y[i]), int(x[i])])
                    notch_filter(DATA,  int(y[i]), int(x[i]), 4) 

            DATA = fft.fftshift(DATA)
            KER1 = 1 / KER

            maxker = 100
            KER1 = np.array([[i if np.abs(i) < maxker else np.sign(i) * maxker for i in j] for j in KER1])
            data = np.real(fft.ifft2(DATA * KER1 , norm=norm))
    ax1 = plt.subplot(gs1[index])
    plt.axis('off')
    ax1.set_xticklabels([])
    #ax1.set_yticklabels([])
    if only_pictures=="Yes":
        plt.title("lena_k{}_n{}.pgm, ".format(kernel_num, RMS), fontsize="8")
    else:
        plt.title("lena_k{}_n{}.pgm, ".format(kernel_num, RMS)+ "kernel{}.pgm".format(kernel_num), fontsize="8")
    if RMS=="0":
        plt.ylabel("kernel{}.pgm".format(kernel_num))
        #ax1.set_ylabel("kernel{}.pgm".format(kernel_num),  rotation=0)
    ax1.set_aspect('equal')
    ##plt.title(str(kernel_num)+str(picture_name))
    ax1.imshow(data, cmap="gray", norm = col.Normalize(0, 256))
    return ax1

kernel_list=["1","2","3"]
picture_list=["0","4","8","16","x"]
index=0
for i in kernel_list:
    kernel_num = i
    kernel_name="kernel{}.pgm".format(kernel_num)
    kernel =  THIS_FOLDER +"\\lena_slike\\"+kernel_name
    kernels(kernel, kernel_name)

import matplotlib.gridspec as gridspec
plt.figure(figsize = (3,5))
gs1 = gridspec.GridSpec(3, 5)
gs1.update(wspace=0.525, hspace=0.05)
index=0
axes=[]
only_pictures="No"
gauss_yes="Yes"
filter_yes="No"
for i in tqdm(kernel_list):
    for j in picture_list:
        RMS = j
        kernel_num = i
        kernel_name="kernel{}.pgm".format(kernel_num)
        picture_name="lena_k{}_n{}.pgm".format(kernel_num, RMS)
        kernel =  THIS_FOLDER +"\\lena_slike\\"+kernel_name
        picture = THIS_FOLDER +"\\lena_slike\\"+picture_name
        ax=creator(kernel,picture,index, gs1, RMS, kernel_num, only_pictures, gauss_yes, filter_yes)
        axes.append(ax)
        index+=1
if only_pictures=="Yes":
    plt.suptitle("Prikaz slik")
if only_pictures!="Yes" and gauss_yes=="No" and filter_yes=="No":
    plt.suptitle("Prikaz obdelave slik: samo obdelava")
if only_pictures=="No" and gauss_yes=="Yes" and filter_yes=="No":
    plt.suptitle("Prikaz obdelave slik: Gauss")
if only_pictures=="No" and gauss_yes=="No" and filter_yes=="No":
    plt.suptitle("Prikaz obdelave slik: Gauss + kernel")
if only_pictures!="Yes" and gauss_yes=="No" and filter_yes=="Yes":
    plt.suptitle("Prikaz obdelave slik: Gauss + kernel + singularnosti")
plt.show()