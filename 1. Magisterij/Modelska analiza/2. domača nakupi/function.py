import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def making_b_string(limit):
    b_vector=[]
    index=[]
    for i in range(len(limit)):
        if float(limit[i])!=0:
            b_vector.append(limit[i])
            index.append(i)
    return b_vector, index

    

def making_b_input(data):
    b_vector=[]
    index=[]
    for i in range(len(data[0])):
        if i!="1":
            limit=input("Vnesite količino potrebnih " + str(data[0][i]) + ":")
            if limit == "":
                limit=0
            if float(limit)!=0:
                b_vector.append(limit)
                index.append(i-1)
    return b_vector, index

def making_a(index, data):
    A_matrix=[]
    data=np.transpose(data)
    for i in index:
        A_matrix.append(data[int(i)])
    return A_matrix, index

def making_c(index, data):
    c_matrix=[]
    data=np.transpose(data)
    c_matrix.append(data[int(index)])
    return c_matrix