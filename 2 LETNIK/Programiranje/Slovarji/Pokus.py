import math
def dolzina_vrstice(Vhodna):
    n=0
    with open(Vhodna, 'r') as f1:
        while (True):
            a = f1.readline()
            if a == "" or a == "\n": break
            n+=1
    return n

def beri_tabelo(Vhodna, n):
    with open(Vhodna, 'r') as f1:
        tabela=[None]*n
        for i in range(n):
            tabela[i]=float(f1.readline())
            if tabela[i]== ".":
                tabela[i]=","
    return pisi_tabelo(tabela, Izhodna, n)
def pisi_tabelo(tabela, Izhodna, n):
    with open(Izhodna, 'w') as f2:
        for i in range(n):
             f2.write("%f\n" % tabela[i])


Vhodna=input("Vnesite ime vhodne datoteke: ")
Izhodna=input("Vnesite ime vhodne datoteke: ")




