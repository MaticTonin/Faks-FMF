import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from mpmath import *
import time
epsilon = 1e-10
alpha = 0.355028053887817239  # alpha = Ai(0) = Bi(0)/3^(1/2)
beta = 0.258819403792806798  # beta = Bi'(0)/3^(1/2) = -Ai'(0)



np.seterr(all='warn')
print("Range of numpy double:", np.finfo(np.double).min, np.finfo(np.double).max)
A = np.array([143],dtype='double')
a=A[-1]
print("At the border:", a**a)
B = np.array([144],dtype='double')
b=B[-1]
print("Blowing out of range:", b**b)
#Maclaurinova vrsta ya x~0
def MacLaurinovaVrsta(x):
    fx=1.
    gx=x
    Ai=alpha*fx-beta*gx
    Bi=3.**(0.5)*(alpha*fx+beta*gx)
    #Absolutna napaka
    Ai_pred =float("inf")
    Bi_pred =float("inf")
    i=1

    while(np.abs(Ai_pred-Ai) >= epsilon or np.abs(Bi_pred-Bi) >= epsilon):
          Ai_pred = Ai
          Bi_pred = Bi
          #Sprememba členov z vrsto
          fx=fx*(3. * x**3. *((1. / 3.) + i - 1.)) / ((3.*i - 1.) * (3.*i) * (3.*i - 2.))
          gx=gx*(x ** (3.) * 3. * ((2. / 3.) + i -1.)) / ((3.*i + 1.) * (3.*i) * (3.*i - 1.))
          Ai= Ai+ alpha*fx-beta*gx
          Bi= Bi +3.**(0.5)*(alpha*fx+beta*gx)
          i=i+1
    return Ai,Bi

          

#Asimptotska vrsta
def AsimptotskaVrsta(x):
    ksi= 2./3.* np.abs(x)**(3./2.)
    Ai_pred =float("inf")
    Bi_pred =float("inf")
    j=1
    #Za velike pozitivne x
    if (x>=0.):
        Lfaktor= 1.
        Aifaktor= np.exp(-ksi) / (2* np.pi ** (1/2) * (x)**(1/4))
        Ai=Aifaktor
        Bifaktor= np.exp(ksi) / (np.pi ** 0.5 * x ** 0.25)
        Bi=Bifaktor
        while (np.abs(Ai_pred-Ai) >= epsilon or np.abs(Bi_pred-Bi) >= epsilon):
            Ai_pred =Ai
            Bi_pred =Bi
            #Sprememba L faktorja za vsak potek 
            Lfaktor=Lfaktor *(3. * (j - 1.) + 5. / 2.) * (3. * (j - 1.) + 3. / 2.) * (3. * (j - 1.) + 1. / 2.) / (54 * j * (j - 0.5) * ksi)
            Ai= Ai+ Aifaktor *(-1)**j *Lfaktor
            Bi= Bi+ Bifaktor *Lfaktor
            j=j+1.
            
    #Za absoluto velike negativne člene
    elif (x < 0):
        Pclen = 1.
        Qclen = (2. + 0.5) * (1. + 0.5) / (54. * ksi)
        faktor = 1. / (np.pi ** 0.5 * (-x) ** 0.25)
        Ai = faktor * (np.sin(ksi - np.pi / 4.) * Qclen + np.cos(ksi - np.pi / 4.) * Pclen)
        Bi = faktor * (-np.sin(ksi - np.pi / 4.) * Pclen + np.cos(ksi - np.pi / 4.) * Qclen)
        while (np.abs(Ai - Ai_pred) >= epsilon or np.abs(Bi - Bi_pred) >= epsilon):
            Ai_pred = Ai
            Bi_pred = Bi
            # asimptotski razvoj za velike |x|
            # Gamma(s + 1/2) = (2n)!*(pi^0.5)/((4^n)*n!)
            Pclen = Pclen * (-1.) * (6. * (j - 1.) + 11. / 2.) * (6. * (j - 1.) + 9. / 2.) * (6. * (j - 1.) + 7. / 2.) * (6. * (j - 1.) + 5. / 2.) * (6. * (j - 1.) + 3. / 2.) * (6. * (j - 1.) + 1. / 2.) / ((54. ** 2. * 2. * j * (2. * j - 1.) * (2. * (j - 1.) + 3. / 2.) *(2. * (j - 1.) + 1. / 2.) * ksi ** 2.))
            Qclen = Qclen * (-1.) * (6. * (j - 1.) + 17. / 2.) * (6. * (j - 1.) + 15. / 2.) * (6. * (j - 1.) + 13. / 2.) * (6. * (j - 1.) + 11. / 2.) * (6. * (j - 1.) + 9. / 2.) * (6. * (j - 1.) + 7. / 2.) / ((54. ** 2. * (2. * j + 1.) * 2. * j * (2. * (j - 1.) + 5. / 2.) *(2. * (j - 1.) + 3. / 2.) * ksi ** 2.))

            Ai = Ai + faktor * (np.sin(ksi - np.pi / 4.) * Qclen + np.cos(ksi - np.pi / 4.) * Pclen)
            Bi = Bi + faktor * (-np.sin(ksi - np.pi / 4.) * Pclen + np.cos(ksi - np.pi / 4.) * Qclen)
            j = j + 1.
    return Ai, Bi


#izpis vrednosti funkcij
x0 = np.linspace(-12., 1., 20000)
xAi_list=np.linspace(-12., 1.,20000)
xBi_list=np.linspace(-12., 1.,20000)
yAi_list = np.zeros(20000)
yBi_list = np.zeros(20000)
yAiM_list = np.zeros(20000)
yBiM_list = np.zeros(20000)
yAiA_list = np.zeros(20000)
yBiA_list = np.zeros(20000)

start1 = time.time()
for t in range(20000):
    yAiM_list[t] = MacLaurinovaVrsta(x0[t])[0]
    yBiM_list[t] = MacLaurinovaVrsta(x0[t])[1]
    yAiA_list[t] = AsimptotskaVrsta(x0[t])[0]
    yBiA_list[t] = AsimptotskaVrsta(x0[t])[1]
    yAi_list[t]=airyai(x0[t])
    yBi_list[t]=airybi(x0[t])

end1 = time.time()
print(end1 - start1)
#Razlika med funkcijo in približkom
errorAiM_list=[]
errorBiM_list=[]
errorAiA_list=[]
errorBiA_list=[]
maxAi_A=0
indAi_A=0
indBi_A=0
maxBi_A=0
maxAi_M=0
indAi_M=0
indBi_M=0
maxBi_M=0
for i in range(len(yAiM_list)):
    errorAiM_list.append(yAi_list[i]-yAiM_list[i])
    errorBiM_list.append(yBi_list[i]-yBiM_list[i])
    if abs(errorAiM_list[i])>abs(maxAi_M):
        maxAi_M=errorAiM_list[i]
        indAi_M=yAiM_list[i]
    if abs(errorBiM_list[i])>abs(maxBi_M):
        maxBi_M=errorBiM_list[i]
        indBi_M=yBiM_list[i]
        
for i in range(len(yAiA_list)):
    errorAiA_list.append(abs(yAi_list[i]-yAiA_list[i]))
    errorBiA_list.append(abs(yBi_list[i]-yBiA_list[i]))
    if abs(errorAiA_list[i])>abs(maxAi_A):
        maxAi_A=errorAiA_list[i]
        indAi_A=x0[i]
    if abs(errorBiA_list[i])>abs(maxBi_A):
        maxBi_A=errorBiA_list[i]
        indBi_A=x0[i]
#ZLEPEK FUNKCIJ Z USTREZNIM INTEVALOM
def Ai_funkcija(x):
    if -7.5 <= x < 8.58065:
        return MacLaurinovaVrsta(x)[0]
    else:
        return AsimptotskaVrsta(x)[0]
def Bi_funkcija(x):
    if -7.5 <= x < 8.20202020202:
        return MacLaurinovaVrsta(x)[1]
    else:
        return AsimptotskaVrsta(x)[1]


Ai = np.zeros(20000)
Bi = np.zeros(20000)
start2 = time.time()
for t in range(len(yAiM_list)):
    Ai[t]=Ai_funkcija(x0[t])
    Bi[t]=Bi_funkcija(x0[t])

end2 = time.time()
print(end2 - start2)
#NAPAKA ZLEPKA Z USTREZNIMI INTERVALI
errorAi_list=[]
errorBi_list=[]
for i in range(len(yAi_list)):
    errorAi_list.append(abs(yAi_list[i]-Ai[i]))
    errorBi_list.append(abs(yBi_list[i]-Bi[i]))
    
#Logaritmični zapis napake funkcij
errorAiM_loglist=[]
errorBiM_loglist=[]
errorAiA_loglist=[]
errorBiA_loglist=[]
errorAi_loglist=[]
errorBi_loglist=[]
for i in range(len(yAi_list)):
    errorAiM_loglist.append(np.log(errorAiM_list[i]))
    errorAiA_loglist.append(np.log(errorAiA_list[i]))
    errorAi_loglist.append(np.log(errorAi_list[i]))
    errorBiA_loglist.append(np.log(errorBiA_list[i]))
    errorBiM_loglist.append(np.log(errorBiM_list[i]))
    errorBi_loglist.append(np.log(errorBi_list[i]))

#RELATIVNA NAPAKA PRIBLIŽKOV:
relAiM_list=[]
relAiA_list=[]
relBiM_list=[]
relBiA_list=[]
relAi_list=[]
relBi_list=[]
for i in range(len(yAi_list)):
    relAiM_list.append(np.log(abs((yAi_list[i]-yAiM_list[i])/yAi_list[i])))
    relBiM_list.append(np.log(abs((yBi_list[i]-yBiM_list[i])/yBi_list[i])))
    relAiA_list.append(np.log(abs((yAi_list[i]-yAiA_list[i])/yAi_list[i])))
    relBiA_list.append(np.log(abs((yBi_list[i]-yBiA_list[i])/yBi_list[i])))
    relAi_list.append(np.log(abs((yAi_list[i]-Ai[i])/yAi_list[i])))
    relBi_list.append(np.log(abs((yBi_list[i]-Bi[i])/yBi_list[i])))

maksimum=0

    
#PLOT GRAFOV FUNKCIJ
plt.plot(xAi_list,yAiM_list, '-', label='$Ai(x) MacLauronova$')
plt.plot(xBi_list, yBiM_list, '-', label='$Bi(x)MacLauronova$')
#plt.plot(xAi_list,yAi_list, '--', label='$Ai(x) denajska$')
#plt.plot(xBi_list, yBi_list, '--', label='$Bi(x) dejanska$')
#plt.plot(xBi_list, yBiA_list, '--', label='$Bi(x) Asimptotska$')
plt.title('Graf Ariyjevih funkcij ')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.legend(loc='upper left')
plt.savefig('Graf Ariyjevih funkcij [-12,12].png')
plt.show()

#PLOT NAPAKE MACLAURONOVE VRSTE (Absolutna napaka)
plt.plot(xAi_list,errorAiM_list, '-', label='$ERROR od AiM(x)$')
plt.plot(xBi_list,errorBiM_list, '--', label='$ERROR od BiM(x)$')
plt.title('Napaka MacLaurinove vrste')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.legend(loc='upper left')
plt.savefig('Napaka MacLaurinove vrste [-12,12].png')
plt.show()
#PLOT NAPAKE MACLAURONOVE VRSTE (Absolutna napaka v logaritemski skali)
plt.plot(xAi_list,errorAiM_loglist, '-', label='$ERROR od AiM(x)$')
plt.plot(xBi_list,errorBiM_loglist, '--', label='$ERROR od BiM(x)$')
plt.title('Napaka MacLaurinove vrste v logaritemski skali')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.legend(loc='upper left')
plt.savefig('Napaka MacLaurinove vrste v logaritemski skali [-12,12].png')
plt.show()

#PLOT RELATIVNE NAPAKE MACLAURONOVE VRSTE (Absolutna napaka v logaritemski skali
plt.plot(xAi_list,relAiM_list, '-', label='$RELATIVE ERROR od AiA(x)$')
plt.plot(xBi_list,relBiM_list, '--', label='$RELATIVE ERROR od BiA(x)$')
plt.title('Relativna Napaka MACLAURONOVE vrste v logaritemski skali')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.legend(loc='upper left')
plt.savefig('Relativna Napaka MACLAURONOVE vrste v logaritemski skali [-12,12].png') 
plt.show()

#PLOT NAPAKE ASIMTOTSKE VRSTE (Absolutna napaka)
plt.plot(xAi_list,errorAiA_list, '-', label='$ERROR od AiA(x)$')
plt.plot(xBi_list,errorBiA_list, '--', label='$ERROR od BiA(x)$')
plt.title('Napaka Asimptotske vrste')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.legend(loc='upper left')
plt.savefig('Napaka Asimptotske vrste [-12,12].png') 
plt.show()
#PLOT NAPAKE ASIMTOTSKE VRSTE (Absolutna napaka v logaritemski skali
plt.plot(xAi_list,errorAiA_loglist, '-', label='$ERROR od AiA(x)$')
plt.plot(xBi_list,errorBiA_loglist, '--', label='$ERROR od BiA(x)$')
plt.title('Napaka Asimptotske vrste v logaritemski skali')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.legend(loc='upper left')
plt.savefig('Napaka Asimptotske vrste v logaritemski skali [-12,12].png') 
plt.show()


#PLOT RELATIVNE NAPAKE ASIMTOTSKE VRSTE (Absolutna napaka v logaritemski skali
plt.plot(xAi_list,relAiA_list, '-', label='$RELATIVE ERROR od AiA(x)$')
plt.plot(xBi_list,relBiA_list, '--', label='$RELATIVE ERROR od BiA(x)$')
plt.title('Relativna Napaka Asimptotske vrste v logaritemski skali')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.legend(loc='upper left')
plt.savefig('Relativna Napaka Asimptotske vrste v logaritemski skali [-12,12].png') 
plt.show()

#PLOT RELATIVNE NAPAKE ASIMTOTSKE VRSTE (Absolutna napaka v logaritemski skali
plt.plot(xAi_list,relAi_list, '-', label='$RELATIVE ERROR od Ai(x)$')
plt.plot(xBi_list,relBi_list, '--', label='$RELATIVE ERROR od Bi(x)$')
plt.title('Relativna Napaka zlepka vrste v logaritemski skali')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.legend(loc='upper left')
plt.savefig('Relativna Napaka zlepka vrste v logaritemski skali [-12,12].png') 
plt.show()

#PLOT NAPAKE AI VRSTE (Absolutna napaka v logaritemski skali)
plt.plot(xAi_list,errorAiM_loglist, '-', label='$ERROR od AiM(x)$')
plt.plot(xBi_list,errorAiA_loglist, '--', label='$ERROR od AiA(x)$')
plt.title('Napaka vrste v logaritemski skali za Ai')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.legend(loc='upper left')
plt.savefig('Napaka vrste v logaritemski skali za Ai [-12,12].png')
plt.show()
#PLOT NAPAKE BI VRSTE (Absolutna napaka v logaritemski skali)
plt.plot(xAi_list,errorBiM_loglist, '-', label='$ERROR od BiM(x)$')
plt.plot(xBi_list,errorBiA_loglist, '--', label='$ERROR od BiA(x)$')
plt.title('Napaka vrste v logaritemski skali za Bi')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.legend(loc='upper left')
plt.savefig('Napaka vrste v logaritemski skali za Bi [-12,12].png')
plt.show()
#PLOT ZLEPEK FUNKCIJ Z USTREZNIM INTEVALOM  
plt.plot(xAi_list,Ai, '-', label='$Ai(x)$')
plt.plot(xBi_list,Bi, '--', label='$Bi(x)$')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.title('Zlepek funkcij MacLau and Asimptotske')
plt.legend(loc='upper left')
plt.savefig('Zlepek funkcij MacL and Asimp [-12,12].png') 
plt.show()
 

#PLOT NAPAKE ZLEPKA Z USTREZNIM INTERVALOM
plt.plot(xAi_list,errorAi_list, '-', label='$ERROR od zlepka Ai(x)$')
plt.plot(xBi_list,errorBi_list, '--', label='$ERROR od zlepka Bi(x)$')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.title('Napaka zlepka funkcij vrste')
plt.legend(loc='upper left')
plt.savefig('Napaka zlepka funkcij vrste [-12,12].png')
plt.show()

#PLOT NAPAKE ZLEPKA Z USTREZNIM INTERVALOM (Absolutna napaka v logaritemski skali)
plt.plot(xAi_list,errorAi_loglist, '-', label='$ERROR od zlepka Ai(x)$')
plt.plot(xBi_list,errorBi_loglist, '--', label='$ERROR od zlepka Bi(x)$')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.title('Napaka zlepka funkcij vrste v logaritemski skali')
plt.legend(loc='upper left')
plt.savefig('Napaka zlepka funkcij vrste v logaritemski skali [-12,12].png')
plt.show()
