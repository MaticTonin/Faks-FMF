import numpy as np
import matplotlib.pyplot as plt
from scipy import special

#asimptotski rayvoj
def funkcija(x):
    return ((x**(2./3.))*(1.+(5./48.)*(x**(-2.))-(5./36.)*(x**(-4.))+(77125./82944.)*(x**(-6.))-(108056875./6967296.)*(x**(-8.))))

#referenca
aizeroref = list((special.ai_zeros(100))[0])
bizeroref = list((special.bi_zeros(100))[0])

aizeroseries = np.zeros(100)
bizeroseries = np.zeros(100)

for t in range(100):
    aizeroseries[t] = (-1.)*funkcija((3.*np.pi*(4.*(t+1.)-1.)/8.))
    bizeroseries[t] = (-1.)*funkcija((3.*np.pi*(4.*(t+1.)-3.)/8.))

#abs napake
aizeroerror1 = np.log(np.abs(aizeroref - aizeroseries))
bizeroerror1 = np.log(np.abs(bizeroref - bizeroseries))
aizeroerror1 = np.log(np.abs(aizeroref - aizeroseries))
bizeroerror1 = np.log(np.abs(bizeroref - bizeroseries))

#rel napake
aizeroerror2 = np.log(np.abs((aizeroref - aizeroseries)/aizeroref))
bizeroerror2 = np.log(np.abs((bizeroref - bizeroseries)/bizeroref))
aizeroerror2 = np.log(np.abs((aizeroref - aizeroseries)/aizeroref))
bizeroerror2 = np.log(np.abs((bizeroref - bizeroseries)/bizeroref))


x0 = np.linspace(0., 100., 100)
#Plot odstopanja ničel funkij od približka
plt.plot(x0,aizeroerror1, '.', label='Odstopanje ničel za Ai(x)')
plt.plot(x0, bizeroerror1, '.', label='Odstopanje ničel za Bi(x)')
#plt.plot(xAi_list,yAi_list, '--', label='$Ai(x) denajska$')
#plt.plot(xBi_list, yBi_list, '--', label='$Bi(x) dejanska$')
#plt.plot(xBi_list, yBiA_list, '--', label='$Bi(x) Asimptotska$')
plt.title('Graf odstopanja ničel za Airyjeve funkcije ')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.legend(loc='upper left')
plt.savefig('Graf odstopanja ničel za Airyjeve funkcije.png')
plt.show()

#Plot relativnega odstopanja ničel funkij od približka
plt.plot(x0,aizeroerror2, '.', label='Relativno dstopanje ničel za Ai(x)')
plt.plot(x0, bizeroerror2, '.', label='Relativno odstopanje ničel za Bi(x)')
#plt.plot(xAi_list,yAi_list, '--', label='$Ai(x) denajska$')
#plt.plot(xBi_list, yBi_list, '--', label='$Bi(x) dejanska$')
#plt.plot(xBi_list, yBiA_list, '--', label='$Bi(x) Asimptotska$')
plt.title('Graf relativnega odstopanja ničel za Airyjeve funkcije ')
plt.axhline(y=0,color='0', linewidth=0.5, linestyle=':')
plt.axvline(x=0, color='0', linewidth=0.5, linestyle=':')
plt.legend(loc='upper left')
plt.savefig('Graf relativnega odstopanja ničel za Airyjeve funkcije.png')
plt.show()
