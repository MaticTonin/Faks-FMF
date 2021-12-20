#optimizacija hitrosti
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt


def v(t,v_0):
    l=1
    t_0=1
    lamda=6/(t_0**3)*(l-v_0*t_0)

    return -lamda/4*t**2+lamda/2*t_0*t+v_0

x=np.arange(0,1,0.01)
y=v(x,0.7)
y2=v(x,2)


def dolzina(x,zacetna_hit):
    N=len(x)-1
    return sum(x)-x[-1]/2-zacetna_hit/2-N




def funkcija(tocke_hitrost,zacet_speed):

    n=len(tocke_hitrost)
    dt=1/(n-1)
    vsota1=0

    #zacetno
    vsota1+=0.5*((tocke_hitrost[0]-zacet_speed))**2

    for i in range(n-1):
        vsota1+=((tocke_hitrost[i+1]-tocke_hitrost[i]))**2
    vsota1+=0.5*((tocke_hitrost[i]-tocke_hitrost[i-1]))**2

    return vsota1

def optimizacija(zacetna_hitrost,N):
    hitrost=np.ones(N)*zacetna_hitrost
    opt=optimize.minimize(funkcija,hitrost,args=zacetna_hitrost,constraints=({'type': 'eq', 'fun': lambda x: sum(x)-x[-1]/2+zacetna_hitrost/2-N})).x
    return np.append([zacetna_hitrost],opt)

N=10
x=np.arange(0,1,0.1)
res=optimizacija(2,N)[:N]
y2=v(x,2)
napaka_2=np.log(abs(res-y2))
res2=optimizacija(3,N)[:N]
y3=v(x,3)
napaka_1=np.log(abs(res2-y3))

res10=optimizacija(10,N)[:N]
y10=v(x,10)
napaka_10=np.log(abs(res10-y10))


res3=optimizacija(0.5,N)[:N]
y=v(x,0.5)
napaka_3=np.log(abs(res3-y))


plt.title("Odvisnosti napake od izbire začetne hitrosti pri N=100")
plt.plot(x,napaka_2,'-',label=r'$v(0)=2$')
plt.plot(x,napaka_1,'-',label=r'$v(0)=3$')
plt.plot(x,napaka_3,'-',label=r'$v(0)=0.5$')
plt.plot(x,napaka_10,'-',label=r'$v(0)=10$')
plt.ylabel(r"log($v_{num}-v_{ana})$")
plt.legend()
plt.xlabel('čas')
plt.show()
# plt.plot(res2)
# plt.plot(res3)
plt.plot(x,y3,'-',label=r'Analitična $v(0)=1$')
plt.plot(x,res2[:N],'x-',label=r'Numerična $v(0)=1$')
plt.plot(x,y,'-',label=r'Analitična $v(0)=0.5$')
plt.plot(x,res3[:N],'x-',label=r'Numerična $v(0)=0.5$')
plt.legend()
plt.title('Primerjava rešitev za različne začetne hitrosti pri N='+str(N))
plt.ylabel('hitrost')
plt.xlabel('čas')
plt.show()
naslov='primerjava_anal_num_n_'+str(N)+'.png'
plt.plot(x,y2,'-',label=r'Analitična $v(0)=2$')
plt.plot(x,res[:N],'x-',label=r'Numerična $v(0)=2$')
# plt.plot(res2)
# plt.plot(res3)
plt.plot(x,y3,'-',label=r'Analitična $v(0)=1$')
plt.plot(x,res2[:N],'x-',label=r'Numerična $v(0)=1$')
plt.plot(x,y,'-',label=r'Analitična $v(0)=0.5$')
plt.plot(x,res3[:N],'x-',label=r'Numerična $v(0)=0.5$')
plt.legend()
plt.title('Primerjava rešitev za različne začetne hitrosti pri N='+str(N))
plt.ylabel('hitrost')
plt.xlabel('čas')
plt.savefig(naslov)
plt.show()
n=[5,10,20,40,60]
x=np.arange(0,1,0.01)
y2=v(x,2)
y4=v(x,8)
y6=v(x,16)
plt.subplot(1,2,1)
plt.plot(x,y2,'-',label=r'Analitična $v(0)=2$')
plt.plot(x,y4,'-',label=r'Analitična $v(0)=8$')
plt.plot(x,y6,'-',label=r'Analitična $v(0)=16$')
color=["r","g","b","orange","gray", "purple"]
index=0
for i in n:
    plt.title(r'Prikaz konvergence rešitev k analitični vrednosti, $v_0>1$')
    x=np.arange(0,1,1/i)
    res2=optimizacija(2,i)
    y2=v(x,2)
    res4=optimizacija(8,i)
    y4=v(x,8)
    res6=optimizacija(16,i)
    y6=v(x,16)
    plt.plot(x,res2[:i],'.',color=color[index])
    plt.plot(x,res4[:i],'.',color=color[index])
    plt.plot(x,res6[:i],'.',color=color[index],label=r'N='+str(i))
    plt.legend(loc='upper right')
    index+=1

plt.subplot(1,2,2)
y2=v(x,0.02)
y4=v(x,0.4)
y6=v(x,0.96)
plt.plot(x,y2,'-',label=r'Analitična $v(0)=0.02$')
plt.plot(x,y4,'-',label=r'Analitična $v(0)=0.4$')
plt.plot(x,y6,'-',label=r'Analitična $v(0)=0.96$')
plt.title(r'Prikaz konvergence rešitev k analitični vrednosti, $v_0<1$')
index=0
for i in n:
    x=np.arange(0,1,1/i)
    res2=optimizacija(0.02,i)
    y2=v(x,0.02)
    res4=optimizacija(0.4,i)
    y4=v(x,0.4)
    res6=optimizacija(0.96,i)
    y6=v(x,0.96)
    plt.plot(x,res2[:i],'.',color=color[index])
    plt.plot(x,res4[:i],'.',color=color[index])
    plt.plot(x,res6[:i],'.',color=color[index],label=r'N='+str(i))
    plt.ylabel('Hitrost')
    plt.legend(loc='lower right')
    plt.xlabel('Čas')
    index+=1
plt.show()