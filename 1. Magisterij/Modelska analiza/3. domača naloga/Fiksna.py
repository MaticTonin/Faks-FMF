#optimizacija hitrosti
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt


def v(t,v_0):
    l=1
    t_0=1
    lamda=6/(t_0**3)*(l-v_0*t_0)

    return -lamda/4*t**2+lamda/2*t_0*t+v_0

x=np.arange(0,1.1,0.01)
y=v(x,0.7)
y2=v(x,2)
x=x*100

def dolzina(x,zacetna_hit,konc):
    N=len(x)-1
    return sum(x)-zacetna_hit/2-konc/2-N




def funkcija(tocke_hitrost,zacet_speed,koncna):

    n=len(tocke_hitrost)
    vsota1=0

    #zacetno
    vsota1+=0.5*((tocke_hitrost[0]-zacet_speed))**2

    for i in range(n-1):
        vsota1+=((tocke_hitrost[i+1]-tocke_hitrost[i]))**2
    vsota1+=0.5*((koncna-tocke_hitrost[-1]))**2

    return vsota1

def optimizacija(zacetna_hitrost,konec,N):
    hitrost=np.ones(N-1)*zacetna_hitrost
    opt=optimize.minimize(funkcija,hitrost,args=(zacetna_hitrost,konec),constraints=({'type': 'eq', 'fun': lambda x: sum(x)+konec/2+zacetna_hitrost/2-N})).x
    return np.append([zacetna_hitrost],np.append(opt,[konec]))
"""
N=100
vk=0.5

res=optimizacija(2,vk,N)


res2=optimizacija(2.5,vk,N)


res3=optimizacija(0.7,vk,N)


res4=optimizacija(0.3,vk,N)

# print(res4)
# print(sum(res4))

# plt.plot(x,y,'-')
# plt.plot(x,y2,'-')
x=np.arange(0,1.01,0.01)

naslov='numericno_fiskna_koncna_n_'+str(N)+'vk_'+str(vk)+'.png'
plt.subplot(2,2,1)
plt.title(r'Minimizacija pri končni hitrosti $v=0.5$')
plt.plot(x,res3,label=r'$v(0)=0.7$')
plt.plot(x,res4,label=r'$v(0)=0.3$')
plt.plot(x,res,label=r'$v(0)=2$')
plt.plot(x,res2,label=r'$v(0)=2.5$')
plt.legend(loc=2)
plt.grid()
plt.ylabel('Hitrost')
plt.xlabel('Čas')

plt.subplot(2,2,2)
vk=1
res=optimizacija(2,vk,N)
res2=optimizacija(2.5,vk,N)
res3=optimizacija(0.7,vk,N)
res4=optimizacija(0.3,vk,N)

plt.plot(x,res3,label=r'$v(0)=0.7$')
plt.plot(x,res4,label=r'$v(0)=0.3$')
plt.plot(x,res,label=r'$v(0)=2$')
plt.plot(x,res2,label=r'$v(0)=2.5$')
plt.legend(loc=2)
plt.ylabel('Hitrost')
plt.xlabel('Čas')
plt.grid()
plt.title(r'Minimizacija pri končni hitrosti $v=1$')

plt.subplot(2,2,3)
vk=3.5
res=optimizacija(2,vk,N)
res2=optimizacija(2.5,vk,N)
res3=optimizacija(0.7,vk,N)
res4=optimizacija(0.3,vk,N)

plt.plot(x,res3,label=r'$v(0)=0.7$')
plt.plot(x,res4,label=r'$v(0)=0.3$')
plt.plot(x,res,label=r'$v(0)=2$')
plt.plot(x,res2,label=r'$v(0)=2.5$')
plt.legend(loc=2)
plt.ylabel('Hitrost')
plt.xlabel('Čas')
plt.grid()
plt.title(r'Minimizacija pri končni hitrosti $v=3.5$')
plt.savefig(naslov)


plt.subplot(2,2,4)
vk=6
res=optimizacija(2,vk,N)

res2=optimizacija(2.5,vk,N)

res3=optimizacija(0.7,vk,N)
res4=optimizacija(0.3,vk,N)

plt.plot(x,res3,label=r'$v(0)=0.7$')
plt.plot(x,res4,label=r'$v(0)=0.3$')
plt.plot(x,res,label=r'$v(0)=2$')
plt.plot(x,res2,label=r'$v(0)=2.5$')
plt.legend(loc=2)
plt.ylabel('Hitrost')
plt.xlabel('Čas')
plt.grid()
plt.title(r'Minimizacija pri končni hitrosti $v=6$')
plt.savefig(naslov)
plt.show()


n=[5,7,10,30,60,100]
x=np.arange(0,1,0.01)
y2=v(x,2)
y4=v(x,8)
y6=v(x,16)
plt.subplot(1,2,1)
color=["r","g","b","orange","yellow", "gray","black"]
index=0
for i in n:
    plt.title(r'Prikaz konvergence rešitev k analitični vrednosti, $v_0>1$, $v_k=6$')
    x=np.arange(0,1,1/i)
    res2=optimizacija(2,vk,i)
    y2=v(x,2)
    res4=optimizacija(8,vk,i)
    y4=v(x,8)
    res6=optimizacija(16,vk,i)
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
plt.title(r'Prikaz konvergence rešitev k analitični vrednosti, $v_0<1$, $v_k=6$')
index=0
for i in n:
    x=np.arange(0,1,1/i)
    res2=optimizacija(0.02,vk,i)
    y2=v(x,0.02)
    res4=optimizacija(0.4,vk,i)
    y4=v(x,0.4)
    res6=optimizacija(0.96,vk,i)
    y6=v(x,0.96)
    plt.plot(x,res2[:i],'.',color=color[index])
    plt.plot(x,res4[:i],'.',color=color[index])
    plt.plot(x,res6[:i],'.',color=color[index],label=r'N='+str(i))
    plt.ylabel('Hitrost')
    plt.legend(loc='lower right')
    plt.xlabel('Čas')
    index+=1
plt.show()"""

v0=1
n=[50,70,100,300,600,1000]
x=np.arange(0,1,0.01)
y2=v(x,2)
y4=v(x,8)
y6=v(x,16)
plt.subplot(1,2,1)
color=["r","g","b","orange","yellow", "gray","black"]
index=0
for i in n:
    plt.title(r'Prikaz konvergence rešitev k analitični vrednosti, $v_0=1$, $v_k>1$')
    x=np.arange(0,1,1/i)
    res2=optimizacija(v0,2,i)
    y2=v(x,v0)
    res4=optimizacija(v0,8,i)
    y4=v(x,v0)
    res6=optimizacija(v0,16,i)
    y6=v(x,v0)
    plt.plot(x,res2[:i],'.',color=color[index])
    plt.plot(x,res4[:i],'.',color=color[index])
    plt.plot(x,res6[:i],'.',color=color[index],label=r'N='+str(i))
    plt.legend(loc='upper left')
    index+=1

plt.subplot(1,2,2)
y2=v(x,0.02)
y4=v(x,0.4)
y6=v(x,0.96)
plt.title(r'Prikaz konvergence rešitev k analitični vrednosti, $v_0=1$, $v_k<1$')
index=0
for i in n:
    x=np.arange(0,1,1/i)
    res2=optimizacija(v0,0.02,i)
    y2=v(x,0.02)
    res4=optimizacija(v0,0.4,i)
    y4=v(x,0.4)
    res6=optimizacija(v0,0.96,i)
    y6=v(x,0.96)
    plt.plot(x,res2[:i],'.',color=color[index])
    plt.plot(x,res4[:i],'.',color=color[index])
    plt.plot(x,res6[:i],'.',color=color[index],label=r'N='+str(i))
    plt.ylabel('Hitrost')
    plt.legend(loc='lower left')
    plt.xlabel('Čas')
    index+=1
plt.show()