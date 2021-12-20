# Uporabne stvari, ki sem jih spoznal do sedaj 
# Dodatek: Neznanke so določene tako, da program deluje kljub temu, da je množica zmešnjav



# Splošno
a=10

# 1. Za ostanek pri deljenju
b= a % 3
print("Ostanek "+ str(a) + " pri deljenju s 3 je "+ str(b) )

# 2. Celoštevilsko delitelj
c= a // 3
print("Celoštevilski delitelj števila "+ str(a) + " je "+ str(c) )


# Funkcije

# 1. Definiranje funkcije
def ime_funkcije(vrednost_ki_jo_v_funkciji_uporabim):
    # napišeš, kaj bi rad, da funkcija počne
    return vrednost_ki_jo_v_funkciji_uporabim # ali pa kar v return pišem, kaj naj dela

# 2. Pisanje matematičnih izrazov:
import math
Matematični_izraz = math.tan(math.pi) 
#tudi za matematične konstante uporabljamo math



# Nizi
# https://docs.python.org/2/library/string.html 
# 1. Pisanje nizov
#    Če naprimer navedemo neko vrednost, ki ni chr ampak je int, moramo pri pisanju niza reči:
"Imaš "+ str(Matematični_izraz)+ " limon"

# 2. Pisanje tabel (saj je podobno kot niz) in matrik
#    Za določanje tabele, zapišemo:
matrika=[]
#    Če želimo v  to tabelo dodajati elemente, zapišemo:
print("Dodajanje elemenov v matriko")
for i in range(10):
    matrika.append(i)
print(matrika) 
# Uporabil sem zanko, da je bolj očitno in na koncu sprintal matriko

# 3.Dolžina matrike 
n=len(matrika)
print("Dolzina matrike je enaka " + str(n))

# 4. Štetje znakov v nizu
stetje=matrika.count(5) # ker je 5 v bistvu kar niz
print("Število ponovitve znaka 5 v matriki je enaka " + str(stetje))

# 5. Rezanje nizov
i=5
od_0_do_i=matrika[i:]
od_i_do_konca=matrika[:i]
print("Izrezan niz od 0 do i (i:) izgleda takole "+ str(od_0_do_i))
print("Izrezan niz od i do konca (:i ) izgleda takole "+ str(od_i_do_konca))
# Več o tem: http://www.nauk.si/materials/5251/out/#state=5

# 6. Intigerji od določenih znakov
#    Kot prvo moramo vedeti, koliko je intiger določenega znaka, kar izvemo na spletni strani
#    http://www.asciitable.com/
# če želimo vedeti mesto določenega znaka
mesto_od_znaka=ord("A")
print("Mesto, na katerem se nahaja znak A je "+ str(mesto_od_znaka)) # Dobimo 65
# če želimo vedeti, kateremu znaku pripada neko mesto
znak_iz_mesta=chr(90)
print("Znak, ki se nahaja na mestu 90 je "+ str(znak_iz_mesta)) # Dobimo Z

# 7. Povečanje črk določenega niza iz malih v velike 
niz="testiranje"
kopija=niz.upper()
print("Kaj se zgodi, če z funkcijo niz.upper spremenimo niz ¨testiranje¨v nov niz: "+ kopija)

# 8. Iskanje nizov v nizih.
# Naprimer, da moraš najti, kje je določen samoglasnik v besedi, ne poženeš funkcije
# for i in range(len(niz)) ampak napišeš
samoglasniki="aeiou"
samoglasniki_v_besedi_testiranje=""
for i in niz:
    if i in samoglasniki:
        samoglasniki_v_besedi_testiranje+=i
print("Tako vemo, da so samoglasniki v besedi ¨testiranje¨: " + samoglasniki_v_besedi_testiranje)




# ta del pytona ne bo deloval zaradi datotek, ki jih vaš računalnik ne vsebuje


# Datoteke

# 1. Odpiranje datoteke
#  with open(ime_datoteke.txt,"w") as f: 
    # w se uporablja za branje, r za pisanje w+ za pisanje in shranjevanje

# 2. Kopiranje iz ene datoteke in printanje spremenjenega besedila v drugo datoteko 
# def disemvowel_datoteko(vhodna, izhodna):
    # with open(izhodna,"w") as f:
        # for vrstica in open(vhodna):
           # print(funkcija_ki jo_zelim_izvesti(vrstica.strip()), file=f)  
# vrstica.strip() pomeni, da vsaki vrsti zbriše ostanek vrste, akka presledke na koncu besedila.