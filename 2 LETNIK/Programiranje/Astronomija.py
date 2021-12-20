import math
def menjavanje(vrstica):
    for i in vrstica:
        if i == "|" or i== "-":
            i=" "
    return i

f=open("astronomija.txt", "w")
for vrstica in open("Podatki"):
        print(menjavanje(vrstica.strip()), file=f)
print("Končano")