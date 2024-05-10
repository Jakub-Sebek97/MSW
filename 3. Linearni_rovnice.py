import numpy as np
import matplotlib.pyplot as plt
from time import time

np.random.seed(245)

def Metoda_prima(A, B):
    """Výpočet časové náročnosti řešení soustavy lineárních rovnic Gaussovo eliminační přímou metodou.
        parametry: A: čtvercová matice,
                   B: matice pravých stran
    """
    t_0 = time()
    hodnost = len(A)
    AB = np.column_stack((A,B))
    
    for i in range(hodnost):
        for j in range(i+1, hodnost):
            koeficient = AB[j, i] / AB[i, i]
            AB[j] -= koeficient*AB[i]

    X = np.zeros(hodnost)
    for j in range(hodnost-1, -1, -1):
        X[j] = (AB[j,-1] - (AB[j, :-1] @ X)) / AB[j,j]
    t_1 = time()

    return t_1 - t_0 

def Metoda_iteracni(A, B, i=10, X0=None):
    """Výpočet časové náročnosti řešení soustavy lineárních rovnic Gaussovo-Seidelovo iterační metodou.
        parametry: A: čtvercová matice,
                   B: matice pravých stran
    """
    t_0 = time()
    if X0 is None:
        X0 = np.ones(len(A))
    X = X0
    Horni, Dolni = np.triu(A, k=1), np.tril(A, k=0)
    Y, Z = -np.linalg.inv(Dolni) @ Horni, np.linalg.inv(Dolni) @ B
    
    for _ in range(i):
        X = (Y @ X) + Z
    t_1 = time()
    
    return t_1 - t_0

def Mereni_casu(n=200, opak=30):
    """Měření průměrného času řešení soustavy lineárních rovnic.
       Výstup: matice časů pro jednotlivé funkce
       Parametry: n: velikost matice typu [n x n],
                  opak: počet opakování
    """    
    casy = np.zeros((2, n))
  
    for i in range(1, n + 1):
        for _ in range(opak):
            A = np.random.rand(i, i) + 1
            B = np.random.rand(i, 1)
            casy[0, i - 1] += Metoda_prima(A, B)
            casy[1, i - 1] += Metoda_iteracni(A, B)
    casy /= opak

    return casy

osa_x = range(1, 201)
osa_y_prima, osa_y_iteracni = Mereni_casu()
tlouska_liny = 2

plt.plot(osa_x, osa_y_prima, linewidth=tlouska_liny, color="forestgreen", label="Přímá metoda")
plt.plot(osa_x, osa_y_iteracni, linewidth=tlouska_liny, color="mediumslateblue", label="Iterační metoda")
plt.title("Porovnání času potřebného k získání řešení soustavy lineárních rovnic pomocí přímé a iterační metody")
plt.xlabel("Matice typu [n x n]")
plt.ylabel("Čas [s]")
plt.axhline(0, color="grey") 
plt.grid(True)
plt.legend()

plt.show()