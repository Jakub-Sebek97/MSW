import numpy as np
import matplotlib.pyplot as plt
from time import time

def f(x):
    return 2*x**4 - x**3 - 3*x**2 - x + 2

def g(x):
    return x**x - np.exp(x) + x**2 - 5

def h(x):
    return np.e*np.cos((1/5)*x)*(1/2)*x - 3

def Derivace(x, h=1e-10):
    return (f(x + h) - f(x - h))/(2*h)

def Bisekce(f, a, b, eps):
    """Výpočet časové náročnosti nalezení kořene zadané funkce pomocí metody bisekce.
       Parametry: f: funkce,
                  a: levý odhad,
                  b: pravý odhad,
                  eps: požadovaná přesnost
    """
    t_0 = time()
    while (b - a) > 2*eps:
        x_i = (a + b)/2
        if f(a)*f(x_i) < 0:
            b = x_i
        else:
            a = x_i
    t_1 = time()
    
    return t_1 - t_0

def Metoda_tecen(f, a, b, eps):
    """Výpočet časové náročnosti nalezení kořene zadané funkce pomocí metody tečen.
       Parametry: f: funkce,
                  a: levý odhad,
                  b: pravý odhad,
                  eps: požadovaná přesnost
    """
    t_0 = time()
    x_i1 = (a + b)/2
    x_i = a

    while abs(x_i1 - x_i) > eps:
        x_i = x_i1
        x_i1 = x_i - f(x_i)/Derivace(x_i)
    t_1 = time()
    
    return t_1 - t_0

def Mereni_casu(eps=14, opak=1_000):
    """Měření průměrného času hledání kořene rovnic pomocí metody bisekce a metody tečen.
       Výstup: matice časů pro jednotlivé funkce
       Parametry: eps: požadovaná přesnost ve tvaru (1/10)^eps,
                  opak: počet opakování
    """
    casy = np.zeros((6, eps))
    for i in range(eps):
        for _ in range(opak):
            casy[0, i] += Bisekce(f, -10, 5, 10**(-(i+1)))
            casy[1, i] += Bisekce(g, 0, 3, 10**(-(i+1)))
            casy[2, i] += Bisekce(h, 0, 5, 10**(-(i+1)))
            casy[3, i] += Metoda_tecen(f, -10, 5, 10**(-(i+1)))
            casy[4, i] += Metoda_tecen(g, 0, 3, 10**(-(i+1)))
            casy[5, i] += Metoda_tecen(h, 0, 5, 10**(-(i+1)))
    casy /= opak

    return casy

osa_x = range(1, 15)
tloustka_liny = 3

casy = Mereni_casu()

plt.suptitle("Závislost rychlosti nalezení kořene na přesnosti kořene pomocí metody tečen a metody bisekce")

plt.subplot(1, 3, 1)
plt.title("Polynomická funkce f(x)")
plt.plot(osa_x, casy[0], linewidth=tloustka_liny, color="forestgreen", marker="o", label="Bisekce")
plt.plot(osa_x, casy[3], linewidth=tloustka_liny, color="mediumslateblue", marker="o", label="Metoda tečen")
plt.xlabel("Přesnost kořene v mocninách 1/10 [-]")
plt.ylabel("Průměrný čas nalezení kořene [s]")
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.title("Exponenciální funkce g(x)")
plt.plot(osa_x, casy[1], linewidth = tloustka_liny, color="forestgreen", marker="o", label="Bisekce")
plt.plot(osa_x, casy[4], linewidth = tloustka_liny, color="mediumslateblue", marker="o", label="Metoda tečen")
plt.xlabel("Přesnost kořene v mocninách 1/10 [-]")
plt.ylabel("Průměrný čas nalezení kořene [s]")
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.title("Harmonická funkce h(x)")
plt.plot(osa_x, casy[2], linewidth = tloustka_liny, color="forestgreen", marker="o", label="Bisekce")
plt.plot(osa_x, casy[5], linewidth = tloustka_liny, color="mediumslateblue", marker="o", label="Metoda tečen")
plt.xlabel("Přesnost kořene v mocninách 1/10 [-]")
plt.ylabel("Průměrný čas nalezení kořene [s]")
plt.grid(True)
plt.legend()

plt.show()