import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sindg, cosdg, tandg
from scipy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.misc import derivative
from scipy.integrate import quad
from random import randint, uniform
from time import time

np.random.seed(100)

def Skalarni_soucin(n=100_000):
    """Výpočet časové náročnosti skalárního součinu pomocí klasického Pythonu a knihoven NumPy a SciPy.
       Výstup: trojice časů
       Parametry: n: počet iterací
    """
    def Python(vektor_1, vektor_2):
        t_0 = time()
        skalar_soucin = sum(vektor_1[i]*vektor_2[i] for i in range(len(vektor_1)))
        t_1 = time()
        return t_1 - t_0
    def NumPy(vektor_1, vektor_2):
        t_0 = time()
        skalar_soucin = vektor_1 @ vektor_2
        t_1 = time()
        return t_1 - t_0
    def SciPy(vektor_1, vektor_2):
        t_0 = time()
        skalar_soucin = csr_matrix.dot(vektor_1, vektor_2)
        t_1 = time()
        return t_1 - t_0
    
    casy = np.zeros(3)
    for _ in range(n):
        u = np.array([randint(-100, 100) for __ in range(1_000)])
        v = np.array([randint(-100, 100) for __ in range(1_000)])
        casy[0] += Python(u, v)
        casy[1] += NumPy(u, v)
        casy[2] += SciPy(u, v)
    
    return casy

def Goniometricke_funkce(n=1_000_000):
    """Výpočet časové náročnosti goniometrických funkcí pomocí klasického Pythonu a knihoven NumPy a SciPy.
       Výstup: trojice časů
       Parametry: n: počet iterací
    """
    def Python(uhel):
        t_0 = time()
        sin, cos = math.sin(uhel), math.cos(uhel)
        tg, cotg = math.tan(uhel), 1 / math.tan(uhel)
        t_1 = time()
        return t_1 - t_0
    def NumPy(uhel):
        t_0 = time()
        sin, cos = np.sin(uhel), np.cos(uhel)
        tg, cotg = np.tan(uhel), 1 / np.tan(uhel)
        t_1 = time()
        return t_1 - t_0
    def SciPy(uhel):
        t_0 = time()
        sin, cos = sindg(uhel), cosdg(uhel)
        tg, cotg = tandg(uhel), 1 / tandg(uhel)
        t_1 = time()
        return t_1 - t_0
    
    casy = np.zeros(3)
    for _ in range(n):
        uhel_rad = uniform(1e-5, np.pi/2)
        uhel_deg = randint(1, 89)
        casy[0] += Python(uhel_rad)
        casy[1] += NumPy(uhel_rad)
        casy[2] += SciPy(uhel_deg)

    return casy

def Norma_vektoru(n=100_000):
    """Výpočet časové náročnosti normy vektoru pomocí klasického Pythonu a knihoven NumPy a SciPy.
       Výstup: trojice časů
       Parametry: n: počet iterací
    """
    def Python(vektor):
        t_0 = time()
        norma = math.sqrt(sum(vektor[i]**2 for i in range(len(vektor))))
        t_1 = time()
        return t_1 - t_0
    def NumPy(vektor):
        t_0 = time()
        norma = np.linalg.norm(vektor)
        t_1 = time()
        return t_1 - t_0
    def SciPy(vektor):
        t_0 = time()
        norma = norm(vektor)
        t_1 = time()
        return t_1 - t_0
    
    casy = np.zeros(3)
    for _ in range(n):
        u = np.array([randint(-100, 100) for __ in range(1_000)])
        casy[0] += Python(u)
        casy[1] += NumPy(u)
        casy[2] += SciPy(u)

    return casy

def Integrace(n=50):
    """Výpočet časové náročnosti integrace funkce pomocí klasického Pythonu a knihoven NumPy a SciPy.
       Výstup: trojice časů
       Parametry: n: počet iterací
    """
    def f(x):
        return 64*(x**7) - 112*(x**5) + 56*(x**3) - 7*x
    
    def Python(a, b, dx):
        t_0 = time()
        n = int((b - a)//dx) + 1
        integral = f(a) + f(b) + sum(2*f(a + i*dx) for i in range(1, n))
        integral *= dx/2
        t_1 = time()
        return t_1 - t_0
    def NumPy(x, y):
        t_0 = time()
        integral = np.trapz(y, x)
        t_1 = time()
        return t_1 - t_0
    def SciPy(a, b):
        t_0 = time()
        integral = quad(f, a, b)
        t_1 = time()
        return t_1 - t_0
    
    casy = np.zeros(3)
    dx = 1e-5
    for _ in range(n):
        a, b = randint(0, 10), randint(11, 20)
        x = np.arange(a, b, dx)
        casy[0] += Python(a, b, dx)
        casy[1] += NumPy(x, f(x))
        casy[2] += SciPy(a, b)

    return casy

def Derivace(n=500_000):
    """Výpočet časové náročnosti derivace funkce pomocí klasického Pythonu a knihoven NumPy a SciPy.
       Výstup: trojice časů
       Parametry: n: počet iterací
    """
    def f(x):
        return np.cos(x) + (1/2)*np.sin(2*x) - (1/3)*x*np.cos(3*x)
    
    def Python(x, h):
        t_0 = time()
        derivace = (f(x + h) - f(x)) / h
        t_1 = time()
        return t_1 - t_0
    def NumPy(a, h):
        x = np.arange(a - 0.001, a, h)
        t_0 = time()
        derivace = np.gradient(f(x), x)
        t_1 = time()
        return t_1 - t_0
    def SciPy(x, h):
        t_0 = time()
        derivace = derivative(f, x, h)
        t_1 = time()
        return t_1 - t_0
    
    casy = np.zeros(3)
    h = 1e-5
    for _ in range(n):
        x = randint(0, 100)
        casy[0] += Python(x, h)
        casy[1] += NumPy(x, h)
        casy[2] += SciPy(x, h)

    return casy

osa_x = ["Python", "NumPy", "SciPy"]

plt.suptitle("Časová náročnost matematických operací knihoven NumPy a SciPy a klasického Pythonu")

plt.subplot(2, 3, 1)
plt.title("Skalární součin")
plt.bar(osa_x, Skalarni_soucin(), color="lightcoral", edgecolor="black")
plt.xlabel("Knihovna")
plt.ylabel("Čas [s]")

plt.subplot(2, 3, 2)
plt.title("Norma vektoru")
plt.bar(osa_x, Norma_vektoru(), color="orange", edgecolor="black")
plt.xlabel("Knihovna")
plt.ylabel("Čas [s]")

plt.subplot(2, 3, 3)
plt.title("Goniometrické funkce")
plt.bar(osa_x, Goniometricke_funkce(), color="forestgreen", edgecolor="black")
plt.xlabel("Knihovna")
plt.ylabel("Čas [s]")

plt.subplot(2, 3, 4)
plt.title("Derivace")
plt.bar(osa_x, Derivace(), color="dodgerblue", edgecolor="black")
plt.xlabel("Knihovna")
plt.ylabel("Čas [s]")

plt.subplot(2, 3, 5)
plt.title("Integrace")
plt.bar(osa_x, Integrace(), color="mediumslateblue", edgecolor="black")
plt.xlabel("Knihovna")
plt.ylabel("Čas [s]")

plt.show()