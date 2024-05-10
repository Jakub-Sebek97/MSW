import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, romberg, simpson, quad

def f(x):
    return 64*(x**7) - 112*(x**5) + 56*(x**3) - 7*x

def g(x):
    return np.exp(-x) - np.exp(x) + x**(np.e) + (1/2)*(x**2)

def h(x):
    return np.cos(x) + (1/2)*np.sin(2*x) - (1/3)*np.cos(3*x)

def Integrace_zakladni(f, a, b, dx=1e-7):
    """Výpočet určitého integrálu pomocí lichoběžníků zadané funkce v mezích od a do b.
       Parametry: f: funkce,
                  a: levá mez,
                  b: pravá mez,
                  dx: diferenciál
    """
    x = np.arange(a, b, dx)
    f_x, f_xdx = f(x), f(x + dx)
    integral = np.sum((dx*(f_x + f_xdx))/2)

    return integral

def Integrace_Newton_Cotes(f, a, b, dx=1e-7):
    """Výpočet určitého integrálu pomocí Newton-Cotesovo vzorce zadané funkce v mezích od a do b.
       Parametry: f: funkce,
                  a: levá mez,
                  b: pravá mez,
                  dx: diferenciál
    """
    x = np.arange(a, b, dx)
    f_x = f(x)
    integral = f(a) + f(b) + 2*np.sum(f_x[1:])
    integral *= dx/2
    return integral

x = [np.arange(-1, 2 + 1e-7, 1e-7), np.arange(0, 4 + 1e-7, 1e-7), np.arange(-1, 3 + 1e-7, 1e-7)]
y = [f(x[0]), g(x[1]), h(x[2])]

np_trapezoid = [np.trapz(y=y[0], x=x[0]),
                np.trapz(y=y[1], x=x[1]),
                np.trapz(y=y[2], x=x[2])]
sc_trapezoid = [trapezoid(y=y[0], x=x[0]),
                trapezoid(y=y[1], x=x[1]),
                trapezoid(y=y[2], x=x[2])]
sc_simpson = [simpson(y=y[0], x=x[0]),
              simpson(y=y[1], x=x[1]),
              simpson(y=y[2], x=x[2])]
sc_romberg = [romberg(f, -1, 2),
              romberg(g, 0, 4),
              romberg(h, -1, 3)]
sc_quad = [quad(f, -1, 2)[0],
           quad(g, 0, 4)[0],
           quad(h, -1, 3)[0]]

integrace_zakladni = [Integrace_zakladni(f, -1, 2),
                      Integrace_zakladni(g, 0, 4),
                      Integrace_zakladni(h, -1, 3)]
integrace_N_C = [Integrace_Newton_Cotes(f, -1, 2),
                 Integrace_Newton_Cotes(g, 0, 4),
                 Integrace_Newton_Cotes(h, -1, 3)]

reseni = [8*(2**8) - (56/3)*(2**6) + 14*(2**4) - (7/2)*(2**2) - 8*(-1)**8 + (56/3)*(-1)**6 - 14*(-1)**4 + (7/2)*(-1)**2,
          2 - np.exp(-4) - np.exp(4) + (4**(np.e + 1))/(np.e + 1) + ((1/6)*(4**3)),
          np.sin(3) - (1/4)*np.cos(6) - (1/9)*np.sin(9) - np.sin(-1) + (1/4)*np.cos(-2) + (1/9)*np.sin(-3)]

osa_x = ["NumPy\nTrapezoid", "SciPy\nTrapezoid", "SciPy\nSimpson", "SciPy\nRomberg", "SciPy\nQuad", "Základní\nintegrace", "Newton\nCotes\nintegrace"]
osa_y_fx = np.array([np_trapezoid[0], sc_trapezoid[0], sc_simpson[0], sc_romberg[0], sc_quad[0], integrace_zakladni[0], integrace_N_C[0]]) - reseni[0]
osa_y_gx = np.array([np_trapezoid[1], sc_trapezoid[1], sc_simpson[1], sc_romberg[1], sc_quad[1], integrace_zakladni[1], integrace_N_C[1]]) - reseni[1]
osa_y_hx = np.array([np_trapezoid[2], sc_trapezoid[2], sc_simpson[2], sc_romberg[2], sc_quad[2], integrace_zakladni[2], integrace_N_C[2]]) - reseni[2]

plt.suptitle("Porovnání odchylky řešení integrace funkcí f(x), g(x) a h(x)\nvůči analytickému řešení pomocí různých metod s přesností 1e-7")

plt.subplot(1, 3, 1)
plt.title("Odchylka funkce f(x)")
plt.bar(osa_x, np.abs(osa_y_fx), color="lightcoral", edgecolor="black")
plt.xticks(fontsize=8)
plt.xlabel("Metoda výpočtu")
plt.ylabel("Absolutní odchylka [-]")

plt.subplot(1, 3, 2)
plt.title("Odchylka funkce g(x)")
plt.bar(osa_x, np.abs(osa_y_gx), color="forestgreen", edgecolor="black")
plt.xticks(fontsize=8)
plt.xlabel("Metoda výpočtu")
plt.ylabel("Absolutní odchylka [-]")

plt.subplot(1, 3, 3)
plt.title("Odchylka funkce h(x)")
plt.bar(osa_x, np.abs(osa_y_hx), color="mediumslateblue", edgecolor="black")
plt.xticks(fontsize=8)
plt.xlabel("Metoda výpočtu")
plt.ylabel("Absolutní odchylka [-]")

plt.show()