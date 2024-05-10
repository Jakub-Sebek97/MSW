import sympy
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 128*x**8 - 256*x**6 + 160*x**4 - 32*x**2 + 1

def Derivace_analyticka(bod_x):
    """Výpočet derivace funkce v bodě x analytickou metodou.
       Parametry: bod_x: bod, ve kterém bude vypočtena derivace
    """
    x = sympy.symbols("x")
    derivace = sympy.diff(f(x))
    return derivace.subs(x, bod_x).evalf()

def Derivace_staticka(typ, x, h=1e-8):
    """Výpočet derivace funkce v bodě x se staticky nastaveným krokem.
       Parametry: typ: 1 - zpětná, 2 = centrální, 3 = dopředná,
                  x: bod, ve kterém bude vypočtena derivace
                  h: krok
    """
    if typ == 1:
        return (f(x) - f(x - h)) / h
    elif typ == 2:
        return (f(x + h) - f(x - h)) / (2*h)
    elif typ == 3:
        return (f(x + h) - f(x)) / h
    else:
        raise ValueError("Neplatná hodnota. Jsou povoleny pouze hodnoty typu 1, 2 nebo 3")

def Derivace_adaptivni(typ, x, n=2, h=0.1, eps=1e-8):
    """Výpočet derivace funkce v bodě x s dynamicky nastavitelným krokem
       Parametry: typ: typ derivace: 1 - zpětná, 2 = centrální, 3 = dopředná,
                  x: bod, ve kterém bude vypočtena derivace,
                  n: nastavení délky kroku,
                  h: výchozí krok pro výpočet derivace,
                  eps: požadovaná přesnost
    """
    if typ not in [1, 2, 3]:
        raise ValueError("Neplatná hodnota. Jsou povoleny pouze hodnoty typu 1, 2 nebo 3")

    h_n1 = h
    stop_podminka = 1
    derivace = Derivace_staticka(typ, x, h)

    while stop_podminka > eps:
        h_n1 /= n
        derivace_n1 = Derivace_staticka(typ, x, h_n1)
        stop_podminka = abs(derivace - derivace_n1)
        derivace = derivace_n1
    return derivace

osa_x = ["Zpětná", "Centrální", "Dopředná"]
adapt_analytik = np.array([Derivace_adaptivni(1, 0.5), 
                           Derivace_adaptivni(2, 0.5),
                           Derivace_adaptivni(3, 0.5)]) - Derivace_analyticka(0.5)
adapt_statik = np.array([Derivace_staticka(1, 0.5) - Derivace_adaptivni(1, 0.5),
                         Derivace_staticka(2, 0.5) - Derivace_adaptivni(2, 0.5),
                         Derivace_staticka(3, 0.5) - Derivace_adaptivni(3, 0.5)])

plt.suptitle("Porovnání řešení derivace s adaptivním krokem vůči analytickému řešení a derivaci se statickým krokem s přesností 1e-8")

plt.subplot(1, 2, 1)
plt.title("Absolutní odchylka vůči analytickému řešení")
plt.bar(osa_x, np.abs(adapt_analytik), color="forestgreen", edgecolor="black")
plt.xlabel("Derivace s adaptivním krokem")
plt.ylabel("Absolutní odchylka [-]")

plt.subplot(1, 2, 2)
plt.title("Absolutní odchylka vůči derivaci se statickým krokem")
plt.bar(osa_x, np.abs(adapt_statik), color="mediumslateblue", edgecolor="black")
plt.xlabel("Derivace s adaptivním krokem")
plt.ylabel("Absolutní odchylka [-]")

plt.show()