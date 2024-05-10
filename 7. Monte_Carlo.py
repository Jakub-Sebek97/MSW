import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from matplotlib.animation import FuncAnimation

np.random.seed(100)

def f(x):
  return np.sin(9*x + 2)*x**(1/2) + (x/10)*np.cos(5*x - 1)

def Monte_Carlo_extremy(a, b, i=1_000_000):
  """Nalezení minimálních a maximálních funkčních hodnot na intervalu od "a" do "b".
     parametry: a: levá mez
                b: pravá mez
                i: počet opakování
  """
  min_x, min_y = None, float("inf")
  max_x, max_y = None, float("-inf")

  hodnoty_x_min, hodnoty_y_min = [], []
  hodnoty_x_max, hodnoty_y_max = [], []

  for _ in range(i):
    x = uniform(a, b)
    y = f(x)

    if y < min_y:
        min_x, min_y = x, y
        hodnoty_x_min.append(min_x)
        hodnoty_y_min.append(min_y)

    if y > max_y:
       max_x, max_y = x, y
       hodnoty_x_max.append(max_x)
       hodnoty_y_max.append(max_y)

  return hodnoty_x_min, hodnoty_y_min, hodnoty_x_max, hodnoty_y_max

x_min, y_min, x_max, y_max = Monte_Carlo_extremy(0, 7)

obr, osy = plt.subplots()
osy.set_xlim(0, 7)
osy.set_ylim(-3.5, 3.5)

body_minima, = osy.plot([], [], "ro")
body_maxima, = osy.plot([], [], "bo")

def init_1():
   body_minima.set_data([], [])
   return body_minima,

def init_2():
   body_maxima.set_data([], [])
   return body_maxima,

def Aktualizace_1(i):
    x = x_min[i]
    y = f(x)
    body_minima.set_data(np.append(body_minima.get_xdata(), x), np.append(body_minima.get_ydata(), y))
    return body_minima,

def Aktualizace_2(i):
    x = x_max[i]
    y = f(x)
    body_maxima.set_data(np.append(body_maxima.get_xdata(), x), np.append(body_maxima.get_ydata(), y))
    return body_maxima,

animace_1 = FuncAnimation(obr, Aktualizace_1, frames=range(len(x_min)), init_func=init_1, interval=250)
animace_2 = FuncAnimation(obr, Aktualizace_2, frames=range(len(x_max)), init_func=init_2, interval=250)

osa_x = np.arange(0, 7, 0.001)
osa_y = f(osa_x)

plt.plot(osa_x, osa_y, color="black")
plt.title("Hledání extrémů funkce f(x) pomocí metody Monte Carlo")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(color="grey")
plt.axvline(color="grey")
plt.grid(True)
plt.show()