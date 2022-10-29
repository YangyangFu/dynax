import taichi as ti
import taichi.math as tm
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


ti.init(arch=ti.cpu)
dtype = ti.f32
needs_grad = False

def f(t, x, a):
    dx = -a*x

    return dx

@ti.func
def _euler(dx:ti.template(), x:ti.template(), dt:ti.f32):
    return x + dx*dt

@ti.kernel
def euler(dx: ti.template(), x: ti.template(), dt: ti.f32) -> ti.f32:
    return _euler(dx, x, dt)

ts = 0
te = 10
dt = 0.2

n = int((te-ts) / dt)

x0 = 1
a = 0.8

t_now = ts
x = [x0]
t = [t_now]
for i in range(n):
    
    dxi = f(t_now, x0, a)
    x_next = euler(dxi, x0, dt)
    t_next = t_now + dt

    x0 = x_next
    t_now = t_next

    t.append(t_next)
    x.append(x_next)

plt.figure()
plt.plot(t,x)
plt.ylabel('x')
plt.xlabel('t')
plt.show()
