import taichi as ti
import taichi.math as tm
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)
dtype = ti.f32
needs_grad = False

Rg = ti.field(dtype=dtype, shape=(), needs_grad = needs_grad)
Re = ti.field(dtype=dtype, shape=(), needs_grad = needs_grad)
Ri = ti.field(dtype=dtype, shape=(), needs_grad = needs_grad)
Rw = ti.field(dtype=dtype, shape=(), needs_grad = needs_grad)
Cwe = ti.field(dtype=dtype, shape=(), needs_grad = needs_grad)
Cwi = ti.field(dtype=dtype, shape=(), needs_grad = needs_grad)
Cai = ti.field(dtype=dtype, shape=(), needs_grad=needs_grad)

# a simple SGD with gradient clip
@ti.data_oriented
class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for w in self.params:
            self._step(w)

    @ti.kernel
    def _step(self, w: ti.template()):
        for I in ti.grouped(w):
            w[I] -= min(max(w.grad[I], -20.0), 20.0) * self.lr

    def zero_grad(self):
        for w in self.params:
            w.grad.fill(0.0)


# a simple RC zone model -> ODE system
"""
This is a 4r3c model for zone temperature prediction
ODE::   xdot = Ax + Bd
        y = Cx
States:
        x = [Tai, Twe, Twi]'
Disturbances:
        u = [Tao, qCon_i, qHVAC, qRad_e, qRad_i]'
Output:
        y = [Tai]
===================================================
"""

def get_ABCD(Cai, Cwe, Cwi, Re, Ri, Rw, Rg):
    A = np.zeros((3,3))
    B = np.zeros((3,5))
    C = np.zeros((1,3))
    A[0,0] = -1/Cai*(1/Rg+1/Ri)
    A[0,2] = 1/(Cai*Ri)
    A[1,1] = -1/Cwe*(1/Re+1/Rw)
    A[1,2] = 1/(Cwe*Rw)
    A[2,0] = 1/(Cwi*Ri)
    A[2,1] = 1/(Cwi*Rw)
    A[2,2] = -1/Cwi*(1/Rw+1/Ri)

    B[0,0] = 1/(Cai*Rg)
    B[0,1] = 1/Cai
    B[0,2] = 1/Cai 
    B[1,0] = 1/(Cwe*Re)
    B[1,3] = 1/Cwe
    B[2,4] = 1/Cwi

    C[0,0] = 1

    D = 0
    
    return A, B, C, D

def zone_state_space(t, x, A, B, d):
    x = x.reshape(-1,1)
    dx = np.matmul(A,x) + np.matmul(B,d) 
    dx = dx.reshape(-1)

    return dx

x0 = np.array([20, 32, 26.5])
Cz = 6953.9422092947289
Cwe = 21567.368048437285
Cwi = 188064.81655062342
Re = 1.4999822619982095
Ri = 0.55089086571081913
Rw = 5.6456475609117183
Rg = 3.9933826145529263
A, B, C, D = get_ABCD(Cz, Cwe, Cwi, Re, Ri, Rw, Rg)

d = np.zeros((5, 1))
d[0] = 30
# test implementation
dx0 = zone_state_space(0,x0, A, B, d)

# solve ode
sol = solve_ivp(zone_state_space, [0, 3600*24.], x0, args=(A, B, d))

# plot results
plt.figure(figsize = (12, 8))
plt.plot(sol.t, sol.y[0], label="Tz")
plt.plot(sol.t, sol.y[1], label="Twe")
plt.plot(sol.t, sol.y[2], label="Twi")
plt.xlabel('t')
plt.ylabel('Temperature')
plt.legend()
plt.savefig('simpleRC.png')
