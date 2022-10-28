import taichi as ti
import taichi.math as tm
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

# get some data
dt = 3600
disturbances = pd.read_csv('./data/disturbance_1min.csv', index_col=[0])
n = len(disturbances)
index = range(0,n*60, 60)
disturbances.index = index
disturbances_dt = disturbances.groupby([disturbances.index // dt]).mean()


Cz = 6953.9422092947289
Cwe = 21567.368048437285
Cwi = 188064.81655062342
Re = 1.4999822619982095
Ri = 0.55089086571081913
Rw = 5.6456475609117183
Rg = 3.9933826145529263
A, B, C, D = get_ABCD(Cz, Cwe, Cwi, Re, Ri, Rw, Rg)

d = disturbances_dt.values
print(d.shape)
# solve ode

def solve_zone(ts, te, x0, nsteps, A, B, d):
    # 
    t = []
    x1 = []
    x2 = []
    x3 = []

    # dt
    dt = int((te - ts) / nsteps)

    # do step
    x0 = x0
    for i in range(nsteps):
        tspan = [dt*i, dt*(i+1)]
        di = d[i,:].reshape(-1,1)
        #print(x0.shape)
        sol = solve_ivp(zone_state_space, tspan, x0, args=(A, B, di))

        # out
        t.extend(sol.t.tolist())
        x1.extend(sol.y[0].tolist())
        x2.extend(sol.y[1].tolist())
        x3.extend(sol.y[2].tolist())
        x0 = sol.y[:,-1]

    x = [x1, x2, x3]

    return t, x

ts = 0
nsteps = 24*28 #d.shape[0]
te = ts + nsteps*dt
x0 = np.array([20, 32, 26.5])

t, x = solve_zone(ts, te, x0, nsteps, A, B, d)

# plot results
plt.figure(figsize = (12, 8))
plt.plot(t, x[0], label="Tz")
plt.plot(t, x[1], label="Twe")
plt.plot(t, x[2], label="Twi")

plt.xlabel('t')
plt.ylabel('Temperature')
plt.legend()
plt.savefig('simpleRC.png')
