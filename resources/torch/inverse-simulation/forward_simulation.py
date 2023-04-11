import torch
import numpy as np
import json
import pandas as pd
import time
import matplotlib.pyplot as plt

from torch.autograd import grad, backward
from torchdiffeq import odeint
import torch.optim as optim


def get_ABCD(Cai, Cwe, Cwi, Re, Ri, Rw, Rg):
    A = torch.zeros((3, 3))
    B = torch.zeros((3, 5))
    C = torch.zeros((1, 3))
    A[0, 0] = -1/Cai*(1/Rg+1/Ri)
    A[0, 2] = 1/(Cai*Ri)
    A[1, 1] = -1/Cwe*(1/Re+1/Rw)
    A[1, 2] = 1/(Cwe*Rw)
    A[2, 0] = 1/(Cwi*Ri)
    A[2, 1] = 1/(Cwi*Rw)
    A[2, 2] = -1/Cwi*(1/Rw+1/Ri)

    B[0, 0] = 1/(Cai*Rg)
    B[0, 1] = 1/Cai
    B[0, 2] = 1/Cai
    B[1, 0] = 1/(Cwe*Re)
    B[1, 3] = 1/Cwe
    B[2, 4] = 1/Cwi

    C[0, 0] = 1

    D = 0

    return A, B, C, D

# state space model


class LSSMWithConstantInputs(torch.nn.Module):
    def __init__(self, A, B, C, D, d):
        super(LSSMWithConstantInputs, self).__init__()
        self.A = A  # state matrix must be 3x3
        self.B = B  # input vector must be 3x5
        self.C = C  # output vector must be 1x3
        self.D = D
        self.d = d  # input vector must be 5x1

    def forward(self, t, x):
        x = x.reshape(-1, 1)  # reshape from (3) to (3,1)

        dx = torch.matmul(self.A, x) + torch.matmul(self.B, self.d)  # (3,1)
        dx = dx.reshape(-1)  # reshape from (3,1) to (3)

        return dx


class Simulator(torch.nn.Module):
    def __init__(self, params, ts, te, dt, x0, d, solver):
        super(Simulator, self).__init__()
        self.params = torch.nn.Parameter(torch.as_tensor(params))
        self.ts = ts
        self.te = te
        self.dt = dt
        self.x0 = x0
        self.odeint = odeint
        self.d = d
        self.solver = solver
        self.t = torch.arange(self.ts, self.te, self.dt)

        self.A, self.B, self.C, self.D = get_ABCD(*params)

    def forward(self):

        ## solve odes for given time steps with piece-wise constant inputs
        # initialize x0
        x = self.x0
        # initialize output
        x_all = self.x0.reshape(1, -1)
        # main loop
        for i in range(len(self.t)-1):
            # inputs at time ti
            di = self.d[i, :].reshape(-1, 1)
            rhs = LSSMWithConstantInputs(self.A, self.B, self.C, self.D, di)
            # solve ode for ti to ti+1
            x = odeint(rhs, x, self.t[i:i+2], method=self.solver)
            # only take the last step
            x = x[-1, :].reshape(1, -1)
            # concatenate
            x_all = torch.cat((x_all, x), dim=0)
            # reshape for next iteration
            x = x.reshape(-1)

        return x_all

### forward simulation

# load training data - 1-min sampling rate
data = pd.read_csv('./disturbance_1min.csv', index_col=[0])
index = range(0, len(data)*60, 60)
data.index = index

# sample every hour
dt = 3600.
data = data.groupby([data.index // dt]).mean()
n = len(data)

# split training and testing
ratio = 0.75
n_train = int(len(data)*ratio)
print(n_train)
data_train = data.iloc[:n_train, :]
data_test = data.iloc[n_train:, :]

# define training parameters
ts = 0.
te = ts + n_train*dt


# forward steps
Cz = 6953.9422092947289
Cwe = 21567.368048437285
Cwi = 188064.81655062342
Re = 1.4999822619982095
Ri = 0.55089086571081913
Rw = 5.6456475609117183
Rg = 3.9933826145529263
d = torch.from_numpy(data_train.values[:, :5]).float()

x0 = torch.asarray([20., 30., 26.])
#solver = Dopri5()
solver = 'euler'

params = [Cz, Cwe, Cwi, Re, Ri, Rw, Rg]
sim = Simulator(params, ts, te, dt, x0, d, solver)

# sovle with timer
st = time.process_time()
t_all = torch.arange(ts, te, dt)
x_all = sim()
et = time.process_time()
print(f"execution time is: {et - st}s")

# plot
plt.figure()
plt.plot(t_all, x_all[:, 0], label='Tz Predicted')
plt.plot(t_all, x_all[:, 1], label='Twe')
plt.plot(t_all, x_all[:, 2], label='Twi')
plt.plot(t_all, data_train['weighted_average'], label='Tz Actual')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.savefig('forward_simulation.png')
plt.show()
