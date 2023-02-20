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

# simulator
class Simulator(torch.nn.Module):
    def __init__(self, params, ts, te, dt, x0, d, solver):
        super(Simulator, self).__init__()
        self.params = torch.nn.Parameter(torch.as_tensor(params), requires_grad=True)
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

### Parameter Inference
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
d = torch.from_numpy(data_train.values[:, :5]).float()

x0 = torch.asarray([20., 30., 26.])
#solver = Dopri5()
solver = 'euler'

# scale parameters
scale = torch.from_numpy(np.array([3.0E4, 5.0E5, 5.0E6, 10., 5., 10., 50., 36.0, 30.0])).float()
#scale = jnp.array([1.0E5, 3.0E4, 5.0E5, 5., 5., 5., 5., 36.0, 30.0])

# loss function
def loss_fcn_customized(y_pred, y_true, p_lb, p_ub, p):
    loss = np.mean((y_pred[1:, 0] - y_true)**2)

    penalty = np.sum(torch.nn.Relu(
        p['rc'] - p_ub) + torch.nn.Relu(p_lb - p['rc']))

    norm = (p['rc'] - p_lb) / (p_ub - p_lb)
    reg = torch.linalg.norm(norm, 2)
    return loss + penalty


# data preparation
d = torch.from_numpy(data_train.values[:, :5]).float()
y_train = torch.from_numpy(data_train.values[:, 5]).float()

## Run optimization for inference
# parameter settings
p_lb = torch.from_numpy(np.array([1.0E3, 1.0E4, 1.0E5, 1.0, 1E-01, 1.0, 1.0]))
p_ub = torch.from_numpy(np.array([1.0E5, 3.0E4, 3.0E5, 10., 1., 10., 10.]))
#p_lb = torch.from_numpy(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
#p_ub = torch.from_numpy(np.array([1., 1., 1., 1., 1., 1., 1.]))
#p0 = jnp.array([9998.0869140625, 99998.0859375, 999999.5625, 9.94130802154541, 0.6232420802116394, 1.1442776918411255, 5.741048812866211, 34.82638931274414, 26.184139251708984])

p0 = p_ub
#p0 = torch.from_numpy(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))

print(p0, x0)
#print(loss_fcn({'rc': p0}, x0, y_train, p_lb, p_ub))

n_epochs = 100000  # 5000000

sim = Simulator(p0, ts, te, dt, x0, d, solver)
for name, param in sim.named_parameters():
    print(name, param)
    print(sim.state_dict())

optimizer = optim.Adam(sim.parameters(), lr=1e-3)
loss = torch.nn.MSELoss
#loss = torch.autograd.Variable(torch.nn.MSELoss(), requires_grad=True)

s = time.time()
for i in range(n_epochs):
    optimizer.zero_grad()
    y_pred = sim()[:,0]
    loss_value = loss()(y_pred, y_train)
    loss_value = torch.autograd.Variable(loss_value, requires_grad=True)
    loss_value.backward()
    optimizer.step()
    if i % 1000 == 0:
        print(f"Epoch {i} loss: {loss_value.item()}")
        print(sim.parameters())
params = sim.parameters()
e = time.time()
print(f"execution time is: {e-s} seconds !")

## run for performance check
# forward simulation with infered parameters for the whole data set
ts = 0
te = ts + n*dt
d = data.values[:, :5]
y = data.values[:, 5]
forward_ts = time.time()
sim_final = Simulator(params, ts, te, dt, x0, d, solver)
t_pred, ys_pred = sim_final()
forward_te = time.time()
print(f"single forward simulation costs {forward_te-forward_ts} s!")
y_pred = ys_pred[:, 0]

print(t_pred.shape, y_pred.shape)
print(y.shape)

plt.figure(figsize=(12, 6))
plt.plot(y, 'b-', label='Target')
plt.plot(y_pred, 'r-', label="Prediction")
plt.ylabel('Temperature (C)')
plt.legend()
plt.savefig('parameter_inference.png')

# save the parameters
params_tolist = [float(p) for p in params*scale]
with open('zone_coefficients.json', 'w') as f:
    json.dump(params_tolist, f)


