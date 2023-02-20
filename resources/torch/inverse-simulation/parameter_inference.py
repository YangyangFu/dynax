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
        self.A = A # state matrix must be 3x3
        self.B = B # input vector must be 3x5
        self.C = C # output vector must be 1x3
        self.D = D # 
        self.d = d # input vector must be 5x1

    def forward(self,t, x):
        x = x.reshape(-1, 1) # reshape from (3) to (3,1)
        
        dx = torch.matmul(self.A, x) + torch.matmul(self.B, self.d) # (3,1)
        dx = dx.reshape(-1) # reshape from (3,1) to (3)

        return dx

class Simulator(torch.nn.Module):
    def __init__(self,params):
        super(Simulator, self).__init__(self, params, ts, te, dt, x0, d, solver)
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
            arg_i = (A, B, d[i, :])
            di = self.d[i,:].reshape(1,-1)
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

def forward_parameters(p, x, ts, te, dt, solver, d, scale):
    """
    p is [Cai, Cwe, Cwi, Re, Ri, Rw, Rg, Twe0, Twi0]
    x is Tz0
    """
    #Cai, Cwe, Cwi, Re, Ri, Rw, Rg, Twe0, Twi0 = p['rc']
    rc_norm = p['rc']
    rc = rc_norm*scale
    # scale up to normal range
    Cai, Cwe, Cwi, Re, Ri, Rw, Rg, Twe0, Twi0 = rc

    A, B, C, D = get_ABCD(Cai, Cwe, Cwi, Re, Ri, Rw, Rg)
    args = (A, B, d)

    # intial point
    x0 = torch.asarray([x, Twe0, Twi0])

    # forward calculation
    t, x = forward(zone_state_space, ts, te, dt, x0, solver, args)

    return t, x


def model(p, x): return forward_parameters(p, x, ts, te, dt, solver, d, scale)


# loss function
def loss_fcn(p, x, y_true, p_lb, p_ub):
    _, y_pred = model(p, x)
    loss = np.mean((y_pred[1:, 0] - y_true)**2)

    penalty = np.sum(torch.nn.Relu(
        p['rc'] - p_ub) + torch.nn.Relu(p_lb - p['rc']))

    norm = (p['rc'] - p_lb) / (p_ub - p_lb)
    reg = torch.linalg.norm(norm, 2)
    return loss + penalty


# data preparation
d = torch.from_numpy(data_train.values[:, :5]).float()
y_train = torch.from_numpy(data_train.values[:, 5]).float()

"""
def fit(data, n_epochs, params: optax.Params, optimizer: optax.GradientTransformation, p_lb, p_ub) -> optax.Params:
    # initialize params
    states = optimizer.init(params)
    x, y = data

    @jit
    def step(params, states, x, y, p_lb, p_ub):
        loss, grads = jax.value_and_grad(loss_fcn)(params, x, y, p_lb, p_ub)
        updates, states = optimizer.update(grads, states, params)
        params = optax.apply_updates(params, updates)

        return params, states, loss, grads

    for epoch in range(n_epochs):
        params, states, loss, grads = step(params, states, x, y, p_lb, p_ub)
        if epoch % 1000 == 0:
            print(f'epoch {epoch}, training loss: {loss}')
            #print(grads['rc'].max(), grads['rc'].min())

    return params
"""

## Run optimization for inference
# parameter settings
#p_lb = torch.from_numpy(np.array([1.0E3, 1.0E4, 1.0E5, 1.0, 1E-01, 1.0, 1.0, 20.0, 20.0]))
#p_ub = torch.from_numpy(np.array([1.0E5, 3.0E4, 3.0E5, 10., 1., 10., 10., 35.0, 30.0]))
p_lb = torch.from_numpy(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.65]))
p_ub = torch.from_numpy(np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]))
#p0 = jnp.array([9998.0869140625, 99998.0859375, 999999.5625, 9.94130802154541, 0.6232420802116394, 1.1442776918411255, 5.741048812866211, 34.82638931274414, 26.184139251708984])

p0 = p_ub
p0 = torch.from_numpy(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8]))

print(p0, x0)
print(loss_fcn({'rc': p0}, x0, y_train, p_lb, p_ub))

n_epochs = 100000  # 5000000


initial_params = {'rc': p0}
optimizer = optim.Adam(initial_params, learning_rate=1e-3)

s = time.time()
for i in range(n_epochs):
    optimizer.zero_grad()

e = time.time()
print(f"execution time is: {e-s} seconds !")

## run for performance check
# forward simulation with infered parameters for the whole data set
ts = 0
te = ts + n*dt
d = data.values[:, :5]
y = data.values[:, 5]
forward_ts = time.time()
def model(p, x): return forward_parameters(p, x, ts, te, dt, solver, d, scale)


t_pred, ys_pred = model(params, x0)
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
params_tolist = [float(p) for p in params['rc']*scale]
with open('zone_coefficients.json', 'w') as f:
    json.dump(params_tolist, f)

A, B, C, D = get_ABCD(*initial_params['rc'][:-2])
print(jnp.linalg.eig(A))

A, B, C, D = get_ABCD(*params['rc'][:-2])
print(jnp.linalg.eig(A))

