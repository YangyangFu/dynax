from typing import Callable
import jax 
import jax.numpy as jnp
from jax.tree_util import Partial
import flax.linen as nn
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax.training.train_state import TrainState
from flax.core.frozen_dict import freeze
import optax 
import json 

from dynax.models.RC import Discrete4R3C
from dynax.models.RC import Continuous4R3C
from dynax.simulators.simulator import DifferentiableSimulator

# FIXME: 
#   - the backward propagation is slower than functional programming as previous paper implementation 
#   - this approach seems leads to one-step delayed in the simulation compared with the data. see the forward simulation plots
#   - understand why the baseline model (simply uses values from previous time step as current prediction) is accurate enough for output predictions
# ===========================================================
# instantiate a model
#model = Discrete4R3C()
model = Continuous4R3C()
state_dim = model.state_dim
input_dim = model.input_dim
output_dim = model.output_dim

# load calibration data
data = pd.read_csv('./disturbance_1min.csv', index_col=[0])
n_samples = len(data)
index = range(0, n_samples*60, 60)
data.index = index

# resample to a given time step
dt = 3600
data = data.groupby([data.index // dt]).mean()

# split training and testing
ratio = 0.75
n_train = int(len(data)*ratio)
print(n_train)
data_train = data.iloc[:n_train, :]
data_test = data.iloc[n_train:, :]

# to model signals
u_train = jnp.array(data_train.values[:,:5])
y_train = jnp.array(data_train.values[:,5])

# forward simulation
tsol = jnp.arange(0, len(u_train), 1)*dt
state = jnp.array([y_train[0], 36., 25.])  # initial state

# simulator
simulator = DifferentiableSimulator(model, tsol, dt)

# seed
key = jax.random.PRNGKey(0)

# initialize model parameters
params_init = simulator.init(key, state, u_train) 
print(params_init)
print(simulator.tabulate(jax.random.PRNGKey(0), jnp.zeros((model.state_dim,)), u_train))

# parameter bounds settings
params_lb = params_init.unfreeze()
params_lb['params']['model']['Cai'] = 3.0E3
params_lb['params']['model']['Cwe'] = 5.0E4
params_lb['params']['model']['Cwi'] = 5.0E5
params_lb['params']['model']['Re'] = 1.0
params_lb['params']['model']['Ri'] = 0.5
params_lb['params']['model']['Rw'] = 1.0
params_lb['params']['model']['Rg'] = 5.0
params_lb = freeze(params_lb)

params_ub = params_init.unfreeze()
params_ub['params']['model']['Cai'] = 3.0E4
params_ub['params']['model']['Cwe'] = 5.0E5
params_ub['params']['model']['Cwi'] = 5.0E6
params_ub['params']['model']['Re'] = 1.0E1
params_ub['params']['model']['Ri'] = 5.0
params_ub['params']['model']['Rw'] = 1.0E1
params_ub['params']['model']['Rg'] = 5.0E1
params_ub = freeze(params_ub)

# inverse problem train state
class InverseProblemState(TrainState):
    params_lb: nn.Module
    params_ub: nn.Module

# inverse simulation train_step
@jax.jit
def train_step(train_state, state_init, u, target):

    def mse_loss(params):
        # prediction
        _, outputs_pred = train_state.apply_fn(params, state_init, u)
        # mse loss: match dimensions
        pred_loss = jnp.mean((outputs_pred.reshape(-1,1) - target.reshape(-1,1))**2)

        # parameter regularization
        # normalizer = jax.tree_util.tree_map(lambda x, y: x-y, train_state.params_ub, train_state.params_lb)
        normalizer = train_state.params_ub
        over = jax.tree_util.tree_map(lambda x, y, z: jax.nn.relu(x-y)/z, params, train_state.params_ub, normalizer)
        under = jax.tree_util.tree_map(lambda x, y, z: jax.nn.relu(y-x)/z, params, train_state.params_lb, normalizer)
        reg = sum(jax.tree_util.tree_leaves(over)) + sum(jax.tree_util.tree_leaves(under))
        #reg = jnp.sum()
        
        return pred_loss + reg
    
    loss, grad = jax.value_and_grad(mse_loss)(train_state.params)
    train_state = train_state.apply_gradients(grads=grad)

    return loss, grad, train_state

optim = optax.chain(
    #optax.clip_by_global_norm(1.0),
    #optax.clip(1.0),
    #optax.scale(1.2),
    optax.lamb(1e-03),
)

train_state = InverseProblemState.create(
    apply_fn=simulator.apply,
    params=params_ub,
    tx = optim,
    params_lb = params_lb,
    params_ub = params_ub
)

# training
n_epochs = 5000
for epoch in range(n_epochs):
    loss, grad, train_state = train_step(train_state, state, u_train, y_train)
    if epoch % 1000 == 0:
        grad_norm = jnp.linalg.norm(jnp.array(jax.tree_util.tree_leaves(grad)))
        print('loss at epoch %d: %.4f, grad norm %.4f'%(epoch, loss, grad_norm))

# save trained parameters
params_trained = train_state.params.unfreeze()['params']['model']
for k in params_trained.keys():
    params_trained[k] = params_trained[k].item()
print(params_trained)
with open('zone_coefficients.json', 'w') as f:
    json.dump(params_trained, f)


## ==============================================================================
## post processing
## ==============================================================================
# check prediction on training and testing data
u = jnp.array(data.values[:,:5])
y_true = jnp.array(data.values[:,5])
tsol = jnp.arange(0, len(y_true)*dt, dt)
simulator = DifferentiableSimulator(model, tsol, dt)
_, outputs_pred = simulator.apply(train_state.params, state, u)

plt.figure(figsize=(12, 6))
plt.plot(y_true, 'b', label='target')
plt.plot(outputs_pred, 'r', label='prediction')
plt.vlines(x=n_train, ymin=min(y_true), ymax=max(y_true), color='k', linestyles='--', lw=3, label='Train/Test Split')
plt.ylabel('Temperature (C)')
plt.legend()
plt.grid()
plt.savefig('parameter_inference.png')

# TODO: how to understand this?
# how about we compare the prediction with a baseline model
# baseline model: use previous output as current prediction
y_pred_baseline = 23. * jnp.ones_like(y_true)
# update baseline model. do not use jax.ops
y_pred_baseline = y_pred_baseline.at[1:].set(y_true[:-1])
plt.figure(figsize=(12, 6))
plt.plot(y_true, 'b', lw=0.5, label='target')
plt.plot(outputs_pred, 'r', lw=0.5, label='prediction')
plt.plot(y_pred_baseline, 'g', lw=0.5, label='baseline')
plt.vlines(x=n_train, ymin=min(y_true), ymax=max(y_true), color='k', linestyles='--', lw=3, label='Train/Test Split')
plt.ylabel('Temperature (C)')
plt.legend()
plt.grid()
plt.savefig('ssm_vs_baseline.png')

# TODO: plot at a finer resolution. The simulation seems one-step delayed
plt.figure(figsize=(12, 6))
plt.plot(y_true[:24], 'b', label='target')
plt.plot(outputs_pred[:24], 'r', label='prediction')
plt.ylabel('Temperature (C)')
plt.legend()
plt.grid()
plt.savefig('prediction_one_day.png')
