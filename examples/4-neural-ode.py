"""Train a neural ODE based on given data.

The building dynamics is described by a linear state space model in the following format:

$$ \dot x_t =  fx(x_in) $$
$$ y_t = Cx_t $$

"""
from jax import config 
config.update("jax_debug_nans", False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax 
import jax.numpy as jnp 
import flax.linen as nn
import optax 
import pandas as pd 

from dynax.core.base_block_state_space import BaseContinuousBlockSSM
from dynax.simulators.simulator import DifferentiableSimulator
from dynax.trainer.train_state import TrainState

class NeuralDynamic(BaseContinuousBlockSSM):
    state_dim: int 
    input_dim: int
    output_dim: int = 1

    def setup(self):
        super().setup()
        self._fx = self.fx(output_dim = self.state_dim)
        self._fy = self.fy(output_dim = self.output_dim)

    class fx(nn.Module):
        output_dim: int

        def setup(self):
            self.dense = nn.Dense(self.output_dim)
            #self.dense2 = nn.Dense(self.output_dim)
        
        def __call__(self, states, inputs):
            x = self.dense(inputs)
            #x = nn.relu(x)
            #x = self.dense2(x)

            return x
    
    class fy(nn.Module):
        output_dim: int

        def setup(self):
            self.dense = nn.Dense(self.output_dim)
        
        def __call__(self, states, inputs):
            return self.dense(states)

class Model(nn.Module):
    state_dim:int 
    input_dim: int
    output_dim: int = 1
    ts: int = 0.
    dt: int = 900.
    
    def setup(self):
        super().setup()
        self.estimator = nn.Dense(features=self.state_dim, use_bias=True)
        self.dynamic = NeuralDynamic(self.state_dim, self.input_dim, self.output_dim)
        self.simulator = DifferentiableSimulator(
            self.dynamic, 
            dt=self.dt, 
            mode_interp='linear', 
            start_time=self.ts
        )
        
    def __call__(self, x0, u):
        # x0: (Dx0,)
        # u: (L, Du)
        # estimator
        state_est = self.estimator(x0)
        state_est = nn.relu(state_est)

        # dynamics
        # (Dx, ), (L, Du) -> (L, Do)
        _, out = self.simulator(state_est, u)

        return out

key = jax.random.PRNGKey(2023)

model = Model(state_dim=3, input_dim=5, output_dim=1)
init_params = model.init(key, jnp.zeros(6,), jnp.zeros((2,5)))
print(model.tabulate(key, jnp.zeros(6,), jnp.zeros((2,5))))

# test model
yout = model.apply(init_params, jnp.zeros(6,), jnp.zeros((32,5)))
print(yout.shape)

ts = 0
dt = 3600.

# create an inverse problem to learn a model from given data

# inverse simulation train_step
@jax.jit
def train_step(train_state, state_init, u, target):
    def mse_loss(params):
        # prediction
        outputs_pred = train_state.apply_fn(params, state_init, u)
        #outputs_pred = jnp.clip(outputs_pred, 16., 40.)
        # mse loss: match dimensions
        pred_loss = jnp.mean((outputs_pred - target)**2)
        return pred_loss
    
    loss, grad = jax.value_and_grad(mse_loss)(train_state.params)
    train_state = train_state.apply_gradients(grads=grad)

    return loss, grad, train_state

schedule = optax.linear_schedule(
    init_value = 1e-1, 
    transition_steps = 5000, 
    transition_begin=0, 
    end_value=1e-4
)

optim = optax.chain(
    #optax.clip_by_global_norm(1.0),
    optax.clip(1.),
    #optax.scale(1.2),
    #optax.lamb(1e-04),
    optax.adamw(schedule)
)

train_state = TrainState.create(
    apply_fn=model.apply,
    params=init_params,
    tx = optim,
)

# load data
data = pd.read_csv('./data/eplus_1min.csv', index_col=[0])
n_samples = len(data)
index = range(0, n_samples*60, 60)
data.index = index

# resample to a given time step
data = data.groupby([data.index // dt]).mean()

# split training and testing
ratio = 0.75
n_train = int(len(data)*ratio)
data_train = data.iloc[:n_train, :]
data_test = data.iloc[n_train:, :]

# to model signals
u_train = jnp.array(data_train.values[:,:5])
y_train = jnp.array(data_train.values[:,5]).reshape(-1, 1)
print("we have a training data set of :", u_train.shape[0])

def normalize(u):
    mu = jnp.mean(u, axis=0)
    std = jnp.std(u, axis=0)

    return mu, std, (u - mu) / std

u_mu, u_std, u_train = normalize(u_train)
y_mu, y_std, y_train = normalize(y_train)

# ===============================
# training loop
# training
print(u_train.shape, y_train.shape)

init_state = y_train[0,0]*jnp.ones((6,))

# process data for batched training
# train by add `period` length of data to previous trianing data
n_samples = len(u_train)
period = int(24.*3600 / dt)
n_periods = n_samples // period + 1

#==========================
n_epochs = 200
for epoch in range(n_epochs):
    for b in range(n_periods):
        indx_end = min(n_samples, (b+1)*period)
        xb_train = u_train[:indx_end, :]
        yb_train = y_train[:indx_end, :] 
        loss, grad, train_state = train_step(train_state, init_state, xb_train, yb_train)
    #if epoch % 1000 == 0:
        #grad_norm = jnp.linalg.norm(jnp.array(jax.tree_util.tree_leaves(grad)))
    print('loss at epoch %d: %.4f'%(epoch, loss))
print(grad)
# save trained parameters

# check prediction results
# check prediction on training and testing data
u = (jnp.array(data.values[:,:5]) - u_mu)/u_std
y_true = jnp.array(data.values[:,5])
ts = 0 
te = (len(y_true)-1)*dt
outputs_pred = model.apply(train_state.params, init_state, u)
y_pred = (outputs_pred)*y_std + y_mu 

plt.figure(figsize=(12, 6))
plt.plot(y_true, 'b', label='target')
plt.plot(y_pred, 'r', label='prediction')
plt.vlines(x=n_train, ymin=min(y_true), ymax=max(y_true), color='k', linestyles='--', lw=3, label='Train/Test Split')
plt.ylabel('Temperature (C)')
plt.legend()
plt.grid()
plt.savefig('parameter_inference.png')
