"""Train a neural ODE based on given data.

The building dynamics is described by a linear state space model in the following format:

$$ \dot x_t =  fx(x_in) $$
$$ y_t = Cx_t $$


Vanishing gradient problems with long sequence.

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

# =================================
# 
# load data
data = pd.read_csv('./data/eplus_1min.csv', index_col=[0])
n_samples = len(data)
index = range(0, n_samples*60, 60)
data.index = index

# resample to a given time step]
dt = 900
data = data.groupby([data.index // dt]).mean()

# add lag to target
def add_lags(data, col, lags):
    for lag in range(1, lags+1):
        data['y_t-'+str(lag)] = data[col].shift(lag)
    return data
print(data.head())
target_col = 'weighted_average'
lags = 0
data = add_lags(data, target_col, lags)
data = data.dropna()
print(data.head())

# split training and testing
ratio = 0.25
n_train = int(len(data)*ratio)
data_train = data.iloc[:n_train, :]
data_test = data.iloc[n_train:, :]

# to model signals
u_train = jnp.array(data_train.drop(columns=[target_col]).values)
y_train = jnp.array(data_train.loc[:, target_col].values)
print("we have a training data set of :", u_train.shape[0])

def normalize(u):
    mu = jnp.mean(u, axis=0)
    std = jnp.std(u, axis=0)

    return mu, std, (u - mu) / std

mu, std, u_train = normalize(u_train)

# ===============================
# 
# define model
class NeuralRC(BaseContinuousBlockSSM):
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
            self.dense = nn.Dense(self.output_dim, kernel_init=nn.initializers.constant(0.01))
            #self.dense2 = nn.Dense(self.output_dim)
        
        def __call__(self, states, inputs):
            x = self.dense(inputs)
            x = nn.tanh(x)
            #x = self.dense2(x)

            return x
    
    class fy(nn.Module):
        output_dim: int

        def setup(self):
            self.dense = nn.Dense(self.output_dim)
        
        def __call__(self, states, inputs):
            return self.dense(states)
        
key = jax.random.PRNGKey(2023)

estimator = nn.Dense(features=3, use_bias=True)
dynamic = nn.Dense(features=3, use_bias=False)

class Model(nn.Module):
    
    state_dim: int = 3

    @nn.compact 
    def __call__(self, x):
        state_est = nn.Dense(self.state_dim)(x)
        state_est = nn.relu(state_est)
        dynamics = nn.Dense(self.state_dim)(state_est)
        out = nn.Dense(1)(dynamics + state_est)
        
        return jnp.squeeze(out, axis=-1)

model = Model(state_dim=3)


init_params = model.init(key, u_train)
init_out = model.apply(init_params, u_train)

# inverse simulation train_step
@jax.jit
def train_step(train_state, x, target):

    def mse_loss(params):
        # prediction
        outputs_pred = train_state.apply_fn(params, x)
        # mse loss: match dimensions
        pred_loss = jnp.mean((outputs_pred - target)**2)
        print("=================== 111")
        print(outputs_pred.shape, target.shape, (outputs_pred - target).shape)
        return pred_loss
    
    loss, grad = jax.value_and_grad(mse_loss)(train_state.params)
    train_state = train_state.apply_gradients(grads=grad)

    return loss, grad, train_state

schedule = optax.linear_schedule(
    init_value = 1e-2, 
    transition_steps = 10000, 
    transition_begin=0, 
    end_value=1e-05
)

optim = optax.chain(
    #optax.clip_by_global_norm(1.0),
    #optax.clip(0.01),
    #optax.scale(1.2),
    #optax.lamb(1e-04),
    optax.adam(schedule)
)

train_state = TrainState.create(
    apply_fn=model.apply,
    params=init_params,
    tx = optim,
)

# ===============================
# training loop
# training
print(u_train.shape, y_train.shape)

#==========================
n_epochs = 10000
for epoch in range(n_epochs):
    loss, grad, train_state = train_step(train_state, u_train, y_train)
    if epoch % 1000 == 0:
        #grad_norm = jnp.linalg.norm(jnp.array(jax.tree_util.tree_leaves(grad)))
        print('loss at epoch %d: %.4f'%(epoch, loss))
print(grad)
# save trained parameters


# check prediction results
# check prediction on training and testing data
u = (data.drop(columns=[target_col]).values - mu)/std
y_true = data.loc[:, target_col].values
ts = 0 
te = (len(y_true)-1)*dt
y_pred = model.apply(train_state.params, u)

plt.figure(figsize=(12, 6))
plt.plot(y_true, 'b', label='target')
plt.plot(y_pred, 'r', label='prediction')
plt.vlines(x=n_train, ymin=min(y_true), ymax=max(y_true), color='k', linestyles='--', lw=3, label='Train/Test Split')
plt.ylabel('Temperature (C)')
plt.legend()
plt.grid()
plt.savefig('parameter_inference.png')
