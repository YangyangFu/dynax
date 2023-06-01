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

from dynax.models.RC import Discrete4R3C
from dynax.models.RC import Continuous4R3C

# instantiate a model
#model = Discrete4R3C()
model = Continuous4R3C()
state_dim = model.state_dim
input_dim = model.input_dim
output_dim = model.output_dim

# ===========================================================
# Method 1 for forward simulation: jittable function
# ===========================================================
# investigate the model structure
print(model.tabulate(jax.random.PRNGKey(0), jnp.zeros((state_dim,)), jnp.zeros((input_dim,))))

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
tsol = jnp.arange(0, len(u_train)*dt, dt)
state = jnp.array([y_train[0], 36., 25.])  # initial state

# inherite from a nn.Module seems not a good idea as the parameters are hard to propogate from low-level models to high-level simulator
class Simulator(nn.Module):
    model: nn.Module
    t: jnp.ndarray
    dt: float

    @nn.compact
    def __call__(self, x_init, u):
     
        def scan_fn(carry, ts):
            i, xi = carry
            ui = u[i,:]
            xi_rhs, yi = self.model(xi, ui)
            
            # explicit Euler
            xi = xi + xi_rhs*self.dt

            return (i+1, xi), (xi, yi)

        # module has to be called once before the while loop
        _, _ = self.model(x_init, jnp.zeros_like(u[0,:]))

        # main simulation loop
        u = u.reshape(-1, self.model.input_dim)   
        carry_init = (0, x_init)
        carry_final, (xsol, ysol) = jax.lax.scan(scan_fn, carry_init, self.t)

        return xsol, ysol

# seed
key = jax.random.PRNGKey(0)

# simulator
simulator = Simulator(model, tsol, dt)

# mannually set some parameters
params_init = simulator.init(key, state, u_train) 
params_true = params_init.unfreeze()
params = jnp.array([10383.181640625, 499116.6562, 1321286.5, 1.53524649143219, 0.5000227689743042, 1.0003612041473389, 20.09742546081543])
params_true= {}
params_true['params'] = {}
params_true['params']['model'] = {}
params_true['params']['model'] = {
    'Cai': params[0],
    'Cwe': params[1],
    'Cwi': params[2],
    'Re': params[3],
    'Ri': params[4],
    'Rw': params[5],
    'Rg': params[6]
}
params_true = freeze(params_true)

print(params_init)
print(simulator.tabulate(jax.random.PRNGKey(0), jnp.zeros((model.state_dim,)), u_train))

# define a forward simulation
xs, ys = simulator.apply(params_true, state, u_train)
print(jnp.mean((ys - y_train)**2))


plot = True
if plot:
    plt.figure()
    plt.plot(tsol, y_train, label='Measured')
    plt.plot(tsol, ys, label='Simulated')
    plt.legend()
    plt.savefig('forward.png')

# parameter settings
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
print(params_init)
print(params_lb['params']['model'].values())

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
        pred_loss = jnp.mean((outputs_pred - target)**2)

        # parameter regularization
        normalizer = jax.tree_util.tree_map(lambda x, y: y-x, train_state.params_ub, train_state.params_lb)
        over = jax.tree_util.tree_map(lambda x, y, z: jax.nn.relu(x-y)/z, params, train_state.params_ub, normalizer)
        under = jax.tree_util.tree_map(lambda x, y, z: jax.nn.relu(y-x)/z, params, train_state.params_lb, normalizer)
        reg = sum(jax.tree_util.tree_leaves(over)) + sum(jax.tree_util.tree_leaves(under))
        #reg = jnp.sum()
        
        return pred_loss + reg
    
    loss, grad = jax.value_and_grad(mse_loss)(train_state.params)
    train_state = train_state.apply_gradients(grads=grad)

    return loss, grad, train_state

schedule = optax.exponential_decay(
    init_value = 1e-3, 
    transition_steps = 20000, 
    decay_rate = 0.995, 
    transition_begin=1000, 
    staircase=False, 
    end_value=1e-05
)

optim = optax.chain(
    #optax.clip_by_global_norm(1.0),
    #optax.clip(1.0),
    #optax.scale(1.2),
    optax.adamw(1e-04),
)


train_state = InverseProblemState.create(
    apply_fn=simulator.apply,
    params=params_true,
    tx = optim,
    params_lb = params_lb,
    params_ub = params_ub
)

def mse_loss(params, state_init, u, target):
    # prediction
    _, outputs_pred = train_state.apply_fn(params, state_init, u)
    pred_loss = jnp.mean((outputs_pred - target)**2)

    # parameter regularization
    normalizer = jax.tree_util.tree_map(lambda x, y: y-x, train_state.params_ub, train_state.params_lb)
    over = jax.tree_util.tree_map(lambda x, y, z: jax.nn.relu(x-y)/z, params, train_state.params_ub, normalizer)
    under = jax.tree_util.tree_map(lambda x, y, z: jax.nn.relu(y-x)/z, params, train_state.params_lb, normalizer)
    reg = sum(jax.tree_util.tree_leaves(over)) + sum(jax.tree_util.tree_leaves(under))
    #reg = jnp.sum()
    
    return pred_loss, reg

# check the loss
print(train_state.params)
#loss, reg = mse_loss(params_true, state, u_train, y_train)
#print(loss, reg)
# FIXME: the loss is not correct. mean sqared error is jnp.mean((ys - y_train)**2)
print("====================================")
xs, ys = simulator.apply(params_true, state, u_train)
print(jnp.mean(ys - y_train)**2)
print(jnp.mean(ys - y_train))
_, ys = simulator.apply(params_true, state, u_train)
print(jnp.mean((ys - y_train)**2))
print("====================================")


# training
n_epochs = 10000
for epoch in range(n_epochs):
    loss, grad, train_state = train_step(train_state, state, u_train, y_train)
    if epoch % 100 == 0:
        print('loss at epoch %d: %.4f'%(epoch, loss))

# plot the results
_, outputs_pred = train_state.apply_fn(train_state.params, state, u_train)

plt.figure(figsize=(12, 6))
plt.plot(tsol, y_train, 'b', label='target')
plt.plot(tsol, outputs_pred, 'r', label='prediction')
plt.legend()
plt.savefig('prediction.png')

print(train_state.params['params']['model'].values())