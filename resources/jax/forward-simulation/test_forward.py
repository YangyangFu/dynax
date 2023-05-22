from flax import linen as nn
from flax.training import train_state
import jax.numpy as jnp 
import optax 
import jax 

class BlockModel(nn.Module):
    fx: nn.Module
    fu: nn.Module
    #out_dim: int

    def __call__(self, x):
        x1 = self.fx(x)
        x2 = self.fu(x)
        x = x1 + x2
        return x

# blocks

class DNN(nn.Module):
    hidden_dim: int = 64
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


dnn = BlockModel(fx = DNN(), fu = DNN())
#dnn = DNN()

model = dnn

# create toy data
def sine(x):
    return jnp.sin(x)

x = jnp.linspace(0, 2*jnp.pi, 100).reshape(-1,1)
y = sine(x).reshape(-1,1)

@jax.jit
def loss_fn(params, x, y):
    pred = model.apply(params, x)
    print(pred.shape, y.shape)
    return jnp.mean((pred - y)**2)

@jax.jit
def train_step(params, states, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, states = optimizer.update(grads, states, params)
    params = optax.apply_updates(params, updates)
    return params, states, loss, grads

params = model.init(jax.random.PRNGKey(0), x)
print(params.keys())
print(params['params'].keys())
# visualize models
print(model.tabulate(jax.random.PRNGKey(0), jnp.ones((1,1))))


# train
optimizer = optax.sgd(learning_rate = 1e-02)
states_opt = optimizer.init(params)

for i in range(50000):
    params, _, loss, _ = train_step(params, states_opt, x, y)
    if i % 100 == 0:
        print(loss)

# test
x_test = jnp.linspace(0, 2*jnp.pi, 100).reshape(-1,1)
y_test = sine(x_test)
y_pred = model.apply(params, x_test)

print(x_test.shape, y_test.shape, y_pred.shape)

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot(x_test, y_test, label='ground truth')
plt.plot(x_test, y_pred, label='prediction')
plt.savefig('fit.png')
