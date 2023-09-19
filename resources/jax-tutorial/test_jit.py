import jax
import jax.numpy as jnp 
import flax.linen as nn
from dynax.wrapper.envs.rc import RC 
import time 

## NOTE
# - nn.scan seems works only inside a nn.Module

## 
model = nn.Dense(features=5)
key = jax.random.PRNGKey(0)
init_params = model.init(key, jnp.ones((1,5)))
print(init_params)

def rollout(model, carry, keys):
    t, x_prev = carry 

    x = jax.random.normal(keys, shape=(1,5))
    y = model(x + x_prev)
    
    t += 1
    return (t, y), (x, y)

keys = jax.random.split(key, num=10)
carry = (0, jnp.zeros((1,5)))
scan = nn.scan(rollout,
               variable_axes={'params':0},
                variable_broadcast=True,
                in_axes=0,
                out_axes=0,
                )
carry, xsol = scan(model, carry, jnp.array(keys))

# ====================================================
ts = 0
te = 60*60
dt = 60

key = jax.random.PRNGKey(0)
# ================================
#      JIT 3: use nn.scan with a rollout function
# ================================
env = RC(start_time=ts, end_time=te, dt=dt, num_actions=12, name='RC-V1')
t = ts
obs, states, params = env.reset(key)

def rollout(env, carry, inputs):
    t, states = carry 
    action = env.action_space.sample(inputs)
    obs_next, reward, terminated, truncated, info, states = env.step(action, states=states, params=params)
    t += env.dt

    return (t, states), (action, states.x)

key, *keys = jax.random.split(key, num=61)
carry = (env.start_time, states)
print(params)

scan = nn.scan(rollout,
                variable_broadcast='params',
                in_axes=0,
                out_axes=0,
                )
carry, xsol = scan(env, carry, jnp.array(keys))