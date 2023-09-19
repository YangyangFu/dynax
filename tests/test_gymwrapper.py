import jax
import jax.numpy as jnp 
import flax.linen as nn
from dynax.wrapper.envs.rc import RC 
import time 

# setup
ts = 0
te = 60*60
dt = 60

# ================================
#        NO JIT
# ================================
# instantiate an env
env = RC(start_time=ts, end_time=te, dt=dt, num_actions=11, name='RC-V1')
key = jax.random.PRNGKey(0)
t = ts 

obs, states, params = env.reset(key)
init_params = env.init(key, 0, states)

start_time = time.time()
# main loop
while t < te:
    key, key_reset = jax.random.split(key)
    action = env.action_space.sample(key)
    obs_next, reward, terminated, truncated, info, states = env.step(action, states=states, params=params)
    t += dt
print("non-jittable run time: ", time.time() - start_time)

"""
# test jittable conditions
env = nn.jit(RC)(start_time=ts, end_time=te, dt=dt, num_actions=11, name='RC-V1')
t = ts 

obs, states, params = env.reset(key)
init_params = env.init(key, 0, states)

start_time = time.time()
# main loop
while t < te:
    key, key_reset = jax.random.split(key)
    action = env.action_space.sample(key)
    obs_next, reward, terminated, truncated, info, states = env.step(action, states=states, params=params)
    print(action, info)
    t += dt

print("jittable run time: ", time.time() - start_time)
"""

# ================================
#        JIT 1: jit step() only
# ================================
# test jittable conditions
env = nn.jit(RC)(start_time=ts, end_time=te, dt=dt, num_actions=11, name='RC-V1')

t = ts 
obs, states, params = env.reset(key)
init_params = env.init(key, 0, states)
start_time = time.time()

# main loop
while t < te:
    key, key_reset = jax.random.split(key)
    action = env.action_space.sample(key)
    obs_next, reward, terminated, truncated, info, states = jax.jit(env.step)(action, states=states, params=params)
    t += dt

print("jittable 1 run time: ", time.time() - start_time)

# ================================
#       JIT 2: use jax.lax.scan with a rollout function
# ================================

env = RC(start_time=ts, end_time=te, dt=dt, num_actions=12, name='RC-V1')
t = ts 
obs, states, params = env.reset(key)

def rollout(carry, keys):
    step, states = carry 
    action = env.action_space.sample(keys)
    obs_next, reward, terminated, truncated, info, states = env.step(action, states=states, params=params)
    
    step += 1

    return (step, states), (action, states.x)

key, *keys = jax.random.split(key, num=61)
carry = (env.start_time, states)
#scan = jax.lax.scan(rollout,
#                variable_broadcast='params',
#                split_rngs={'params':False},
#                in_axes=0,
#                out_axes=0,
#                )

start_time = time.time()
_, xsol = jax.lax.scan(rollout, carry, jnp.array(keys))
print("jittable run time: ", time.time() - start_time)
actions, xs = xsol
print(actions)

# ================================
#      JIT 3: use nn.scan with a rollout function
# nn.scan seems only works inside a nn.Module
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
                in_axes=0,
                out_axes=0,
                )
carry, xsol = scan(env, carry, jnp.array(keys))