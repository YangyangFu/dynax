import jax
import jax.numpy as jnp 
import flax.linen as nn
from dynax.wrapper.envs.rc import RC 

# setup
ts = 0
te = 60*60
dt = 60

# initialize a random key to cotnrol randomness
key = jax.random.PRNGKey(0) 

# instantiate an env
env = RC(start_time=ts, end_time=te, dt=dt, num_actions=11, name='RC-V1')

# reset the environment with default parameters
obs, states, params = env.reset(key)

# sample an action
action = env.action_space.sample(key)

assert action == 7
assert env.action_space.contains(action)

# step the environment
obs_next, reward, terminated, truncated, info, states = env.step(action, states=states, params=params)

## jit the env
# NOTE: this is not as fast as expected
env = nn.jit(RC)(start_time=ts, end_time=te, dt=dt, num_actions=11, name='RC-V1')
obs, states, params = env.reset(key)
action = env.action_space.sample(key)
obs_next, reward, terminated, truncated, info, states = env.step(action, states=states, params=params)


## jit the step: 10-20X speedup
env = RC(start_time=ts, end_time=te, dt=dt, num_actions=11, name='RC-V1')
obs, states, params = jax.jit(env.reset)(key)
action = env.action_space.sample(key)
obs_next, reward, terminated, truncated, info, states = jax.jit(env.step)(action, states=states, params=params)

## vectorized env
obs_res, states_res, params_res = jax.vmap(env.reset)(jax.random.split(key, 10)) 
print(obs_res.shape, states_res.time.shape)

action = jax.vmap(env.action_space.sample)(jax.random.split(key, 20))
print(action.shape)
obs_next, reward, terminated, truncated, info, states = jax.vmap(env.step)(action, states=states, params=params)
print(obs_next.shape)