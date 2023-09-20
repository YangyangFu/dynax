import jax
import jax.numpy as jnp 
import flax.linen as nn
from dynax.wrapper.envs.rc import RC 

# Expected results
OBS_NEXT = jnp.array([6.0000000e+01, 2.0000000e+01, 2.0036667e+01, 1.2346695e-02, 1.2000000e+00])
REWARD = -0.05974
TERMINATED = False
TRUNCATED = False
TIME_NEXT = 60.0 
STATES_NEXT = jnp.array([20.055023, 29.99874 , 25.999638])

## tests setup
ts = 0
te = 60*60
dt = 60.

# initialize a random key to cotnrol randomness
key = jax.random.PRNGKey(0) 

def test_basic_forward():
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
    
    assert jnp.allclose(obs_next, OBS_NEXT)
    assert reward == REWARD
    assert bool(terminated) == TERMINATED
    assert bool(truncated) == TRUNCATED
    assert states.time == TIME_NEXT
    assert jnp.allclose(states.x, STATES_NEXT)

def test_jittable_module():
    """ JIT the whole module

    NOTE: This might not be faster than jit functions only.
    """ 
    # jit the whole module
    env = nn.jit(RC)(start_time=ts, end_time=te, dt=dt, num_actions=11, name='RC-V1')
    # reset
    obs, states, params = env.reset(key)
    # random action
    action = env.action_space.sample(key)
    # step
    obs_next, reward, terminated, truncated, info, states = env.step(action, states=states, params=params)

    # check
    assert jnp.allclose(obs_next, OBS_NEXT)
    assert reward == REWARD
    assert bool(terminated) == TERMINATED
    assert bool(truncated) == TRUNCATED
    assert states.time == TIME_NEXT
    assert jnp.allclose(states.x, STATES_NEXT)

def test_jittable_functions():
    ## jit the step: 10-20X speedup 
    env = RC(start_time=ts, end_time=te, dt=dt, num_actions=11, name='RC-V1')
    obs, states, params = jax.jit(env.reset)(key)
    action = env.action_space.sample(key)
    obs_next, reward, terminated, truncated, info, states = jax.jit(env.step)(action, states=states, params=params)    

    assert jnp.allclose(obs_next, OBS_NEXT)
    assert reward == REWARD
    assert bool(terminated) == TERMINATED
    assert bool(truncated) == TRUNCATED
    assert states.time == TIME_NEXT
    assert jnp.allclose(states.x, STATES_NEXT)

def test_vectorized_reset():
    """ Tester for a vectorized env

    We instantiate an environment, but reset 20 different states if stochastic reset is allowed.

    """
    nparallels = 20

    # instantiate an env
    env = RC(start_time=ts, end_time=te, dt=dt, num_actions=11, name='RC-V1')
    
    # make vectorized reset func 
    keys = jax.random.split(key, nparallels)

    # NOTE: if use in_axes, then positional argument should be used
    # keyword arguments will be mapped over their axis 0 by default.
    reset_fcn = lambda key, deterministic: env.reset(key, determnistic=deterministic)
    vec_reset = jax.vmap(reset_fcn, in_axes=(0, None))

    # call vectorized reset
    deterministic = False
    obs_res, states_res, params_res = vec_reset(keys, deterministic) 

    assert obs_res.shape == (nparallels, 5)
    assert states_res.x.shape == (nparallels, 3)  

def test_vectorized_step():
    """ Testers for a vectorized env

    We inistantiate an environment with the same starting states, but with forward one step with 20 different actions.

    """
    nparallels = 20

    # instantiate env
    env = RC(start_time=ts, end_time=te, dt=dt, num_actions=11, name='RC-V1')
    # reset to same initial states
    obs, states, params = jax.jit(env.reset)(key)

    # generate random actions
    action = jax.vmap(env.action_space.sample)(jax.random.split(key, nparallels))

    # run different actions at the same time
    obs_next, reward, terminated, truncated, info, states = jax.vmap(env.step, in_axes=(0, None, None))(action, states, params)

    assert obs_next.shape == (nparallels, 5)
    assert reward.shape == (nparallels, )
    assert terminated.shape == (nparallels, )
    assert truncated.shape == (nparallels, )
    assert states.time.shape == (nparallels, )
    assert states.x.shape == (nparallels, 3)


def test_vectorized_all():
    """ Testers for a vectorized env

    We inistantiate an environment with the different starting states, and forward one step with different actions.

    """
    nparallels = 20

    # instantiate an env
    env = RC(start_time=ts, end_time=te, dt=dt, num_actions=11, name='RC-V1')
    
    # make vectorized reset func 
    keys = jax.random.split(key, nparallels)

    # NOTE: if use in_axes, then positional argument should be used
    # keyword arguments will be mapped over their axis 0 by default.
    reset_fcn = lambda key, deterministic: env.reset(key, determnistic=deterministic)
    vec_reset = jax.vmap(reset_fcn, in_axes=(0, None))

    # call vectorized reset
    deterministic = False
    obs_res, states_res, params_res = vec_reset(keys, deterministic)     

    # vectorized action generation
    actions = jax.vmap(env.action_space.sample)(keys)
    
    # run different actions at the same time
    obs_next, reward, terminated, truncated, info, states = jax.vmap(env.step, in_axes=(0, 0, 0))(actions, states_res, params_res)
    
    assert obs_next.shape == (nparallels, 5)
    assert reward.shape == (nparallels, )
    assert terminated.shape == (nparallels, )
    assert truncated.shape == (nparallels, )
    assert states.time.shape == (nparallels, )
    assert states.x.shape == (nparallels, 3)


if __name__ == "__main__":
    test_basic_forward()
    test_jittable_module()
    test_jittable_functions()
    test_vectorized_reset()
    test_vectorized_step()
    test_vectorized_all()

    print("All tests in test_gymwrapper passed !!!")
