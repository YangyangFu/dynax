import jax
import gymnax

rng = jax.random.PRNGKey(0)
rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

# Instantiate the environment & its settings.
env, env_params = gymnax.make("Pendulum-v1")

# Reset the environment.
obs, state = env.reset(key_reset, env_params)

# Sample a random action.
action = env.action_space(env_params).sample(key_act)

# Perform the step transition.
n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)

# Jit-accelerated step transition
jit_step = jax.jit(env.step)

# map (vmap/pmap) across random keys for batch rollouts
reset_rng = jax.vmap(env.reset, in_axes=(0, None))
step_rng = jax.vmap(env.step, in_axes=(0, 0, 0, None))

# map (vmap/pmap) across env parameters (e.g. for meta-learning)
reset_params = jax.vmap(env.reset, in_axes=(None, 0))
step_params = jax.vmap(env.step, in_axes=(None, 0, 0, 0))