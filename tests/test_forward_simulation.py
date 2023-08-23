import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.tree_util import Partial

from dynax.core.discrete_block_state_space import DiscreteLinearSSM
from dynax.models.RC import Discrete4R3C, Continuous4R3C

@Partial(jax.jit, static_argnums=(0,))
def forward_step(model, params, state, input):
    new_state, output = model.apply(params, state, input)
    return new_state, output

# test discrete linear state space model
def test_discrete_linear_ssm():
    # instantiate a 4-state 3-input 4-output discrete linear state space model
    state_dim = 4
    input_dim = 3
    output_dim = 4
    ssm = DiscreteLinearSSM(state_dim=state_dim, input_dim=input_dim, output_dim=output_dim)
    # show model structure
    print(ssm.tabulate(jax.random.PRNGKey(0), jnp.zeros((state_dim,)), jnp.zeros((input_dim,))))

    # initialzie model parameters
    params = ssm.init(jax.random.PRNGKey(0), jnp.zeros((state_dim,)), jnp.zeros((input_dim,)))
    print(params)

    # simulate the model for 1000 steps with jax.jit
    state = jnp.ones((state_dim,))  # initial state
    input = jnp.ones((input_dim,))  # input at the current time step

    n_steps = 1000
    step = 0
    while step < n_steps:
        new_state, output = forward_step(ssm, params, state, input)
        state = new_state
        step += 1
    
    # TODO: add assertion before return
    return

def test_discrete_rc():
    ssm = Discrete4R3C()
    state_dim = ssm.state_dim
    input_dim = ssm.input_dim
    output_dim = ssm.output_dim

    # investigate the model structure
    print(ssm.tabulate(jax.random.PRNGKey(0), jnp.zeros((state_dim,)), jnp.zeros((input_dim,))))

    # assign given params
    rc = {'Cai': 6953.9422092947289,
            'Cwe': 21567.368048437285,
            'Cwi': 188064.81655062342,
            'Re': 1.4999822619982095,
            'Ri': 0.55089086571081913,
            'Rw': 5.6456475609117183,
            'Rg': 3.9933826145529263,
            }
    params = {'params': rc}
    # simulate the model for 100 steps with jax.jit
    state = jnp.ones((state_dim,))  # initial state

    n_steps = 100
    step = 0
    while step < n_steps:
        # random input
        input = jax.random.uniform(jax.random.PRNGKey(step), shape=(input_dim,))
        # advance one step
        new_state, output = forward_step(ssm, params, state, input)

        # update state
        state = new_state
        step += 1

    # add assertion before return 

    return 

if __name__ == '__main__':
    test_discrete_linear_ssm()
    test_discrete_rc() 