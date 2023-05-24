import jax.numpy as jnp
from flax import linen as nn
import jax 

class LinearStateSpaceModel(nn.Module):
    def setup(self):
        self.n_state = 1
        self.n_input = 1
        self.n_output = 1
        self.A = None
        self.B = None
        self.C = None
        self.D = None

    def __call__(self, state, input):
        new_state = self.A @ state + self.B @ input
        output = self.C @ state + self.D @ input
        return new_state, output


class RCModel(LinearStateSpaceModel):
    def setup(self):
        super().setup()
        self.r = self.param('r', nn.initializers.ones, ())
        self.c = self.param('c', nn.initializers.ones, ())
        self.A = -1/(self.r*self.c) * jnp.eye(self.n_state)
        self.B = 1/(self.r*self.c) * jnp.eye(self.n_state)
        self.C = jnp.eye(self.n_output)
        self.D = jnp.zeros((self.n_output, self.n_input))

    def __call__(self, state, input):
        # Calculate A, B, C, D with values based on R and C
        
        
        return super().__call__(state, input)

# example usage
rc_model = RCModel(name="rc_model")
params = rc_model.init(jax.random.PRNGKey(0), jnp.zeros((1,)), jnp.zeros((1,)))
state = jnp.zeros((1,))  # initial state
input = jnp.array([1.0])  # input at the current time step
new_state, output = rc_model.apply(params, state, input)
print(params)

class LinearStateSpaceModel(nn.Module):
    def setup(self):
        self.n_state = 1
        self.n_input = 1
        self.n_output = 1
        self.A = self.param('A', nn.initializers.zeros, (self.n_state, self.n_state))
        self.B = self.param('B', nn.initializers.zeros, (self.n_state, self.n_input))
        self.C = self.param('C', nn.initializers.zeros, (self.n_output, self.n_state))
        self.D = self.param('D', nn.initializers.zeros, (self.n_output, self.n_input))

    def __call__(self, state, input):
        return self._calculate_state_and_output(self.A, self.B, state, input)

    def _calculate_state_and_output(self, A, B, state, input):
        new_state = A @ state + B @ input
        output = self.C @ state + self.D @ input
        return new_state, output


class RCModel(LinearStateSpaceModel):
    def setup(self):
        super().setup()
        self.R = self.param('R', nn.initializers.ones, ())
        self.c = self.param('c', nn.initializers.ones, ())

    def __call__(self, state, input):
        A = -1/(self.R*self.c) * jnp.eye(self.n_state)
        B = 1/(self.R*self.c) * jnp.eye(self.n_state)
        return self._calculate_state_and_output(A, B, state, input)

# example usage
rc_model = RCModel(name="rc_model")
params = rc_model.init(jax.random.PRNGKey(0), jnp.zeros((1,)), jnp.zeros((1,)))
state = jnp.zeros((1,))  # initial state
input = jnp.array([1.0])  # input at the current time step
new_state, output = rc_model.apply(params, state, input)
print(params)