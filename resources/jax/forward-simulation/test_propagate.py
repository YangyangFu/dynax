from typing import Callable, List, Tuple, Union, Optional

import jax 
import jax.numpy as jnp
import flax.linen as nn 

class BaseLinearStateSpaceModel(nn.Module):
    state_dim: int
    input_dim: int 
    output_dim: int
    
    def setup(self):
        self.A = None
        self.B = None
        self.C = None
        self.D = None

    def __call__(self, state, input):
        new_state = self.A @ state + self.B @ input
        output = self.C @ state + self.D @ input
        return new_state, output

class LearnableLinearStateSpaceModel(BaseLinearStateSpaceModel):
    def setup(self):
        super().setup()
        self.A = self.param('A', nn.initializers.zeros, (self.n_state, self.n_state))
        self.B = self.param('B', nn.initializers.zeros, (self.n_state, self.n_input))   
        self.C = self.param('C', nn.initializers.zeros, (self.n_output, self.n_state)) 
        self.D = self.param('D', nn.initializers.zeros, (self.n_output, self.n_input))

    def __call__(self, state, input):
        return super().__call__(state, input)
    
class RCModel(BaseLinearStateSpaceModel):

    def setup(self):
        super().setup()
        Cai = self.param('Cai', nn.initializers.ones, ())
        Cwe = self.param('Cwe', nn.initializers.ones, ())
        Cwi = self.param('Cwi', nn.initializers.ones, ())
        Re = self.param('Re', nn.initializers.ones, ())
        Ri = self.param('Ri', nn.initializers.ones, ())
        Rw = self.param('Rw', nn.initializers.ones, ())
        Rg = self.param('Rg', nn.initializers.ones, ())

        self.A, self.B, self.C, self.D = self._getABCD([Cai, Cwe, Cwi, Re, Ri, Rw, Rg])

    def __call__(self, state, input):
        # Calculate A, B, C, D with values based on R and C
        return super().__call__(state, input)

    def _getABCD(self, params):
        # unpack
        Cai, Cwe, Cwi, Re, Ri, Rw, Rg = params
        # initialzie
        A = jnp.zeros((3, 3))
        B = jnp.zeros((3, 5))
        C = jnp.zeros((1, 3))
        D = jnp.zeros((1, 5))
        A = A.at[0, 0].set(-1/Cai*(1/Rg+1/Ri))
        A = A.at[0, 2].set(1/(Cai*Ri))
        A = A.at[1, 1].set(-1/Cwe*(1/Re+1/Rw))
        A = A.at[1, 2].set(1/(Cwe*Rw))
        A = A.at[2, 0].set(1/(Cwi*Ri))
        A = A.at[2, 1].set(1/(Cwi*Rw))
        A = A.at[2, 2].set(-1/Cwi*(1/Rw+1/Ri))

        B = B.at[0, 0].set(1/(Cai*Rg))
        B = B.at[0, 1].set(1/Cai)
        B = B.at[0, 2].set(1/Cai)
        B = B.at[1, 0].set(1/(Cwe*Re))
        B = B.at[1, 3].set(1/Cwe)
        B = B.at[2, 4].set(1/Cwi)

        C = C.at[0, 0].set(1)

        return A, B, C, D
    
# create model
model = RCModel(name='RC', state_dim=3, input_dim=5, output_dim=1)
print(model.tabulate(jax.random.PRNGKey(0), jnp.zeros((3,)), jnp.zeros((5,))))
params = model.init(jax.random.PRNGKey(0), jnp.zeros((3,)), jnp.zeros((5,)))
print(params)
state = jnp.ones((3,))  # initial state
input = jnp.ones((5,))  # input at the current time step
new_state, output = model.apply(params, state, input)
print(new_state, output)


# another option for linear learnable state space model
class LearnableLinearStateSpaceModel2(nn.Module):
    state_dim: int
    input_dim: int 
    output_dim: int

    def setup(self):
        self.fxx = nn.Dense(features=self.state_dim, use_bias=False)
        self.fxu = nn.Dense(features=self.state_dim, use_bias=False)
        self.fyx = nn.Dense(features=self.output_dim, use_bias=False)
        self.fyu = nn.Dense(features=self.output_dim, use_bias=False)

    def __call__(self, state, input):
        new_state = self.fxx(state) + self.fxu(input)
        output = self.fyx(state) + self.fyu(input)
        return new_state, output

lssm2 = LearnableLinearStateSpaceModel2(name="lssm2", state_dim=3, input_dim=5, output_dim=2)
print(lssm2.tabulate(jax.random.PRNGKey(0), jnp.zeros((3,)), jnp.zeros((5,))))
params = lssm2.init(jax.random.PRNGKey(0), jnp.zeros((3,)), jnp.zeros((5,)))
print(params)
state = jnp.ones((3,))  # initial state
input = jnp.ones((5,))  # input at the current time step
new_state, output = lssm2.apply(params, state, input)
print(new_state, output)

class RCModel(LearnableLinearStateSpaceModel2):

    # need overwrite the learnable parameters using RC parameters
    def setup(self):
        super().setup()
        Cai = self.param('Cai', nn.initializers.ones, ())
        Cwe = self.param('Cwe', nn.initializers.ones, ())
        Cwi = self.param('Cwi', nn.initializers.ones, ())
        Re = self.param('Re', nn.initializers.ones, ())
        Ri = self.param('Ri', nn.initializers.ones, ())
        Rw = self.param('Rw', nn.initializers.ones, ())
        Rg = self.param('Rg', nn.initializers.ones, ())

        A,B,C,D = self._getABCD([Cai, Cwe, Cwi, Re, Ri, Rw, Rg])

        # overwrite the learnable parameters
        self.fxx = lambda x: A @ x
        self.fxu = lambda u: B @ u
        self.fyx = lambda x: C @ x
        self.fyu = lambda u: D @ u

    def __call__(self, state, input):
        # Calculate A, B, C, D with values based on R and C
        return super().__call__(state, input)

    def _getABCD(self, params):
        # unpack
        Cai, Cwe, Cwi, Re, Ri, Rw, Rg = params
        # initialzie
        A = jnp.zeros((3, 3))
        B = jnp.zeros((3, 5))
        C = jnp.zeros((1, 3))
        D = jnp.zeros((1, 5))
        A = A.at[0, 0].set(-1/Cai*(1/Rg+1/Ri))
        A = A.at[0, 2].set(1/(Cai*Ri))
        A = A.at[1, 1].set(-1/Cwe*(1/Re+1/Rw))
        A = A.at[1, 2].set(1/(Cwe*Rw))
        A = A.at[2, 0].set(1/(Cwi*Ri))
        A = A.at[2, 1].set(1/(Cwi*Rw))
        A = A.at[2, 2].set(-1/Cwi*(1/Rw+1/Ri))

        B = B.at[0, 0].set(1/(Cai*Rg))
        B = B.at[0, 1].set(1/Cai)
        B = B.at[0, 2].set(1/Cai)
        B = B.at[1, 0].set(1/(Cwe*Re))
        B = B.at[1, 3].set(1/Cwe)
        B = B.at[2, 4].set(1/Cwi)

        C = C.at[0, 0].set(1)

        return A, B, C, D
     
# create model
model = RCModel(name='RC', state_dim=3, input_dim=5, output_dim=1)
print(model.tabulate(jax.random.PRNGKey(0), jnp.zeros((3,)), jnp.zeros((5,))))
params = model.init(jax.random.PRNGKey(0), jnp.zeros((3,)), jnp.zeros((5,)))
print(params)
state = jnp.ones((3,))  # initial state
input = jnp.ones((5,))  # input at the current time step
new_state, output = model.apply(params, state, input)
print(new_state, output)