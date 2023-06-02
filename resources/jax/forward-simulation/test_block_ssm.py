import jax.numpy as jnp
from flax import linen as nn
import jax 

# another option for linear learnable state space model
class LearnableLinearStateSpaceModel2(nn.Module):
    state_dim: int
    input_dim: int 
    output_dim: int

    def setup(self):
        self._fxx = nn.Dense(features=self.state_dim, use_bias=False)
        self._fxu = nn.Dense(features=self.state_dim, use_bias=False)
        self._fyx = nn.Dense(features=self.output_dim, use_bias=False)
        self._fyu = nn.Dense(features=self.output_dim, use_bias=False)

    def __call__(self, state, input):
        new_state = self._fxx(state) + self._fxu(input)
        output = self._fyx(state) + self._fyu(input)
        return new_state, output

lssm2 = LearnableLinearStateSpaceModel2(name="lssm2", state_dim=3, input_dim=5, output_dim=2)
print(lssm2.tabulate(jax.random.PRNGKey(0), jnp.zeros((3,)), jnp.zeros((5,))))
params = lssm2.init(jax.random.PRNGKey(0), jnp.zeros((3,)), jnp.zeros((5,)))
print(params)
state = jnp.ones((3,))  # initial state
input = jnp.ones((5,))  # input at the current time step
new_state, output = lssm2.apply(params, state, input)
print(new_state, output)


class RCModel(nn.Module):#(LearnableLinearStateSpaceModel2):
    # need overwrite the learnable parameters using RC parameters
    def setup(self):
        super().setup()
        self.Cai = self.param('Cai', nn.initializers.ones, ())
        self.Cwe = self.param('Cwe', nn.initializers.ones, ())
        self.Cwi = self.param('Cwi', nn.initializers.ones, ())
        self.Re = self.param('Re', nn.initializers.ones, ())
        self.Ri = self.param('Ri', nn.initializers.ones, ())
        self.Rw = self.param('Rw', nn.initializers.ones, ())
        self.Rg = self.param('Rg', nn.initializers.ones, ())

        #A,B,C,D = self._getABCD([Cai, Cwe, Cwi, Re, Ri, Rw, Rg])

        # overwrite the learnable parameters
        self._fxx = self.fxx(self.Cai, self.Cwe, self.Cwi, self.Re, self.Ri, self.Rw, self.Rg)
        self._fxu = self.fxu(self.Cai, self.Cwe, self.Cwi, self.Re, self.Rg)
        self._fyx = self.fyx()
        self._fyu = self.fyu()

    #def __call__(self, state, input):
        # Calculate A, B, C, D with values based on R and C
    #    return super().__call__(state, input)
    def __call__(self, state, input):
        new_state = self._fxx(state) + self._fxu(input)
        output = self._fyx(state) + self._fyu(input)
        return new_state, output
    
    class fxx(nn.Module):
        Cai: float
        Cwe: float
        Cwi: float
        Re: float
        Ri: float
        Rw: float
        Rg: float
        def setup(self):
            A = jnp.zeros((3, 3))
            A = A.at[0, 0].set(-1/self.Cai*(1/self.Rg+1/self.Ri))
            A = A.at[0, 2].set(1/(self.Cai*self.Ri))
            A = A.at[1, 1].set(-1/self.Cwe*(1/self.Re+1/self.Rw))
            A = A.at[1, 2].set(1/(self.Cwe*self.Rw))
            A = A.at[2, 0].set(1/(self.Cwi*self.Ri))
            A = A.at[2, 1].set(1/(self.Cwi*self.Rw))
            A = A.at[2, 2].set(-1/self.Cwi*(1/self.Rw+1/self.Ri))
            self.A = A

        def __call__(self, x):
            return self.A @ x
    
    class fxu(nn.Module):
        Cai: float
        Cwe: float
        Cwi: float
        Re: float
        Rg: float

        def setup(self):
            B = jnp.zeros((3, 5))
            B = B.at[0, 0].set(1/(self.Cai*self.Rg))
            B = B.at[0, 1].set(1/self.Cai)
            B = B.at[0, 2].set(1/self.Cai)
            B = B.at[1, 0].set(1/(self.Cwe*self.Re))
            B = B.at[1, 3].set(1/self.Cwe)
            B = B.at[2, 4].set(1/self.Cwi)
            self.B = B 

        def __call__(self, u):
            return self.B @ u
    
    class fyx(nn.Module):
        def setup(self):
            C = jnp.zeros((1, 3))
            C = C.at[0, 0].set(1)
            self.C = C 

        def __call__(self, x):
            return self.C @ x
    
    class fyu(nn.Module):
        def setup(self):
            self.D = jnp.zeros((1, 5))

        def __call__(self, u):
            return self.D @ u


# create model
model = RCModel(name='RC')
print(model.tabulate(jax.random.PRNGKey(0), jnp.zeros((3,)), jnp.zeros((5,))))
params = model.init(jax.random.PRNGKey(0), jnp.zeros((3,)), jnp.zeros((5,)))
print(params)
state = jnp.ones((3,))  # initial state
input = jnp.ones((5,))  # input at the current time step
new_state, output = model.apply(params, state, input)
print(new_state, output)