import flax.linen as nn

from .base_block_state_space import BaseContinuousBlockSSM

class ContinuousLinearSSM(BaseContinuousBlockSSM):

    #state_dim: int
    #input_dim: int 
    #output_dim: int

    def setup(self):
        self._fxx = self.fxx(self.state_dim)
        self._fxu = self.fxu(self.state_dim)
        self._fyx = self.fyx(self.output_dim)
        self._fyu = self.fyu(self.output_dim)

    def __call__(self, state, input):
        return super().__call__(state, input)
    
    # define linear system function
    class fxx(nn.Module):
        state_dim: int
        def setup(self):
            self.dense = nn.Dense(features=self.state_dim, use_bias=False)

        def __call__(self, x):
            return self.dense(x)

    # define linear input function
    class fxu(nn.Module):
        state_dim: int
        def setup(self):
            self.dense = nn.Dense(features=self.state_dim, use_bias=False)

        def __call__(self, u):
            return self.dense(u)

    # define linear output function
    class fyx(nn.Module):
        output_dim: int
        def setup(self):
            self.dense = nn.Dense(features=self.output_dim, use_bias=False)

        def __call__(self, x):
            return self.dense(x)

    # define linear feed-forward function
    class fyu(nn.Module):
        output_dim: int
        def setup(self):
            self.dense = nn.Dense(features=self.output_dim, use_bias=False)

        def __call__(self, x):
            return self.dense(x)