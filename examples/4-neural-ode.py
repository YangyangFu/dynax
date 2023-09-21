"""Train a neural ODE based on given data.

The building dynamics is described by a linear state space model in the following format:

$$ \dot x_t =  fx(x_in) $$
$$ y_t = Cx_t $$

"""
import jax 
import jax.numpy as jnp 
import flax.linen as nn

from dynax.core.base_block_state_space import BaseContinuousBlockSSM


class NeuralRC(BaseContinuousBlockSSM):
    state_dim: int 
    input_dim: int 
    output_dim: int = 1

    def setup(self):
        super().setup()
        self._fx = self.fx(output_dim = self.output_dim)

    def __call__(self, states, inputs):
        rhs = self._fx(inputs)
        y = 0
        
        return rhs, y  
    

    class fx(nn.Module):
        output_dim: int

        def setup(self):
            self.dense1 = nn.Dense(32)
            self.activation1 = nn.activation.relu()
            self.dense2 = nn.Dense(self.output_dim)
        
        def __call__(self, inputs):
            x = self.dense1(inputs)
            x = self.actiatio1(x)
            x = self.dense2(x)
            return x