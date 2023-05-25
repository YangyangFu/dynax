from typing import Callable, List, Tuple, Union, Optional

import jax.numpy as jnp
import flax.linen as nn

# TODO:
# 1. Add type hints for all the functions
# 2. How to specify the x and u for _fx and _fy, which uses only one argument

class BaseBlockSSM(nn.Module):
    """
    Base class for block state space model. This class is used to define the following general form of state space model:

    .. math::
        \\begin{align}
            rhs &= f_{xx}(x_t) + f_{xu}(u_t) = f_x(xu_t) \\\\
            y_t &= f_{yx}(x_t) + f_{yu}(u_t) = f_y(xu_t)   
        \\end{align}
    
    where, 
        :math:`rhs` is the right hand side of the dynamic equation. For discrete model, it is the next state, and for continuous model, it is the derivative of the state,
        :math:`x_t` is the state at time step :math:`t`, 
        :math:`u_t` is the input at time step :math:`t`, 
        :math:`y_t` is the output at time step :math:`t`, 
        :math:`f_{xx}` is the system function that maps the state to the next state, 
        :math:`f_{xu}` is the input function that maps the input to the next state, 
        :math:`f_{yx}` is the output function that maps the state to the output, 
        :math:`f_{yu}` is the feed-forward function that maps the input to the output, 
        :math:`f_x` is the dynamics function that maps the state and input to the next state, 
        :math:`f_y` is the observation function that maps the state and input to the output.
        
    Args:
        state_dim: dimension of the state
        input_dim: dimension of the input
        output_dim: dimension of the output
    
    
    """
    state_dim: int
    input_dim: int
    output_dim: int

    def setup(self):
        self._fxx: Optional[nn.Module] = None
        self._fxu: Optional[nn.Module] = None
        self._fyx: Optional[nn.Module] = None
        self._fyu: Optional[nn.Module] = None
        self._fx: Optional[nn.Module] = None
        self._fy: Optional[nn.Module] = None

    def __call__(self, x, u):

        if self._fxx and self._fxu:
            rhs = self._fxx(x) + self._fxu(u)
        elif self._fx:
            rhs = self._fx(x, u)
        else:
            raise NotImplementedError("dynamic equation is not implemented")
        
        # combinations of observation equation
        if self._fyx and self._fyu:
            y = self._fyx(x) + self._fyu(u)
        elif self._fy:
            y = self._fy(x, u)
        else:
            raise NotImplementedError("observation equation is not implemented")
        
        return rhs, y
           
# discrete block state space model
class BaseDiscreteBlockSSM(BaseBlockSSM):
    
    def __call__(self, x, u):
        x_next, y = super().__call__(x, u)
        return x_next, y

# continuous block state space model
class BaseContinuousBlockSSM(BaseBlockSSM):

    def __call__(self, x, u):
        dx, y = super().__call__(x, u)
        return dx, y

