import abc 
from typing import Callable, Optional, Tuple, Type, TypeVar

import jax 
import jax.numpy as jnp 
import flax
import flax.linen as nn 

class AbstractSolver(nn.Module):
    """ Base class for all ode solvers
    """

    @property
    @abc.abstractmethod
    def interpolation_cls(self) -> Callable[..., AbstractInterpolation]:
        """ Returns the interpolation class used by this solver
        """
        pass