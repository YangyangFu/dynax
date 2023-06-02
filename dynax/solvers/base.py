import abc 
from typing import Callable, Optional, Tuple, Type, TypeVar

import jax 
import jax.numpy as jnp 
import flax
import flax.linen as nn 

from ..utils import AbstractInterpolation

class AbstractSolver(nn.Module):
    """ Base class for all ode solvers
    """

    @property
    @abc.abstractmethod
    def interpolation_cls(self) -> Callable[..., AbstractInterpolation]:
        """ Returns the interpolation class used by this solver to interpolate the solution between steps
        """
    
    def order(self) -> Optional[int]:
        """ Order of the solver
        """
        return None
    
    def strong_order(self) -> Optional[int]:
        """ Strong order of the solver to solve stochastic differential equations
        """
        return None
    
    def error_order(self) -> Optional[int]:
        # reference: https://github.com/patrick-kidger/diffrax/blob/main/diffrax/solver/base.py
        """Order of the error estimate used for adaptive stepping.

        """
        return None
    
    def init_(self, rhs, t0, t1, x0) -> SolverState:
        """
        Args:
            rhs: callable that takes in a time and state and returns the derivative of the state
            t0: initial time
            t1: final time
            x0: initial state
            
            Returns:
                state: the initial state of the solver
            
            The initial solve state, which should be used the first time `step` is called
            """

        return None

    def step(self):
        pass