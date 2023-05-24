from typing import Callable, List, Tuple, Union
#from jaxtyping import Array, Float, Pytree
import jax.numpy as jnp
import flax.linen as nn

class BaseBlockSSM(nn):
    """ base block state-space model as follows

        $$ rhs = f_{xx}(x_t) + f_{xu}(u_t) $$
        $$ y_t = f_{yx}(x_t) + f_{yu}(u_t) $$

        or 

        $$ rhs = f_x(x_t, u_t) $$
        $$ y_t = f_y(x_t, u_t) $$

        $rhs$: right hand side of dynamic equation, could be $x_{t+1}$ for discrete model or $\dot x_t$ for continuous model.
        $x_t$: state variables
        $u_t$: input variables, including control inputs and disturbances if any. By convention, $u_t = [u_{c,t}, u_{d,t}]$ by stacking control inputs and disturbances together.
        $y_t$: output variables 
        $f_{xx}$: system block for state variables, determines how the state variables evolve
        $f_{xu}$: control block, determines how the system inputs affect the state variables
        $f_{yx}$: output block, determines the relationship between state variables and output variables
        $f_{yu}$: feed-forward block, allows for the system input to affect the system output directly. A basic feedback system do not have a feed-forward element.

    """
    _fx: Union[None, Callable, jnp.array, nn.Module]
    _fy: Union[None, Callable, jnp.array, nn.Module]

    def __init__(self, 
                 fxx: Union[None, Callable, jnp.array, nn.Module], 
                 fxu: Union[None, Callable, jnp.array, nn.Module], 
                 fyx: Union[None, Callable, jnp.array, nn.Module], 
                 fyu: Union[None, Callable, jnp.array, nn.Module], 
                 fx: Union[None, Callable, jnp.array, nn.Module] = None, 
                 fy: Union[None, Callable, jnp.array, nn.Module] = None
                 ):
        
        super().__init__()
        self.fxx = fxx
        self.fxu = fxu
        self.fyx = fyx
        self.fyu = fyu
        self.fx = fx
        self.fy = fy

        self._is_valid()

    def _is_valid(self):
        """ check the validity of the model 

        1. dynamic equation can either be an essemble block $f_x$ or seprate blocks made of ($f_{xx}, $f_{xu})
        2. observation equation can either be $f_y$ or ($f_{yx}$, $f_{yu}$)

        """
        # check block combinations
        valid_dyn = (self.fx is not None) or (self.fxx is not None or self.fxu is not None)
        valid_obs = (self.fy is not None) or (self.fyx is not None or self.fyu is not None)

        if not valid_dyn:
            raise ValueError("dynamic equation is not valid. Please check the model definition.")
        if not valid_obs:
            raise ValueError("observation equation is not valid. Please check the model definition.")
        
        # if valid then, prepare for forward simulation
        def _fx(x, u):
            if self.fx is not None:
                return self.fx
            
            fx = lambda x, u: self.fxx(x) if self.fxx is not None else 0 + self.fxu(u) if self.fxu is not None else 0
            return fx

        def _fy(x, u):
            if self.fy is not None:
                return self.fy
            
            fy = lambda x, u: self.fyx(x) if self.fyx is not None else 0 + self.fyu(u) if self.fyu is not None else 0
            return fy        

        self._fx = _fx
        self._fy = _fy

        # check block shapes


        return

    def __call__(self, x, u):
        # forward simulation
        rhs = self._fx(x, u)
        y = self._fy(x, u)

        return rhs, y

class LinearDiscreteSSM(BaseBlockSSM):
    
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        super().__init__(fxx = lambda x: A @ x
                fxu = lambda u: B @ u,
                fyx = lambda x: C @ x,
                fyu = lambda u: D @ u,
                ))
        pass 

         
    






