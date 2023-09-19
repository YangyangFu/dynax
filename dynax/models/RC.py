import jax.numpy as jnp 
import flax.linen as nn

from dynax.core.base_block_state_space import BaseDiscreteBlockSSM, BaseContinuousBlockSSM

# FIXME:
# 1. The model does not support batch operation.
"""
This is a 4r3c model for zone temperature prediction
ODE::   xdot = Ax + Bd
        y = Cx
States:
        x = [Tai, Twe, Twi]'
Disturbances:
        u = [Tao, qCon_i, qHVAC, qRad_e, qRad_i]'
Output:
        y = [Tai]
===================================================
"""

class Discrete4R3C(BaseDiscreteBlockSSM):
    """
    Discrete thermal 4R3C model for a building zone temperature prediction, which consists of 4 resistors and 3 capacitors. The model is described as follows:

    .. math::
        \\begin{align}
            C_{ai} \\dot{T}_{ai} &= \\frac{1}{R_g}(T_{g} - T_{ai}) + \\frac{1}{R_i}(T_{wi} - T_{ai}) \\\\
            C_{we} \\dot{T}_{we} &= \\frac{1}{R_e}(T_{wi} - T_{we}) + \\frac{1}{R_w}(T_{o} - T_{we}) \\\\
            C_{wi} \\dot{T}_{wi} &= \\frac{1}{R_i}(T_{ai} - T_{wi}) + \\frac{1}{R_w}(T_{we} - T_{wi}) \\\\
        \\end{align}

    where,
        :math:`T_{ai}` is the air temperature inside the zone,
        :math:`T_{we}` is the wall temperature inside the zone,
        :math:`T_{wi}` is the wall temperature outside the zone,
        :math:`T_{g}` is the ground temperature,
        :math:`T_{o}` is the outside temperature,
        :math:`C_{ai}` is the air capacitance,
        :math:`C_{we}` is the wall capacitance inside the zone,
        :math:`C_{wi}` is the wall capacitance outside the zone,
        :math:`R_g` is the ground resistance,
        :math:`R_i` is the inner wall resistance,
        :math:`R_e` is the outer wall resistance,
        :math:`R_w` is the wall resistance.
    
    Args:
        name: name of the model
        state_dim: dimension of the state
        input_dim: dimension of the input
        output_dim: dimension of the output
    
    """

    state_dim: int = 3
    input_dim: int = 5
    output_dim: int = 1

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

        # overwrite the learnable parameters
        self._fxx = self.fxx(self.Cai, self.Cwe, self.Cwi, self.Re, self.Ri, self.Rw, self.Rg)
        self._fxu = self.fxu(self.Cai, self.Cwe, self.Cwi, self.Re, self.Rg)
        self._fyx = self.fyx()
        self._fyu = self.fyu()

    def __call__(self, state, input):
        return super().__call__(state, input)

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


class Continuous4R3C(BaseContinuousBlockSSM):
    """
    Continuous thermal 4R3C model for a building zone temperature prediction, which consists of 4 resistors and 3 capacitors. The model is described as follows:

    .. math::
        \\begin{align}
            C_{ai} \\dot{T}_{ai} &= \\frac{1}{R_g}(T_{g} - T_{ai}) + \\frac{1}{R_i}(T_{wi} - T_{ai}) \\\\
            C_{we} \\dot{T}_{we} &= \\frac{1}{R_e}(T_{wi} - T_{we}) + \\frac{1}{R_w}(T_{o} - T_{we}) \\\\
            C_{wi} \\dot{T}_{wi} &= \\frac{1}{R_i}(T_{ai} - T_{wi}) + \\frac{1}{R_w}(T_{we} - T_{wi}) \\\\
        \\end{align}

    where,
        :math:`T_{ai}` is the air temperature inside the zone,
        :math:`T_{we}` is the wall temperature inside the zone,
        :math:`T_{wi}` is the wall temperature outside the zone,
        :math:`T_{g}` is the ground temperature,
        :math:`T_{o}` is the outside temperature,
        :math:`C_{ai}` is the air capacitance,
        :math:`C_{we}` is the wall capacitance inside the zone,
        :math:`C_{wi}` is the wall capacitance outside the zone,
        :math:`R_g` is the ground resistance,
        :math:`R_i` is the inner wall resistance,
        :math:`R_e` is the outer wall resistance,
        :math:`R_w` is the wall resistance.
    
    Args:
        name: name of the model
        state_dim: dimension of the state
        input_dim: dimension of the input
        output_dim: dimension of the output
    
    """
    state_dim: int = 3
    input_dim: int = 5
    output_dim: int = 1
    
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

        # overwrite the learnable parameters
        self._fxx = self.fxx(self.Cai, self.Cwe, self.Cwi, self.Re, self.Ri, self.Rw, self.Rg)
        self._fxu = self.fxu(self.Cai, self.Cwe, self.Cwi, self.Re, self.Rg)
        self._fyx = self.fyx()
        self._fyu = self.fyu()

    def __call__(self, state, input):
        return super().__call__(state, input)

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

