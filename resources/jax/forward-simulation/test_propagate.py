from typing import Callable, List, Tuple, Union, Optional

import jax 
import jax.numpy as jnp
import flax.linen as nn 

class LSSM(nn.Module):
    A: Optional[jnp.array]
    Bu: Optional[jnp.array]
    Bd: Optional[jnp.array]
    C: Optional[jnp.array]
    D: Optional[jnp.array]

    def setup(self):
        self.state_dim = self.A.shape[0]
        self.input_dim = self.Bu.shape[1]
        self.output_dim = self.C.shape[0]
        
    def __call__(self, x: jnp.array, u: jnp.array, d: jnp.array):
        """ state-space model
        """
        x = self.A @ x + self.Bu @ u + self.Bd @ d
        y = self.C @ x + self.D @ u
        return x, y

class RC(LSSM):
    name: str = 'RC'
    params: jnp.array

    def setup(self):
        # initialize matrices
        self.A, self.Bu, self.Bd, self.C, self.D = self._getABCD()
        super().setup()

    def _getABCD(self):
        # unpack
        Cai, Cwe, Cwi, Re, Ri, Rw, Rg = self.params
        # initialzie
        A = jnp.zeros((3, 3))
        Bu = jnp.zeros((3, 1))
        Bd = jnp.zeros((3, 4))
        C = jnp.zeros((2, 3))
        D = jnp.zeros((2, 1))

        # set matrix
        A.at[0, 0].set(-1/Cai*(1/Rg+1/Ri))
        A.at(0, 2).set(1/(Cai*Ri))
        A.at[1, 1].set(-1/Cwe*(1/Re+1/Rw))
        A.at[1, 2].set(1/(Cwe*Rw))
        A.at[2, 0].set(1/(Cwi*Ri))
        A.at[2, 1].set(1/(Cwi*Rw))
        A.at[2, 2].set(-1/Cwi*(1/Rw+1/Ri))
        
        Bu.at[0, 0].set(1/Cai)

        Bd.at[0, 0].set(1/(Cai*Rg))
        Bd.at[0, 1].set(1/Cai)
        Bd.at[1, 0].set(1/(Cwe*Re))
        Bd.at[1, 2].set(1/Cwe)
        Bd.at[2, 3].set(1/Cwi)

        C.at[0, 0].set(1)

        D.at[1, 0].set(1)

        return A, Bu, Bd, C, D
    

# create model
model = RC(name='RC', params=jnp.array([1, 1, 1, 1, 1, 1, 1]))
print(model.params)