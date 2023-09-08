from abc import abstractmethod
import flax.linen as nn

class BaseAgent(nn.Module):
    
    @abstractmethod
    def __call__(self, x, *args, **kwargs):
        """ return the action given the state """
        
        raise NotImplementedError