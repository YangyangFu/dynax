from .base import BaseAgent

class Tabular(BaseAgent):

    def __call__(self, x):
        raise NotImplementedError

