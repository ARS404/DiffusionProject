from .base import BaseSolver

class EulerSolver(BaseSolver):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwds):
        pass

    def get_name(self):
        return "Euler"