from base import BaseSolver

class EulerSolver(BaseSolver):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, *args, **kwds):
        pass

    def get_name(self):
        return "Euler"