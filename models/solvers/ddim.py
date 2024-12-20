from .base import BaseSolver


class DDIMSolver(BaseSolver):
    def __init__(self):
        super().__init__()

    def __call__(self, net, *args, **kwds):
        pass

    def get_name(self):
        return "DDM"