from .base import BaseSolver


class DPMSolver(BaseSolver):
    def __init__(self):
        super().__init__()

    def __call__(self, net, *args, **kwds):
        return super().__call__(net, *args, **kwds)
    
    def get_name(self):
        return "DPM"