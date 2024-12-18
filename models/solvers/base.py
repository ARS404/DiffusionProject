class BaseSolver(object):
    def __call__(self, *args, **kwds):
        raise NotImplementedError()
    
    def get_name(self):
        return "I have no name :("