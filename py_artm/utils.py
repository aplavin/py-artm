import inspect


def normalize(mat):
    """ Normalize columns so that each of them sum up to 1 """
    norms = mat.sum(0)
    norms[norms == 0] = 1
    return mat / norms


def call_ignore_extra_args(func_base):
    def wrapper(self, obj, kwargs={}):
        func = getattr(self, func_base.__name__)
        internal_argnames = set(inspect.getargspec(func).args) - {'self'}
        internal_kwargs = {name: (getattr(obj, name) if hasattr(obj, name) else kwargs[name])
                           for name in internal_argnames}
        return func(**internal_kwargs)
    return wrapper
