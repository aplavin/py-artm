import sys
import inspect


def normalize(mat):
    """ Normalize columns so that each of them sum up to 1 """
    norms = mat.sum(0)
    norms[norms == 0] = 1
    mat /= norms
    return mat


def call_ignore_extra_args(func_base):
    def wrapper(self, obj, kwargs={}):
        func = getattr(self, func_base.__name__)
        internal_argnames = set(inspect.getargspec(func).args) - {'self'}
        internal_kwargs = {name: (getattr(obj, name) if hasattr(obj, name) else kwargs[name])
                           for name in internal_argnames}
        return func(**internal_kwargs)
    return wrapper


def public(f):
    """"Use a decorator to avoid retyping function/class names.

    * Based on an idea by Duncan Booth:
    http://groups.google.com/group/comp.lang.python/msg/11cbb03e09611b8a
    * Improved via a suggestion by Dave Angel:
    http://groups.google.com/group/comp.lang.python/msg/3d400fb22d8a42e1
    """
    all = sys.modules[f.__module__].__dict__.setdefault('__all__', [])
    if f.__name__ not in all:  # Prevent duplicates if run from an IDE.
        all.append(f.__name__)
    return f
