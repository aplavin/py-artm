import math
import numpy as np
import numexpr as ne
import warnings
from ..plsa import QuantityBase
from ..utils import public

try:
  import pyximport
  pyximport.install()
  import perplexity_cython
except ImportError:
    warnings.warn('Could not import MKL-based code, computations with sparse matrices are disabled.')


@public
class Perplexity(QuantityBase):

    def __init__(self, exact=False):
        self.exact = exact

    def _items(self, n, nwd, phi, theta, pwd, issparse):
        if not issparse:
            if self.exact:
                s = ne.evaluate('where(nwd == 0, 0, nwd * log(pwd))').sum()
            else:
                mat = ne.evaluate('nwd * (pwd_i * a + b)',
                                  local_dict={'nwd': nwd,
                                              'a': np.float32(8.2629582881927490e-8),
                                              'b': np.float32(-87.989971088),
                                              'pwd_i': pwd.view(np.int32)},
                                  casting='unsafe')
                s = np.einsum('ij -> ', mat)
        else:
            # pwd is None here
            s = perplexity_cython.perplexity_sparse(nwd, phi, theta)

        yield ('perplexity', math.exp(-1 / n * s))
