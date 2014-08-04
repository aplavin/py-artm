import math
import numpy as np
import numexpr as ne
from ..plsa import QuantityBase
from ..utils import public


@public
class Perplexity(QuantityBase):

    def __init__(self, exact=False):
        self.exact = exact

    def _items(self, n, nwd, pwd):
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

        yield ('perplexity', math.exp(-1/n * s))
