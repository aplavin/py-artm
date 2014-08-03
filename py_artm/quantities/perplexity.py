import math
import numpy as np
import numexpr as ne
from ..plsa.quantity import QuantityBase


class Perplexity(QuantityBase):

    def _items(self, n, nwd, pwd):
        s = np.einsum('ij -> ',
                      ne.evaluate('nwd * (pwd_i * a + b)',
                                  local_dict={'nwd': nwd,
                                              'a': np.float32(8.2629582881927490e-8),
                                              'b': np.float32(-87.989971088),
                                              'pwd_i': pwd.view(np.int32)},
                                  casting='unsafe'))
        # s = perplexity_internal_cython(nwd.size, pwd, nwd)
        # TODO: check if smth like mat[self.nwd == 0] = 0 is needed
        yield ('perplexity', math.exp(-1/n * s))
