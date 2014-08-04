import numpy as np
from ..plsa import QuantityBase
from ..utils import public


@public
class TopicsLeft(QuantityBase):

    def _items(self, theta):
        yield ('topics_left', np.count_nonzero(theta.sum(1)))
