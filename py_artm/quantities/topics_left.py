import numpy as np
from ..plsa.quantity import QuantityBase


class TopicsLeft(QuantityBase):

    def _items(self, theta):
        yield ('topics_left', np.count_nonzero(theta.sum(1)))
