import numpy as np
from ..plsa import RegularizerBase
from ..utils import public


@public
class ThetaSparse(RegularizerBase):
    r"""
    ### $\Theta$ Sparsifying regularizer
    * $R = -\sum_{td} \ln \theta_{td}$
    * $\frac{\partial R}{\partial \Theta_{td}} = -\frac{1}{\Theta_{td}}$
    """
    def _dr_dtheta(self, theta):
        return -1/theta
