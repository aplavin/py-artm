import numpy as np
from ..plsa import RegularizerBase
from ..utils import public


@public
class PhiSparse(RegularizerBase):
    r"""
    ### $\Phi$ Sparsifying regularizer
    * $R = -\sum_{wt} \ln \phi_{wt}$
    * $\frac{\partial R}{\partial \Phi_{td}} = -\frac{1}{\Phi_{wt}}$
    """
    def _dr_dphi(self, phi):
        return -1/phi