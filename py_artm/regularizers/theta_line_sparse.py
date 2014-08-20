import numpy as np
from ..plsa import RegularizerBase
from ..utils import public


@public
class ThetaLineSparse(RegularizerBase):
    r"""
    ### Line sparsifying regularizer
    * $R = \frac{1}{|T|} \sum_t \ln \frac{1/|T|}{\sum_d \theta_{td} n_d / n}$
    * $\frac{\partial R}{\partial \Theta_{td}} = -\frac{1}{|T|} \frac{n_d}{\sum_{d'} \theta_{td'} n_{d'}} = -\frac{1}{|T|} \frac{N_d}{(\Theta N)_t}$

    ### Another version
    * $R = \sum_t \left( \sum_d \theta_{td} n_d / n \right) \ln \frac{\sum_d \theta_{td} n_d / n}{1/|T|} \propto \sum_t \left( \sum_d \theta_{td} n_d \right) (\ln \sum_d \theta_{td} n_d - \ln n/|T|)$
    * $\frac{\partial R}{\partial \Theta_{td}} = n_d \left( 1 + \ln \sum_{d'} \theta_{td'} n_{d'} - \ln \frac{n}{T} \right) = N_d \left( 1 + \ln (\Theta N)_t - \ln \frac{n}{T} \right)$
    """
    def _dr_dtheta(self, T_init, nd, D, theta):
        return -1.0 * D / T_init * nd / np.dot(theta, nd.reshape((-1, 1)))