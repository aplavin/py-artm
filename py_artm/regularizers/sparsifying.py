import numpy as np
from ..plsa import RegularizerBase


class PhiSparsifying(RegularizerBase):
    def _dr_dphi(self, phi):
        return -1/phi


class ThetaLineSparsifying(RegularizerBase):
    def _dr_dtheta(self, T_init, nd, n, theta):
        return -1/T_init * nd / np.dot(theta, nd.reshape((-1, 1)))


class ThetaSparsifying(RegularizerBase):
    def _dr_dtheta(self, theta):
        return -1/theta
