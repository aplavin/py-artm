from time import time
import numpy as np
import ctypes as c
import scipy.sparse
import numexpr as ne
import warnings
from ipy_progressbar import ProgressBar
from .regularizer import RegularizerWithCoefficient
from .quantity import QuantityBase
from .stop_condition import StopConditionBase
from ..utils import normalize, public

try:
    import pyximport
    pyximport.install()
    import cython_support
except ImportError:
    warnings.warn('Could not import MKL-based code, computations with sparse matrices are disabled.')


try:
    mkl = c.cdll.LoadLibrary("libmkl_rt.so")
except OSError:
    warnings.warn('Could not import MKL, using dot() from numpy.')
    dot = np.dot
else:
    def dot(A, B, C=None):
        assert A.shape[1] == B.shape[0]
        if C is None:
            C = np.empty((A.shape[0], B.shape[1]), dtype=np.float32)

        Order = 101  # 101 for row-major, 102 for column major data structures
        alpha = 1
        beta = 0

        m = A.shape[0]
        k = A.shape[1]
        n = B.shape[1]
        ldc = n

        if A.flags.c_contiguous:
            TransA = 111 # 111 for no transpose, 112 for transpose, and 113 for conjugate transpose
            lda = k
        elif A.flags.f_contiguous:
            TransA = 112
            lda = m

        if B.flags.c_contiguous:
            TransB = 111 # 111 for no transpose, 112 for transpose, and 113 for conjugate transpose
            ldb = n
        elif B.flags.f_contiguous:
            TransB = 112
            ldb = k

        mkl.cblas_sgemm( c.c_int(Order), c.c_int(TransA), c.c_int(TransB),
                         c.c_int(m), c.c_int(n), c.c_int(k),
                         c.c_float(alpha),
                         A.ctypes.data_as(c.POINTER(c.c_float)), c.c_int(lda),
                         B.ctypes.data_as(c.POINTER(c.c_float)), c.c_int(ldb),
                         c.c_float(beta),
                         C.ctypes.data_as(c.POINTER(c.c_float)), c.c_int(ldc) )
        return C


# def dot(A, B, C=None):
#     assert A.shape[1] == B.shape[0]
#     if C is None:
#         C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
#     for i in range(A.shape[0]):
#         for j in range(B.shape[1]):
#             for k in range(A.shape[1]):
#                 C[i, j] += A[i, k] * B[k, j]
#     return C


@public
class PlsaEmRational(object):

    """
    ### Non-matrix form
    * zero all $n_{wt}, n_{dt}, n_{t}$
    * for all $d, w$:
      * $Z = \sum_t \phi_{wt} \theta_{td}$
      * for all $t$:
        * increase $n_{wt}, n_{dt}, n_{t}$ by $\delta = n_{dw} \phi_{wt} \theta_{td} / Z$
    * $\phi_{wt} \propto \left( n_{wt} + \phi_{wt} \frac{\partial R}{\partial \phi_{wt}} \right)_+$
    * $\theta_{td} \propto \left( n_{dt} + \theta_{td} \frac{\partial R}{\partial \theta_{td}} \right)_+$

    ### Matrix form
    * $P_{wd} = \Phi_{wt} \Theta_{td}$
    * $P'_{wd} = N_{wd} / P_{wd}$
    * $N_{wt} = \Phi_{wt} \cdot P'_{wd} \Theta_{td}^T$
    * $N_{td} = \Theta_{td} \cdot \Phi_{wt}^T P'_{wd}$
    * $\Phi_{wt} \propto \left( N_{wt} + \Phi_{wt} \cdot \frac{\partial R}{\partial \Phi_{wt}} \right)_+ = \Phi_{wt} \cdot \left( P'_{wd} \Theta_{td}^T + \frac{\partial R}{\partial \Phi_{wt}} \right)_+$
    * $\Theta_{td} \propto \left( N_{td} + \Theta_{td} \cdot \frac{\partial R}{\partial \Theta_{td}} \right)_+ = \Theta_{td} \cdot \left( \Phi_{wt}^T P'_{wd} + \frac{\partial R}{\partial \Theta_{td}} \right)_+$
    """

    def __init__(self, nwd, T_init, modifiers):
        self.nwd = nwd
        self.W, self.D = self.nwd.shape
        self.T_init = T_init
        self.issparse = scipy.sparse.issparse(self.nwd)

        def type_checker(t):
            return lambda obj: isinstance(obj, t)
        self.regularizers = filter(type_checker(RegularizerWithCoefficient), modifiers)
        self.quantities = filter(type_checker(QuantityBase), modifiers)
        self.stop_conditions = filter(type_checker(StopConditionBase), modifiers)

        self.progress = []

    def generate_initial(self):
        self.phi = np.random.random((self.W, self.T_init)).astype(np.float32)
        normalize(self.phi)

        self.theta = np.random.random((self.T_init, self.D)).astype(np.float32)
        self.theta = np.asfortranarray(self.theta)
        normalize(self.theta)

        self.nd = np.squeeze(np.array(self.nwd.sum(0)))
        self.nw = np.squeeze(np.array(self.nwd.sum(1)))
        self.n = self.nwd.sum()

        # preallocate arrays once
        if not self.issparse:
            self.pwd = np.empty_like(self.nwd)
            self.npwd = np.empty_like(self.nwd)
        else:
            self.pwd = None
            self.npwd = None

    def calc_multipliers(self):
        if not self.issparse:
            ne.evaluate('where(nwd * pwd > 0, nwd / pwd, 0)', out=self.npwd, local_dict={'nwd': self.nwd, 'pwd': self.pwd})
            self.phi_sized = dot(self.npwd, self.theta.T)
            self.theta_sized = dot(self.phi.T, self.npwd)
        else:
            self.phi_sized = np.empty_like(self.phi)
            self.theta_sized = np.empty_like(self.theta)
            cython_support.calc_nwt(
                nwd=self.nwd,
                phi=self.phi,
                theta=self.theta,
                nwt_out=self.phi_sized)
            cython_support.calc_ntd(
                ndw=self.nwd.T,
                phi=self.phi,
                theta=self.theta,
                ntd_out=self.theta_sized)

    def iteration(self):
        if not self.issparse and self.itnum == 0:
            dot(self.phi, self.theta, self.pwd)

        self.calc_multipliers()

        dr_dphi = sum(reg.dr_dphi(self) for reg in self.regularizers)
        self.phi_new = self.phi * np.clip(self.phi_sized + dr_dphi, 0, float('inf'))

        dr_dtheta = 1.0 * self.n / self.D * sum(reg.dr_dtheta(self) for reg in self.regularizers)
        self.theta_new = self.theta * np.clip(self.theta_sized + dr_dtheta, 0, float('inf'))

        nonzero_t = ~np.all(self.theta_new == 0, axis=1)
        if nonzero_t.sum() > 0 and nonzero_t.sum() < self.theta.shape[0] / 1.1:
            self.phi_new = np.ascontiguousarray(self.phi_new[:, nonzero_t])
            self.theta_new = np.asfortranarray(self.theta_new[nonzero_t, :])

        self.phi = normalize(self.phi_new)
        self.theta = np.asfortranarray(normalize(self.theta_new))

        np.putmask(self.phi, self.phi < 1e-18, 0)
        np.putmask(self.theta, self.theta < 1e-18, 0)

        if not self.issparse:
            dot(self.phi, self.theta, self.pwd)

    def iterate(self, maxiter, prefunc=lambda s: s, quiet=False):
        self.maxiter = maxiter
        self.generate_initial()
        prefunc(self)
        try:
            pb = ProgressBar(range(maxiter), quiet=quiet >= 2, title='PLSA EM', key='plsa_em')

            for self.itnum in pb:
                start_time = time()
                self.iteration()
                end_time = time()

                progress_items = [
                    ('iteration', self.itnum),
                    ('time', end_time - start_time),
                ] + [(k, v) for q in self.quantities for k, v in q.items(self)]
                self.progress.append(progress_items)

                pb.set_extra_text('; '.join(['%s: %s' % (k.capitalize(), v) for k, v in progress_items[2:]]))
                if not quiet:
                    print '; '.join(['%s: %s' % (k.capitalize(), v) for k, v in progress_items])

                if any(sc.is_stop(self, dict(progress_items)) for sc in self.stop_conditions):
                    break
        except KeyboardInterrupt:
            print '<Interrupted>'
