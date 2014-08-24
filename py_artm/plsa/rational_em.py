from time import time
import numpy as np
import numexpr as ne
from ipy_progressbar import ProgressBar
from .regularizer import RegularizerWithCoefficient
from .quantity import QuantityBase
from .stop_condition import StopConditionBase
from ..utils import normalize, public


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
        normalize(self.theta)

        self.nd = self.nwd.sum(0)
        self.nw = self.nwd.sum(1)
        self.n = self.nwd.sum()

        # preallocate arrays once
        self.pwd = np.empty_like(self.nwd)
        self.npwd = np.empty_like(self.nwd)

    def iteration(self):
        if self.itnum == 0:
            np.dot(self.phi, self.theta, out=self.pwd)

        ne.evaluate('where(nwd * pwd > 0, nwd / pwd, 0)', out=self.npwd, local_dict={'nwd': self.nwd, 'pwd': self.pwd})

        dr_dphi = sum(reg.dr_dphi(self) for reg in self.regularizers)

        self.phi_sized = np.dot(self.npwd, self.theta.T)
        self.phi_new = self.phi * np.clip(self.phi_sized + dr_dphi, 0, float('inf'))

        dr_dtheta = 1.0 * self.n / self.D * sum(reg.dr_dtheta(self) for reg in self.regularizers)

        self.theta_sized = np.dot(self.phi.T, self.npwd)
        self.theta_new = self.theta * np.clip(self.theta_sized + dr_dtheta, 0, float('inf'))

        nonzero_t = ~np.all(self.theta_new == 0, axis=1)
        if nonzero_t.sum() < self.theta.shape[0] / 1.1:
            self.phi_new = self.phi_new[:, nonzero_t]
            self.theta_new = self.theta_new[nonzero_t, :]

        self.phi = normalize(self.phi_new)
        self.theta = normalize(self.theta_new)

        np.putmask(self.phi, self.phi < 1e-18, 0)
        np.putmask(self.theta, self.theta < 1e-18, 0)
        np.dot(self.phi, self.theta, out=self.pwd)

    def iterate(self, maxiter, quiet=False):
        self.maxiter = maxiter
        self.generate_initial()
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
