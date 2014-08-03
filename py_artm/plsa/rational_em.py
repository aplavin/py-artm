from time import time
import math
import numpy as np
import numexpr as ne
from ipy_progressbar import ProgressBar
from ..utils import normalize


class PlsaEmRational(object):
    def __init__(self, nwd, T_init, regularizers, quantities):
        self.nwd = nwd
        self.W, self.D = self.nwd.shape
        self.T_init = T_init
        self.regularizers = regularizers
        self.quantities = quantities
        self.progress = []

    def generate_initial(self):
        self.phi = np.random.random((self.W, self.T_init)).astype(np.float32)
        self.phi = normalize(self.phi)

        self.theta = np.random.random((self.T_init, self.D)).astype(np.float32)
        self.theta = normalize(self.theta)

        self.nd = self.nwd.sum(0)
        self.nw = self.nwd.sum(1)
        self.n = self.nwd.sum()

        # preallocate arrays once
        self.pwd = np.empty_like(self.nwd)
        self.npwd = np.empty_like(self.nwd)
        self.phi_sized = np.empty_like(self.phi)
        self.theta_sized = np.empty_like(self.theta)

    def iteration(self):
        if self.itnum == 0:
            np.dot(self.phi, self.theta, out=self.pwd)

        ne.evaluate('where(nwd * pwd > 0, nwd / pwd, 0)', out=self.npwd, local_dict={'nwd': self.nwd, 'pwd': self.pwd})

        dr_dphi = sum(reg.dr_dphi(self) for reg in self.regularizers)

        np.dot(self.npwd, self.theta.T, out=self.phi_sized)
        self.phi_new = self.phi * np.clip(self.phi_sized + dr_dphi, 0, float('inf'))

        dr_dtheta = sum(reg.dr_dtheta(self) for reg in self.regularizers)

        np.dot(self.phi.T, self.npwd, out=self.theta_sized)
        self.theta_new = self.theta * np.clip(self.theta_sized + dr_dtheta, 0, float('inf'))

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
        except KeyboardInterrupt:
            print '<Interrupted>'
