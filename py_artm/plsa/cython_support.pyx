#cython: boundscheck=False, wraparound=False, embedsignature=True, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange


def calc_nwt(nwd,
             np.ndarray[np.float32_t, ndim=2] phi,
             np.ndarray[np.float32_t, ndim=2] theta,
             np.ndarray[np.float32_t, ndim=2] nwt_out):
    nwd = nwd.tocsr()
    cdef np.ndarray[np.int32_t, ndim=1] nwd_indptr = nwd.indptr
    cdef np.ndarray[np.int32_t, ndim=1] nwd_indices = nwd.indices
    cdef np.ndarray[np.float32_t, ndim=1] nwd_data = nwd.data

    theta = np.asfortranarray(theta)

    cdef int W = phi.shape[0]
    cdef int T = phi.shape[1]
    cdef int D = theta.shape[1]

    cdef int w, t, i, d
    cdef int i_0, i_1
    cdef np.float32_t pwd_val

    for w in range(W):
        i_0 = nwd_indptr[w]
        i_1 = nwd_indptr[w + 1]

        for t in range(T):
            nwt_out[w, t] = 0

        for i in range(i_0, i_1):
            d = nwd_indices[i]
            pwd_val = 0
            for t in range(T):
                pwd_val += phi[w, t] * theta[t, d]
            if pwd_val == 0:
                continue
            for t in range(T):
                nwt_out[w, t] += nwd_data[i] * theta[t, d] / pwd_val


def calc_ntd(ndw,
             np.ndarray[np.float32_t, ndim=2] phi,
             np.ndarray[np.float32_t, ndim=2] theta,
             np.ndarray[np.float32_t, ndim=2] ntd_out):
    ndw = ndw.tocsr()
    cdef np.ndarray[np.int32_t, ndim=1] ndw_indptr = ndw.indptr
    cdef np.ndarray[np.int32_t, ndim=1] ndw_indices = ndw.indices
    cdef np.ndarray[np.float32_t, ndim=1] ndw_data = ndw.data

    theta = np.asfortranarray(theta)

    cdef int W = phi.shape[0]
    cdef int T = phi.shape[1]
    cdef int D = theta.shape[1]

    cdef int w, t, i, d
    cdef int i_0, i_1
    cdef np.float32_t pwd_val

    for d in range(D):
        i_0 = ndw_indptr[d]
        i_1 = ndw_indptr[d + 1]

        for t in range(T):
            ntd_out[t, d] = 0

        for i in range(i_0, i_1):
            w = ndw_indices[i]
            pwd_val = 0
            for t in range(T):
                pwd_val += phi[w, t] * theta[t, d]
            if pwd_val == 0:
                continue
            for t in range(T):
                ntd_out[t, d] += ndw_data[i] * phi[w, t] / pwd_val
