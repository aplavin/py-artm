#cython: boundscheck=False, wraparound=False, embedsignature=True, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport log


def perplexity_internal_cython(int n, np.ndarray[np.float32_t, ndim=2] pwd, np.ndarray[np.float32_t, ndim=2] nwd):
    cdef int* pwd_ = <int*> &pwd[0, 0]
    cdef float* nwd_ = &nwd[0, 0]
    cdef float res = 0
    cdef int i, j
    for i in prange(n, nogil=True):
        res += nwd_[i] * (pwd_[i] * 8.2629582881927490e-8 - 87.989971088)
    return res


def perplexity_sparse(nwd,
                      np.ndarray[np.float32_t, ndim=2] phi,
                      np.ndarray[np.float32_t, ndim=2] theta):
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

    cdef np.float32_t result = 0

    for w in range(W):
        i_0 = nwd_indptr[w]
        i_1 = nwd_indptr[w + 1]

        for i in range(i_0, i_1):
            d = nwd_indices[i]

            pwd_val = 0
            for t in range(T):
                pwd_val += phi[w, t] * theta[t, d]

            # result += nwd_data[i] * log(pwd_val)
            result += nwd_data[i] * ((<int*>&pwd_val)[0] * <float>8.2629582881927490e-8 - <float>87.989971088)

    return result
