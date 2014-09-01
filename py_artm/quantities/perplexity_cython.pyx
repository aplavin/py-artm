#cython: boundscheck=False, wraparound=False, embedsignature=True, cdivision=True
import numpy as np
cimport numpy as np


cdef extern from "mkl.h" nogil:
    float cblas_sdot(int size,
                     float *x, int xstride,
                     float *y, int ystride)


def perplexity_internal_cython(int n, float[:, :] pwd, float[:, :] nwd):
    cdef int* pwd_ = <int*> &pwd[0, 0]
    cdef float* nwd_ = &nwd[0, 0]
    cdef float res = 0
    cdef int i, j
    for i in range(n):
        res += nwd_[i] * (pwd_[i] * <float>8.2629582881927490e-8 - <float>87.989971088)
    return res


def perplexity_sparse(nwd,
                      float[:, ::1] phi,
                      float[::1, :] theta):
    nwd = nwd.tocsr()
    cdef int[:] nwd_indptr = nwd.indptr
    cdef int[:] nwd_indices = nwd.indices
    cdef float[:] nwd_data = nwd.data

    cdef int W = phi.shape[0]
    cdef int T = phi.shape[1]
    cdef int D = theta.shape[1]

    cdef int w, i, d
    cdef int i_0, i_1

    cdef float pwd_val
    cdef float result = 0

    for w in range(W):
        i_0 = nwd_indptr[w]
        i_1 = nwd_indptr[w + 1]

        for i in range(i_0, i_1):
            d = nwd_indices[i]
            pwd_val = cblas_sdot(T, &phi[w, 0], 1, &theta[0, d], 1)
            result += nwd_data[i] * ((<int*>&pwd_val)[0] * <float>8.2629582881927490e-8 - <float>87.989971088)

    return result
