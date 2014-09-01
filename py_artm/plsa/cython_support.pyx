#cython: boundscheck=False, wraparound=False, embedsignature=True, cdivision=True
import numpy as np
cimport numpy as np


cdef extern from "mkl.h" nogil:
    float cblas_sdot(int size,
                     float *x, int xstride,
                     float *y, int ystride)

    void cblas_saxpy(int size,
                     float alpha,
                     float *x, int xstride,
                     float *y, int ystride)


def calc_nwt(nwd,
             float[:, ::1] phi,
             float[::1, :] theta,
             float[:, ::1] nwt_out):
    nwd = nwd.tocsr()
    cdef int[:] nwd_indptr = nwd.indptr
    cdef int[:] nwd_indices = nwd.indices
    cdef float[:] nwd_data = nwd.data

    cdef int W = phi.shape[0]
    cdef int T = phi.shape[1]
    cdef int D = theta.shape[1]

    cdef int w, t, i, d
    cdef int i_0, i_1
    cdef float pwd_val

    for w in range(W):
        i_0 = nwd_indptr[w]
        i_1 = nwd_indptr[w + 1]

        for t in range(T):
            nwt_out[w, t] = 0

        for i in range(i_0, i_1):
            d = nwd_indices[i]
            pwd_val = cblas_sdot(T, &phi[w, 0], 1, &theta[0, d], 1)
            if pwd_val == 0:
                continue
            cblas_saxpy(T, nwd_data[i] / pwd_val, &theta[0, d], 1, &nwt_out[w, 0], 1)


def calc_ntd(ndw,
             float[:, ::1] phi,
             float[::1, :] theta,
             float[::1, :] ntd_out):
    ndw = ndw.tocsr()
    cdef int[:] ndw_indptr = ndw.indptr
    cdef int[:] ndw_indices = ndw.indices
    cdef float[:] ndw_data = ndw.data

    cdef int W = phi.shape[0]
    cdef int T = phi.shape[1]
    cdef int D = theta.shape[1]

    cdef int w, t, i, d
    cdef int i_0, i_1
    cdef float pwd_val

    for d in range(D):
        i_0 = ndw_indptr[d]
        i_1 = ndw_indptr[d + 1]

        for t in range(T):
            ntd_out[t, d] = 0

        for i in range(i_0, i_1):
            w = ndw_indices[i]
            pwd_val = cblas_sdot(T, &phi[w, 0], 1, &theta[0, d], 1)
            if pwd_val == 0:
                continue
            cblas_saxpy(T, ndw_data[i] / pwd_val, &phi[w, 0], 1, &ntd_out[0, d], 1)
