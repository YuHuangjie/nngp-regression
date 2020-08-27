from ctypes import *
import numpy as np

kernel_lib = cdll.LoadLibrary('./libkernel.so')
# void build_kernel(
#         double *data,
#         const void *indices,
#         const void *indptr,
#         size_t rows,
#         const double *train_x,
#         const double *x,
#         double l_pts,
#         double l_dir,
#         double power_pts,
#         double power_dir)
kernel_lib.build_kernel.restype = None
kernel_lib.build_kernel.argtypes = [
        POINTER(c_double),
        POINTER(c_int64),
        POINTER(c_int64),
        c_size_t,
        POINTER(c_double),
        POINTER(c_double),
        c_double,
        c_double,
        c_double,
        c_double
]


def build_kernel(graph, train_x, x, l_pts, l_dir, gamma_pts, gamma_dir):
        if graph.data.dtype != np.float64 \
          or graph.indices.dtype != np.int64 \
          or graph.indices.dtype != graph.indices.dtype \
          or train_x.dtype != np.float64 \
          or x.dtype != np.float64:
                raise TypeError("miss-matched type")

        data_p = graph.data.ctypes.data_as(POINTER(c_double))
        indices_p = graph.indices.ctypes.data_as(POINTER(c_int64))
        indptr_p = graph.indptr.ctypes.data_as(POINTER(c_int64))
        trainx_p = train_x.ctypes.data_as(POINTER(c_double))
        x_p = x.ctypes.data_as(POINTER(c_double))

        kernel_lib.build_kernel(
                data_p,
                indices_p,
                indptr_p,
                graph.shape[0],
                trainx_p,
                x_p,
                c_double(l_pts),
                c_double(l_dir),
                c_double(gamma_pts),
                c_double(gamma_dir)
        )
