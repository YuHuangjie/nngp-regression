from ctypes import *
import numpy as np

kernel_lib = cdll.LoadLibrary('./libkernel.so')
# void build_kernel(
#         double *data,
#         const int *indices,
#         const int *indptr,
#         size_t rows,
#         const double *train_x,
#         const double *x,
#         double l_pos,
#         double power)
kernel_lib.build_kernel.restype = None
kernel_lib.build_kernel.argtypes = [
        POINTER(c_double),
        c_void_p,
        c_void_p,
        c_size_t,
        c_size_t,
        POINTER(c_double),
        POINTER(c_double),
        c_double,
        c_double
]


def build_kernel(graph, train_x, x, l_pos, power):
        if graph.data.dtype != np.float64 \
          or (graph.indices.dtype != np.int32 and graph.indices.dtype != np.int64) \
          or graph.indices.dtype != graph.indices.dtype \
          or train_x.dtype != np.float64 \
          or x.dtype != np.float64:
                raise TypeError("miss-matched type")

        data_p = graph.data.ctypes.data_as(POINTER(c_double))
        indices_p = graph.indices.ctypes.data_as(c_void_p)
        indptr_p = graph.indptr.ctypes.data_as(c_void_p)
        trainx_p = train_x.ctypes.data_as(POINTER(c_double))
        x_p = x.ctypes.data_as(POINTER(c_double))

        kernel_lib.build_kernel(
                data_p,
                indices_p,
                indptr_p,
                c_size_t(4 if graph.indices.dtype==np.int32 else 8),
                graph.shape[0],
                trainx_p,
                x_p,
                c_double(l_pos),
                c_double(power)
        )
        return graph