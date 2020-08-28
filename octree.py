from ctypes import *
import numpy as np
import scipy.sparse as sparse

octree_lib = cdll.LoadLibrary('./liboctree.so')
# void nn_fit(const double *pts, size_t size, uint32_t bucket_size=32)
octree_lib.nn_fit.restype = None
octree_lib.nn_fit.argtypes = [
        POINTER(c_double),
        c_size_t,
        c_uint32
]
# int64_t nn_radius(const double *query, size_t sz, double radius,
#         int32_t **indices, int64_t **indptr, float **data)
octree_lib.nn_radius.restype = c_int64
octree_lib.nn_radius.argtypes = [
        POINTER(c_double),
        c_size_t,
        c_double,
        POINTER(POINTER(c_int32)),
        POINTER(POINTER(c_int64)),
        POINTER(POINTER(c_float))
]
# void m_dot_v(const int64_t *ia, const int32_t *ja, const float *a, 
#       size_t n, const double *x, double *y, size_t nrhs)
octree_lib.m_dot_v.restype = None
octree_lib.m_dot_v.argtypes = [
        POINTER(c_int64),
        POINTER(c_int32),
        POINTER(c_float),
        c_size_t,
        POINTER(c_double),
        POINTER(c_double),
        c_size_t
]

class graph:
        def __init__(self, indices, indptr, data, nnz, shape):
                self.indices = indices
                self.indptr = indptr
                self.data = data
                self.nnz = nnz
                self.shape = shape

        def dot(self, x):
                if x.dtype != np.float64:
                        raise TypeError('miss-matched type')

                y = np.zeros((self.shape[0], x.shape[1]), dtype=np.float64)
                indices_p = self.indices.ctypes.data_as(POINTER(c_int32))
                indptr_p = self.indptr.ctypes.data_as(POINTER(c_int64))
                data_p = self.data.ctypes.data_as(POINTER(c_float))
                x_p = x.ctypes.data_as(POINTER(c_double))
                y_p = y.ctypes.data_as(POINTER(c_double))
                
                octree_lib.m_dot_v(
                        indptr_p, 
                        indices_p, 
                        data_p, 
                        c_size_t(self.shape[0]),
                        x_p,
                        y_p,
                        c_size_t(x.shape[1])
                )
                return y

class octree:
        def nn_fit(self, train_x, bucket_size=32):
                if train_x.dtype != np.float64:
                        raise TypeError("miss-matched type")

                data_p = train_x.ctypes.data_as(POINTER(c_double))
                octree_lib.nn_fit(
                        data_p, 
                        c_size_t(train_x.shape[0]),
                        c_uint32(bucket_size)
                )
                self.ntrain = train_x.shape[0]

        def nn_radius(self, query, radius):
                if query.dtype != np.float64:
                        raise TypeError("miss-matched type")

                q_p = query.ctypes.data_as(POINTER(c_double))
                indices_p = POINTER(c_int32)()
                indices_pp = pointer(indices_p)
                indptr_p = POINTER(c_int64)()
                indptr_pp = pointer(indptr_p)
                data_p = POINTER(c_float)()
                data_pp = pointer(data_p)
                nquery = query.shape[0]

                nnz = octree_lib.nn_radius(
                        q_p,
                        c_size_t(nquery),
                        c_double(radius),
                        indices_pp,
                        indptr_pp,
                        data_pp
                )
                indices = np.ctypeslib.as_array(indices_p, np.array([nnz,]))
                indptr = np.ctypeslib.as_array(indptr_p, np.array([nquery+1,]))
                data = np.ctypeslib.as_array(data_p, np.array([nnz,]))
                return graph(indices, indptr, data, nnz, [nquery, self.ntrain])
