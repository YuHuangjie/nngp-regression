from ctypes import *
import numpy as np

linsys_lib = cdll.LoadLibrary('./liblinsys.so')
# void solve_system(
#         const int64_t *ia, 
#         const int32_t *ja, 
#         const float *a, 
#         int n, 
#         const double *mrhs, 
#         int nrhs, 
#         double *x, 
#         int64_t *itercount)
linsys_lib.solve_system.restype = None
linsys_lib.solve_system.argtypes = [
        POINTER(c_int64),
        POINTER(c_int32),
        POINTER(c_float),
        c_int,
        POINTER(c_double),
        c_int,
        POINTER(c_double),
        POINTER(c_int64)
]

def solve_system(ia, ja, a, n, mrhs, x):
        if ia.dtype != np.int64 \
          or ja.dtype != np.int32 \
          or a.dtype != np.float32 \
          or mrhs.dtype != np.float64 \
          or x.dtype != np.float64:
                raise TypeError("miss-matched types")

        ia_p = ia.ctypes.data_as(POINTER(c_int64))
        ja_p = ja.ctypes.data_as(POINTER(c_int32))
        a_p = a.ctypes.data_as(POINTER(c_float))
        mrhs_p = mrhs.ctypes.data_as(POINTER(c_double))
        x_p = x.ctypes.data_as(POINTER(c_double))

        itercount = np.zeros((mrhs.shape[0],),dtype=np.int64)
        itercount_p = itercount.ctypes.data_as(POINTER(c_int64))

        linsys_lib.solve_system(
                ia_p,
                ja_p,
                a_p,
                c_int(n),
                mrhs_p,
                c_int(mrhs.shape[0]),
                x_p,
                itercount_p
        )
        return itercount
