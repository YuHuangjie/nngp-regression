# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interpolate in NNGP grid.
"""

from ctypes import *
import numpy as np

def interp_lin(x, y, xp, log_spacing=False):
    """Linearly interpolate. 

    x is evenly spaced grid coordinates, with values y,
    xp are the locations to which to interpolate.
    x and xp must be 1d tensors.
    """
    if log_spacing:
        x = np.log(x)
        xp = np.log(xp)

    spacing = x[1] - x[0]
    grid = (xp - x[0]) / spacing
    ind1 = grid.astype(np.int64)
    ind2 = ind1 + 1
    max_ind = x.shape[0]
    # set top and bottom indices identical if extending past end of range
    ind2 = np.minimum(max_ind - 1, ind2)

    weight1 = np.abs(xp - x[ind1]) / spacing
    weight2 = np.abs(xp - x[ind2]) / spacing
    if log_spacing:
      weight1 = np.exp(weight1)
      weight2 = np.exp(weight2)

    weight1 = 1. - np.reshape(weight1, [-1] + [1] * (len(y.shape) - 1))
    weight2 = 1. - np.reshape(weight2, [-1] + [1] * (len(y.shape) - 1))

    weight_sum = weight1 + weight2
    weight1 /= weight_sum
    weight2 /= weight_sum

    y1 = y[ind1]
    y2 = y[ind2]
    yp = y1 * weight1 + y2 * weight2
    return yp

interp_lib = cdll.LoadLibrary('./libinterp.so')
# void recursive_kernel(
#         const double *x, 
#         size_t sz_x, // no. elements in x
#         const double *y, 
#         size_t sz_y, // no. elements in y
#         const double *z, 
#         float *yp_in_out, 
#         size_t sz_yp, 
#         size_t depth,
#         double weight_var,
#         double bias_var,
#         const double *layer_qaa)
interp_lib.recursive_kernel.restype = c_double
interp_lib.recursive_kernel.argtypes = [
    POINTER(c_double), 
    c_size_t,
    POINTER(c_double), 
    c_size_t, 
    POINTER(c_double), 
    POINTER(c_double), 
    c_size_t,
    c_size_t,
    c_double,
    c_double,
    POINTER(c_double)]

def recursive_kernel(x, y, z, yp, depth, weight_var, bias_var, layer_qaa):
    if x.dtype != np.float64 or y.dtype != np.float64 \
        or z.dtype != np.float64 or yp.dtype != np.float64 \
        or layer_qaa.dtype != np.float64:
        raise TypeError("miss-matched type")

    return interp_lib.recursive_kernel(
        x.ctypes.data_as(POINTER(c_double)), 
        x.size, 
        y.ctypes.data_as(POINTER(c_double)), 
        y.size,
        z.ctypes.data_as(POINTER(c_double)),
        yp.ctypes.data_as(POINTER(c_double)), 
        c_size_t(yp.size),
        c_size_t(depth), 
        c_double(weight_var),
        c_double(bias_var),
        layer_qaa.ctypes.data_as(POINTER(c_double)))
    