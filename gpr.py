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

"""Gaussian process regression model based on GPflow.
"""

import time
import logging

import ray
import numpy as np
import scipy.sparse as sparse
from sklearn.neighbors import KDTree, NearestNeighbors
from sksparse.cholmod import cholesky

ray.init()

@ray.remote
def solve_system(A, b):
    return sparse.linalg.cg(A, b, atol=1e-4, tol=1e-4, maxiter=100)[0]

class GaussianProcessRegression():
    """Gaussian process regression model based on GPflow.

    Args:
        input_x: numpy array, [data_size, input_dim]
        output_x: numpy array, [data_size, output_dim]
        kern: NNGPKernel class
    """

    def __init__(self, input_x, output_y, kern, l = 0.01, radius=1e-2):
        self.input_x = input_x.astype(np.float64)
        self.output_y = output_y.astype(np.float64)
        self.num_train, self.input_dim = input_x.shape
        _, self.output_dim = output_y.shape

        self.kern = kern

        self.nn = NearestNeighbors(radius=radius, algorithm='kd_tree', n_jobs=-1).fit(input_x[:,:3])
        self.radius = radius
        self.l = l

        self.v = None

    def _build_predict(self, test_x, full_cov=False):
        logging.info("Performing bayesian inference")
        self.s_test_data, self.c_test_data = self.kern.k_full(self.nn, test_x, 
            self.input_x, self.l, self.radius)
        self.fmean = self.s_test_data.dot(self.v) + self.c_test_data * np.sum(self.v, axis=0)

    def _build_inv_KDD(self):
        '''
          K_DD = S_DD + C where C is a NxN matrix with all elements being 1
          According to Sherman-Morrison formula: 
             inv(K_DD) = inv(S_DD) - /frac(inv(S_DD)*u*u^T*inv(S_DD))(1+u^T*inv(S_DD)*u)
          where UU^T = C.
          So the algorithm below calculates inv(K_DD) * y
        '''
        ## SPD test
        # factor = cholesky(self.s_data_data, 1e-8,use_long=True)
        # x = factor(self.output_y)
        logging.info('Building inv K_DD')
        
        C = self.c_data_data
        a = np.zeros((self.num_train, 3)) # inv(S_DD) * y, Nx3
        # parallel solving for inv(S_DD)\y and inv(S_DD)\u
        A_id = ray.put(self.s_data_data)
        result_ids = []
        for i in range(3):
            result_ids.append(solve_system.remote(A_id, self.output_y[:,i]))
        result_ids.append(solve_system.remote(A_id, np.ones((self.num_train,1)) * C**0.5))
        results = ray.get(result_ids) 

        for i in range(3):
            a[:,i] = results[i]
        d = np.expand_dims(results[3], -1) # inv(S_DD)*u, Nx1

        b = C**0.5 * np.sum(a, axis=0, keepdims=True)     # U^T * inv(S_DD) * y, 1x3
        b = np.matmul(d, b)                 # inv(S_DD) * U * U^T * inv(S_DD) * y, Nx3
        d = 1 + C**0.5 * np.sum(d, axis=0)  # 1 + U^T * inv(S_DD) * U, scalar
        self.v = a - b / d

    def predict(self, test_x, get_var=False):
        """Compute mean and varaince prediction for test inputs.

        Raises:
            ArithmeticError: Cholesky fails even after increasing to large values of
                stability epsilon.
        """
        if self.v is None:
            start_time = time.time()
            self.s_data_data, self.c_data_data = self.kern.k_full(self.nn,
                self.input_x, self.input_x, self.l, self.radius)
            logging.info("Computed full K_DD in {:.2f} secs".format(time.time()-start_time))

            start_time = time.time()
            self._build_inv_KDD()
            logging.info("Computed inv K_DD in {:.3f} secs".format(time.time()-start_time))

            # don't predict training set
            return self.output_y

        start_time = time.time()
        self._build_predict(test_x.astype(np.float64), get_var)
        logging.info("Did regression in {:.3f} secs".format(time.time()-start_time))

        return self.fmean

