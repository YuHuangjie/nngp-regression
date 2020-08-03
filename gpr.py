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

import numpy as np
from scipy.linalg import solve_triangular

# Option to print out kernel
print_kernel = False

class GaussianProcessRegression():
    """Gaussian process regression model based on GPflow.

    Args:
        input_x: numpy array, [data_size, input_dim]
        output_x: numpy array, [data_size, output_dim]
        kern: NNGPKernel class
    """

    def __init__(self, input_x, output_y, kern):
        self.input_x = input_x.astype(np.float64)
        self.output_y = output_y.astype(np.float64)
        self.num_train, self.input_dim = input_x.shape
        _, self.output_dim = output_y.shape

        self.kern = kern
        self.current_stability_eps = 1e-10

        self.l = None

    def _build_predict(self, test_x, full_cov=False):
        logging.info("Using pre-computed Kernel")
        self.k_data_test = self.kern.k_full(self.input_x, test_x)

        a = solve_triangular(self.l, self.k_data_test, lower=True)
        fmean = np.matmul(a.T, self.v)

        if full_cov:
            fvar = self.kern.k_full(test_x) - np.matmul(
                a.T, a)
            shape = [1, 1, self.output_dim]
            fvar = np.tile(np.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.k_diag(test_x) - np.sum(a**2, 0)
            fvar = np.tile(np.reshape(fvar, (-1, 1)), [1, self.output_dim])
        
        self.fmean = fmean
        self.fvar = fvar

    def _build_cholesky(self):
        logging.info('Computing Kernel')
        self.k_data_data_reg = self.k_data_data + np.eye(
            self.num_train, dtype=np.float64) * self.current_stability_eps
        if print_kernel:
            print(f"K_DD = {self.k_data_data_reg}")
        self.l = np.linalg.cholesky(self.k_data_data_reg)
        self.v = solve_triangular(self.l, self.output_y, lower=True)

    def predict(self, test_x, get_var=False):
        """Compute mean and varaince prediction for test inputs.

        Raises:
            ArithmeticError: Cholesky fails even after increasing to large values of
                stability epsilon.
        """
        if self.l is None:
            start_time = time.time()
            self.k_data_data = self.kern.k_full(self.input_x)
            logging.info("Computed K_DD in {:.2f} secs".format(time.time()-start_time))

            while self.current_stability_eps < 1:
                try:
                    start_time = time.time()
                    self._build_cholesky()
                    logging.info("Computed L_DD in {:.3f} secs".format(
                        time.time()-start_time))
                    break
                except RuntimeError as e:
                    self.current_stability_eps *= 10
                    logging.info(f"Cholesky decomposition failed {e}, trying larger "+
                        f"epsilon {self.current_stability_eps}")

        if self.current_stability_eps > 0.2:
            raise ArithmeticError("Could not compute cholesky decomposition.")

        start_time = time.time()
        self._build_predict(test_x.astype(np.float64), get_var)
        logging.info("Did regression in {:.3f} secs".format(time.time()-start_time))

        if get_var:
            return self.fmean, self.fvar, self.current_stability_eps
        else:
            return self.fmean, self.current_stability_eps

