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

"""Neural Network Gaussian Process (nngp) kernel computation.

Implementaion based on
"Deep Neural Networks as Gaussian Processes" by
Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz,
Jeffrey Pennington, Jascha Sohl-Dickstein
arXiv:1711.00165 (https://arxiv.org/abs/1711.00165).
"""
import os
import logging

import numpy as np
from scipy.special import logsumexp
import interp_fast as interp
import scipy.sparse as sparse
from kernel import *
import gc

class NNGPKernel():
    """The iterative covariance Kernel for Neural Network Gaussian Process.

    Args:
        depth: int, number of hidden layers in corresponding NN.
        nonlin_fn: torch ops corresponding to point-wise non-linearity in corresponding
            NN. e.g.) F.relu, F.sigmoid, lambda x: x * F.sigmoid(x), ...
        weight_var: initial value for the weight_variances parameter.
        bias_var: initial value for the bias_variance parameter.
        n_gauss: Number of gaussian integration grid. Choose odd integer, so that
            there is a gridpoint at 0.
        n_var: Number of variance grid points.
        n_corr: Number of correlation grid points.
    """
    def __init__(self,
                depth=1,
                nonlin_fn=lambda x: x * (x > 0),
                weight_var=1.,
                bias_var=1.,
                n_gauss=101,
                n_var=151,
                n_corr=131,
                max_var=100,
                max_gauss=100,
                use_precomputed_grid=False,
                grid_path=None):
        
        self.depth = depth
        self.weight_var = weight_var
        self.bias_var = bias_var
        if use_precomputed_grid and (grid_path is None):
                raise ValueError("grid_path must be specified to use precomputed grid.")
        self.use_precomputed_grid = use_precomputed_grid
        self.grid_path = grid_path
        self.nonlin_fn = nonlin_fn
        self.var_aa_grid, self.corr_ab_grid, self.qaa_grid, self.qab_grid = \
            self.get_grid(n_gauss, n_var, n_corr, max_var, max_gauss)

    def get_grid(self, n_gauss, n_var, n_corr, max_var, max_gauss):
        """Get covariance grid by loading or computing a new one.
        """
        # File configuration for precomputed grid
        if self.use_precomputed_grid:
            grid_path = self.grid_path
            # TODO(jaehlee) np.save have broadcasting error when n_var==n_corr.
            if n_var == n_corr:
                n_var += 1
            grid_file_name = "grid_{0:s}_ng{1:d}_ns{2:d}_nc{3:d}".format(
                self.nonlin_fn.__name__, n_gauss, n_var, n_corr)
            grid_file_name += "_mv{0:d}_mg{1:d}".format(max_var, max_gauss)

        # Load grid file if it exists already
        if self.use_precomputed_grid and os.path.exists(os.path.join(grid_path, grid_file_name)):
            with open(os.path.join(grid_path, grid_file_name), "rb") as f:
                grid_data = np.load(f, allow_pickle=True, encoding='bytes')
                logging.info("Loaded interpolation grid from "
                            f"{os.path.join(grid_path, grid_file_name)}")
        else:
            logging.info("Generating interpolation grid...")
            grid_data = _compute_qmap_grid(self.nonlin_fn, n_gauss, n_var, n_corr,
                                    max_var=max_var, max_gauss=max_gauss)
            if self.use_precomputed_grid:
                os.makedirs(grid_path, exist_ok=True)
                with open(os.path.join(grid_path, grid_file_name), "wb") as f:
                    np.save(f, [grid_data[i].numpy() for i in range(4)])

                with open(os.path.join(grid_path, grid_file_name), "rb") as f:
                    grid_data = np.load(f, allow_pickle=True, encoding='bytes')
                    logging.info("Loaded interpolation grid from "
                                f"{os.path.join(grid_path, grid_file_name)}")

        return grid_data

    def k_diag(self, input_x, init_cov=1.):
        """Iteratively building the diagonal part (variance) of the NNGP kernel.

        Args:
            input_x: tensor of input of size [num_data, input_dim].
            init_cov: initial covariance of two identical samples (default to 1)

        Sets self.layer_qaa of {layer #: qaa at the layer}

        Returns:
        qaa: variance at the output.
        """
        ## squared exponential kernel
        current_qaa = self.weight_var * np.array([init_cov]) + self.bias_var
        self.layer_qaa = np.zeros((self.depth+1,))
        self.layer_qaa[0] = current_qaa
        for l in range(self.depth):
            samp_qaa = interp.interp_lin(self.var_aa_grid, self.qaa_grid, current_qaa)
            samp_qaa = self.weight_var * samp_qaa + self.bias_var
            self.layer_qaa[l + 1] = samp_qaa
            current_qaa = samp_qaa

        return current_qaa

    def k_full(self, nn, x, train_x, l_pts, l_dir, gamma_pts, gamma_dir, radius):
        """Iteratively building the full NNGP kernel.
        """
        # Initial covariance S is designed so that it's a super sparse matrix. 
        # After k linear layers, S becomes a sparse matrix plus a constant C.
        graph = nn.radius_neighbors_graph(x[:,:3], radius, mode='connectivity')
        gc.collect()
        if graph.indptr.dtype != np.int64:
            graph.indptr = graph.indptr.astype(np.int64, copy=False)
            graph.indices = graph.indices.astype(np.int64, copy=False)
        logging.info(f'initial cov contains {graph.nnz} non-zeros')
        build_kernel(graph, train_x, x, l_pts, l_dir, gamma_pts, gamma_dir)
        graph.eliminate_zeros()

        self.k_diag(x, 1.0)
        cov0 = interp.recursive_kernel(x=self.var_aa_grid,
                                y=self.corr_ab_grid,
                                z=self.qab_grid,
                                yp=graph.data,
                                depth=self.depth,
                                weight_var=self.weight_var,
                                bias_var=self.bias_var,
                                layer_qaa=self.layer_qaa)
        return graph, cov0

def _fill_qab_slice(idx, z1, z2, var_aa, corr_ab, nonlin_fn):
    """Helper method used for parallel computation for full qab."""
    log_weights_ab_unnorm = -(z1**2 + z2**2 - 2 * z1 * z2 * corr_ab) / (
        2 * var_aa[idx] * (1 - corr_ab**2))
    log_weights_ab = log_weights_ab_unnorm - logsumexp(
        log_weights_ab_unnorm, axis=(0, 1), keepdims=True)
    weights_ab = np.exp(log_weights_ab)

    qab_slice = np.sum(nonlin_fn(z1) * nonlin_fn(z2) * weights_ab, axis=(0, 1))
    print(f"Generating slice: [{idx}]")
    return qab_slice

def _compute_qmap_grid(nonlin_fn,
                       n_gauss,
                       n_var,
                       n_corr,
                       log_spacing=False,
                       min_var=1e-8,
                       max_var=100.,
                       max_corr=0.99999,
                       max_gauss=10.):
    """Construct graph for covariance grid to use for kernel computation.

    Given variance and correlation (or covariance) of pre-activation, perform
    Gaussian integration to get covariance of post-activation.

    Raises:
        ValueError: if n_gauss is even integer.

    Args:
        nonlin_fn: tf ops corresponding to point-wise non-linearity in
            corresponding NN. e.g.) tf.nn.relu, tf.nn.sigmoid,
            lambda x: x * tf.nn.sigmoid(x), ...
        n_gauss: int, number of Gaussian integration points with equal spacing
            between (-max_gauss, max_gauss). Choose odd integer, so that there is a
            gridpoint at 0.
        n_var: int, number of variance grid points.get_grid
        n_corr: int, number of correlation grid points.
        log_spacing: bool, whether to use log-linear instead of linear variance
            grid.
        min_var: float, smallest variance value to generate grid.
        max_var: float, largest varaince value to generate grid.
        max_corr: float, largest correlation value to generate grid. Should be
            slightly smaller than 1.
        max_gauss: float, range (-max_gauss, max_gauss) for Gaussian integration.

    Returns:
        var_grid_pts: tensor of size [n_var], grid points where variance are
            evaluated at.
        corr_grid_pts: tensor of size [n_corr], grid points where correlation are
            evalutated at.
        qaa: tensor of size [n_var], variance of post-activation at given
            pre-activation variance.
        qab: tensor of size [n_var, n_corr], covariance of post-activation at
            given pre-activation variance and correlation.
    """
    if n_gauss % 2 != 1:
        raise ValueError(f"n_gauss={n_gauss} should be an odd integer")

    min_var = min_var
    max_var = max_var
    max_corr = max_corr
    max_gauss = max_gauss

    # Evaluation points for numerical integration over a Gaussian.
    z1 = np.reshape(np.linspace(-max_gauss, max_gauss, n_gauss), (-1, 1, 1))
    z2 = np.transpose(z1, (1, 0, 2))

    if log_spacing:
        var_aa = np.exp(np.linspace(np.log(min_var), np.log(max_var), n_var))
    else:
        # Evaluation points for pre-activations variance and correlation
        var_aa = np.linspace(min_var, max_var, n_var)
    corr_ab = np.reshape(np.linspace(-max_corr, max_corr, n_corr), (1, 1, -1))

    # compute q_aa
    log_weights_aa_unnorm = -0.5 * (z1**2 / np.reshape(var_aa, [1, 1, -1]))
    log_weights_aa = log_weights_aa_unnorm - logsumexp(
        log_weights_aa_unnorm, (0, 1), keepdims=True)
    weights_aa = np.exp(log_weights_aa)
    qaa = np.sum(nonlin_fn(z1)**2 * weights_aa, axis=(0, 1))

    # compute q_ab
    # weights to reweight uniform samples by, for q_ab.
    # (weights are probability of z1, z2 under Gaussian
    #  w/ variance var_aa and covariance var_aa*corr_ab)
    # weights_ab will have shape [n_g, n_g, n_v, n_c]
    def fill_qab_slice(idx):
        return _fill_qab_slice(idx, z1, z2, var_aa, corr_ab, nonlin_fn)

    # TODO: multithread
    qab = np.zeros((n_var, n_corr))
    for i in range(n_var):
        qab[i] = fill_qab_slice(i)

    var_grid_pts = np.reshape(var_aa, [-1])
    corr_grid_pts = np.reshape(corr_ab, [-1])

    return var_grid_pts, corr_grid_pts, qaa, qab
