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

"""Tests for nngp.py."""

import unittest
import numpy as np
import torch
import torch.nn.functional as F

import nngp_torch as nngp


class NNGPTest(unittest.TestCase):

    def ExactQabArcCos(self, var_aa, corr_ab):
        """Exact integration result from Cho & Saul (2009).

        Specifically:
            qaa = 0.5*qaa
            qab = (qaa/2*pi)*(sin angle + (pi-angle)*cos angle),

            where cos angle = corr_ab.

        Args:
            var_aa: 1d tensor of variance grid points.
            corr_ab: 1d tensor of correlation grid points.
        Returns:
            qab_exact: tensor, exact covariance matrix.
        """
        angle = torch.acos(corr_ab)
        jtheta = torch.sin(angle) + (np.pi - angle) * torch.cos(angle)

        term1 = torch.unsqueeze(var_aa, 1).repeat([1, corr_ab.shape[0]])
        term2 = torch.unsqueeze(jtheta, 0).repeat([var_aa.shape[0], 1])
        qab_exact = (1 / (2 * np.pi)) * term1 * term2

        return qab_exact

    def testComputeQmapGridRelu(self):
        """Test checks the compute_qmap_grid function.

        (i) Checks sizes are appropriate and (ii) checks
        accuracy of the numerical values generated by the
        grid by comparing against the analytically known
        form for Relu (Cho and Saul, '09).
        """
        n_gauss, n_var, n_corr = 301, 33, 31
        kernel = nngp.NNGPKernel(
            nonlin_fn=F.relu, n_gauss=n_gauss, n_var=n_var, n_corr=n_corr)

        var_aa_grid = kernel.var_aa_grid
        corr_ab_grid = kernel.corr_ab_grid
        qaa_grid = kernel.qaa_grid
        qab_grid = kernel.qab_grid

        qaa_exact = 0.5 * var_aa_grid
        qab_exact = self.ExactQabArcCos(var_aa_grid, corr_ab_grid)

        self.assertEqual(list(var_aa_grid.shape), [n_var])
        self.assertEqual(list(corr_ab_grid.shape), [n_corr])
        self.assertTrue(torch.allclose(qaa_exact, qaa_grid, rtol=1e-6))
        self.assertTrue(torch.allclose(qab_exact, qab_grid, rtol=1e-6, atol=2e-2))

    def testComputeQmapGridReluLogSpacing(self):
        """Test checks the compute_qmap_grid function with log_spacing=True.
        """
        n_gauss, n_var, n_corr = 301, 33, 31
        kernel = nngp.NNGPKernel(
            nonlin_fn=F.relu, n_gauss=n_gauss, n_var=n_var, n_corr=n_corr)

        var_aa_grid = kernel.var_aa_grid
        corr_ab_grid = kernel.corr_ab_grid
        qaa_grid = kernel.qaa_grid
        qab_grid = kernel.qab_grid

        qaa_exact = 0.5 * var_aa_grid
        qab_exact = self.ExactQabArcCos(var_aa_grid, corr_ab_grid)

        self.assertEqual(list(var_aa_grid.shape), [n_var])
        self.assertEqual(list(corr_ab_grid.shape), [n_corr])
        self.assertTrue(torch.allclose(qaa_exact, qaa_grid, rtol=1e-6, atol=2e-2))
        self.assertTrue(torch.allclose(qab_exact, qab_grid, rtol=1e-6, atol=2e-2))

    def testComputeQmapGridEvenNGauss(self):
        n_gauss, n_var, n_corr = 102, 33, 31

        with self.assertRaises(ValueError):
            nngp.NNGPKernel(
                nonlin_fn=F.relu, n_gauss=n_gauss, n_var=n_var, n_corr=n_corr)

if __name__ == "__main__":
    unittest.main()