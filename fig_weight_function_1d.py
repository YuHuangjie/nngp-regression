import nngp
import numpy as np
import interp_fast as interp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def relu(x):
        return x * (x > 0)

# variables
depth = 4
weight_var = 1.0
bias_var = 1.0

# constants
grid_path = './grid_data'
n_gauss = 501
n_var = 501
n_corr = 500
max_gauss = 10
max_var = 100

### RELU nonlinear
nngp_kernel = nngp.NNGPKernel(
        depth=depth,
        weight_var=weight_var,
        bias_var=bias_var,
        nonlin_fn=relu,
        grid_path=grid_path,
        n_gauss=n_gauss,
        n_var=n_var,
        n_corr=n_corr,
        max_gauss=max_gauss,
        max_var=max_var,
        use_precomputed_grid=True)

nngp_kernel.k_diag([], 4.0)
tx = np.linspace(-4., 4., 100, dtype=np.float32)
ty = np.copy(tx)
cov0 = interp.recursive_kernel(x=nngp_kernel.var_aa_grid,
                                y=nngp_kernel.corr_ab_grid,
                                z=nngp_kernel.qab_grid,
                                yp=ty,
                                depth=nngp_kernel.depth,
                                weight_var=nngp_kernel.weight_var,
                                bias_var=nngp_kernel.bias_var,
                                layer_qaa=nngp_kernel.layer_qaa)

### calculate weight function of NNGP with dot product
x = np.linspace(-2, 2, 100)[:,None]
X = np.sort(x, axis=0)

kernels = {
        'simple': lambda x: X@x.T/2,
        'rbf': lambda x: np.exp(-np.sum((x[None, :, :]-X[:, None, :])**2, axis=-1)),
        'exp': lambda x: np.exp(-np.linalg.norm((x[None, :, :]-X[:, None, :]), axis=-1)),
}
title = {
        'simple': r"(a) $x\cdot x'$",
        'rbf': r"(b) $\mathrm{exp}(-\Vert x-x' \Vert^2)$",
        'exp': r"(c) $\mathrm{exp}(-\Vert x-x' \Vert)$",
}
h1 = {}
h2 = {}
for k in kernels:
        base_kernel = kernels[k]
        K = base_kernel(X)      # base kernel matrix
        K = np.interp(K, tx, ty+cov0)    # apply NNGP transformation
        K += 0.01*np.eye(K.shape[0])

        s_p1 = r'-0.5'
        x_p1 = np.array([[-0.5]])
        k_p1 = base_kernel(x_p1)
        k_p1 = np.interp(k_p1, tx, ty+cov0)
        h1[k] = np.linalg.solve(K, k_p1)

        s_p2 = r'0.5'
        x_p2 = np.array([[0.5]])
        k_p2 = base_kernel(x_p2)
        k_p2 = np.interp(k_p2, tx, ty+cov0)
        h2[k] = np.linalg.solve(K, k_p2)

'''
Make figure
'''
params = {'legend.fontsize': 13,
         'axes.labelsize': 13,
         'axes.titlesize': 15,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
matplotlib.rcParams.update(params)
fig, axs = plt.subplots(1, len(kernels), figsize=(len(kernels)*5,5))


# plot NNGP weight function
for i, k in enumerate(kernels):
        axs[i].grid(True)
        axs[i].set_xlim((-2, 2))
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('weight')
        axs[i].set_xticks([-2, -1, 0, 1, 2])
        axs[i].set_title(title[k], y=-0.25)
        axs[i].plot(X, h1[k], marker='*', color='b', label=r'$x^*='+f'{s_p1}'+'$')
        axs[i].plot(X, h2[k], marker='*', color='r', label=r'$x^*='+f'{s_p2}'+'$')


# plt.tight_layout()
plt.legend()
plt.tight_layout()
plt.savefig('fig_weight_function_1d.png')
plt.savefig('fig_weight_function_1d.pdf')
plt.show()