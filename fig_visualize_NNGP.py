import nngp
import numpy as np
import interp_fast as interp
import matplotlib
import matplotlib.pyplot as plt

def relu(x):
        return x * (x > 0)

# variables
depth = 4
weight_var = 1.0
bias_var = 1.0
nonlin_fn = relu

# constants
grid_path = './grid_data'
n_gauss = 501
n_var = 501
n_corr = 500
max_gauss = 10
max_var = 100

nngp_kernel = nngp.NNGPKernel(
        depth=depth,
        weight_var=weight_var,
        bias_var=bias_var,
        nonlin_fn=nonlin_fn,
        grid_path=grid_path,
        n_gauss=n_gauss,
        n_var=n_var,
        n_corr=n_corr,
        max_gauss=max_gauss,
        max_var=max_var,
        use_precomputed_grid=True)

nngp_kernel.k_diag([], 1.0)
tx = np.linspace(0., 1., 1000, dtype=np.float32)
ty = np.copy(tx)
cov0 = interp.recursive_kernel(x=nngp_kernel.var_aa_grid,
                                y=nngp_kernel.corr_ab_grid,
                                z=nngp_kernel.qab_grid,
                                yp=ty,
                                depth=nngp_kernel.depth,
                                weight_var=nngp_kernel.weight_var,
                                bias_var=nngp_kernel.bias_var,
                                layer_qaa=nngp_kernel.layer_qaa)

'''
Make figure
'''
params = {'legend.fontsize': 12,
         'axes.labelsize': 12,
         'axes.titlesize': 14,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
matplotlib.rcParams.update(params)

plt.figure(figsize=(5,4))
plt.plot(tx, ty+cov0)
plt.xlim((0,1))
plt.xlabel(r"$\frac{x\cdot x'}{d_{in}}$")
plt.grid(True, which='major', alpha=.3)
plt.ylabel(r"$K^L(x,x')$")
plt.tight_layout()
plt.savefig('fig_visualize_NNGP.png')
plt.show()
