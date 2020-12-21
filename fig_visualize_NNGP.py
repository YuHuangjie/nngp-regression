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
nonlin_fn = np.tanh

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

nngp_kernel.k_diag([], 1.0)
tx_relu = np.linspace(0., 1., 1000, dtype=np.float32)
ty_relu = np.copy(tx_relu)
cov0_relu = interp.recursive_kernel(x=nngp_kernel.var_aa_grid,
                                y=nngp_kernel.corr_ab_grid,
                                z=nngp_kernel.qab_grid,
                                yp=ty_relu,
                                depth=nngp_kernel.depth,
                                weight_var=nngp_kernel.weight_var,
                                bias_var=nngp_kernel.bias_var,
                                layer_qaa=nngp_kernel.layer_qaa)

### tanh nonlinear
nngp_kernel = nngp.NNGPKernel(
        depth=depth,
        weight_var=weight_var,
        bias_var=bias_var,
        nonlin_fn=np.tanh,
        grid_path=grid_path,
        n_gauss=n_gauss,
        n_var=n_var,
        n_corr=n_corr,
        max_gauss=max_gauss,
        max_var=max_var,
        use_precomputed_grid=True)

nngp_kernel.k_diag([], 1.0)
tx_tanh = np.linspace(0., 1., 1000, dtype=np.float32)
ty_tanh = np.copy(tx_tanh)
cov0_tanh = interp.recursive_kernel(x=nngp_kernel.var_aa_grid,
                                y=nngp_kernel.corr_ab_grid,
                                z=nngp_kernel.qab_grid,
                                yp=ty_tanh,
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

fig, ax1 = plt.subplots()
ax1.plot(tx_relu, ty_relu+cov0_relu, label='relu', color='blue')
ax1.set_xlim((0,1))
ax1.set_xlabel(r"$\frac{x\cdot x'}{d_{in}}$")
ax1.set_ylabel(r"relu $K^L(x,x')$")
ax1.grid(True, which='major', alpha=.3)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(tx_tanh, ty_tanh+cov0_tanh, label='tanh', color='red')
ax2.set_ylabel(r"tanh $K^L(x,x')$")
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('fig_visualize_NNGP.png')
plt.show()
