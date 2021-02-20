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
tx = np.linspace(-4., 4., 1000, dtype=np.float32)
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
x, y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(2*np.pi, 0, 30))
X = np.stack([x, y], axis=-1)
X = np.reshape(X, [-1, 2])

kernels = {
        'exp': lambda x: np.exp(-0.05*np.linalg.norm((x[None, :, :]-X[:, None, :]), axis=-1)),
        'anisotropic': lambda x: np.exp(-3*(np.abs(x[None, :, 0]-X[:, None, 0]))) * \
                np.exp(-0.1*np.linalg.norm(np.hstack([np.cos(x[:,1:]), np.sin(x[:,1:])])[None,:,:] - np.hstack([np.cos(X[:,1:]), np.sin(X[:,1:])])[:,None,:], axis=-1))
}
title = {
        'exp': r"(a) EXP base kernel",
        'anisotropic': r"(b) Compound base kernel"
}
h = {}
for k in kernels:
        base_kernel = kernels[k]
        K = base_kernel(X)      # base kernel matrix
        K = np.interp(K, tx, ty+cov0)    # apply NNGP transformation
        K += 0.01*np.eye(K.shape[0])

        s_p1 = r'[0, 2\pi]^\mathrm{T}'
        x_p1 = np.array([[0, 2*np.pi]])
        k_p1 = base_kernel(x_p1)
        k_p1 = np.interp(k_p1, tx, ty+cov0)
        h[k] = np.linalg.solve(K, k_p1)

'''
Make figure
'''
params = {'legend.fontsize': 13,
         'axes.labelsize': 13,
         'axes.titlesize': 15,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
matplotlib.rcParams.update(params)
fig, axs = plt.subplots(1, len(kernels), figsize=(len(kernels)*6,5))


# plot NNGP weight function
for i, k in enumerate(kernels):
        axs[i].set_xlabel('altitude')
        axs[i].set_ylabel('azimuth')
        axs[i].set_xticks([-3, -1.5, 0, 1.5, 3])
        axs[i].set_yticks([0, np.pi, 2*np.pi])
        axs[i].set_yticklabels(['0', '$\pi$', '$2\pi$'])
        axs[i].set_title(title[k], y=-0.25)
        h[k] = np.reshape(h[k], x.shape)

        im = axs[i].imshow(h[k], extent=[-3,3,0,2*np.pi], vmin=0, interpolation='bilinear')
        axs[i].text(1, 5.7, s=r'$x^*='+f'{s_p1}'+'$', fontsize=13, c='r')

        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.formatter.set_powerlimits((0, 0))

# plt.tight_layout()
plt.tight_layout()
plt.savefig('fig_weight_function_anisotropic.png')
plt.savefig('fig_weight_function_anisotropic.pdf')
plt.show()