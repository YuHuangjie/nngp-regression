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

nngp_kernel.k_diag([], 1.0)
tx = np.linspace(-1., 1., 1000, dtype=np.float32)
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
angles = np.random.rand(100, 1) * 2 * np.pi
angles = np.sort(angles, axis=0)
X = np.hstack([np.cos(angles), np.sin(angles)])
power = 19

kernels = {
        'simple': lambda x: X@x/2,
        'poly1': lambda x: (X@x)**power / 2,
        'poly': lambda x: np.maximum(0, X@x)**power / 2
}
title = {
        'simple': r"(a) $x\cdot x'$",
        'poly1': r"(b) $(x\cdot x')^{" + f'{power}' + r"}$",
        'poly': r"(c) $\mathrm{max}(0, x\cdot x')^{" + f'{power}' + r"}}$"
}
h1 = {}
h2 = {}
for k in kernels:
        base_kernel = kernels[k]
        K = base_kernel(X.T)      # base kernel matrix
        K = np.interp(K, tx, ty+cov0)    # apply NNGP transformation
        K += 0.01*np.eye(K.shape[0])

        a_p1 = np.pi * 1/2
        s_p1 = r'\dfrac{1}{2}\pi'
        x_p1 = np.array([[np.cos(a_p1)], [np.sin(a_p1)]])
        k_p1 = base_kernel(x_p1)
        k_p1 = np.interp(k_p1, tx, ty+cov0)
        h1[k] = np.linalg.solve(K, k_p1)

        a_p2 = np.pi * 4/3
        s_p2 = r'\dfrac{4}{3}\pi'
        x_p2 = np.array([[np.cos(a_p2)], [np.sin(a_p2)]])
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
fig.subplots_adjust(left=0.10, bottom=0.25, right=0.95, top=0.95, wspace=0.25)


# plot NNGP weight function
for i, k in enumerate(kernels):
        axs[i].grid(True, which='major', alpha=.3)
        axs[i].set_xlim((0, 2*np.pi))
        axs[i].set_xticks([0, np.pi/2, np.pi, np.pi*1.5, np.pi*2])
        axs[i].set_xticklabels([r'$0$', r'$\dfrac{1}{2}\pi$', r'$\pi$', r'$\dfrac{3}{2}\pi$', r'$2\pi$'])
        axs[i].set_ylim((min(h1[k].min(), h2[k].min())-0.01, max(h1[k].max(), h2[k].max())+0.01))
        axs[i].set_xlabel('angle')
        if i == 0:
                axs[i].set_ylabel('Weight')
        axs[i].set_title(title[k], y=-0.35)
        axs[i].plot(angles, h1[k], marker='.', color='blue', label=r'$x^*='+f'{s_p1}'+'$')
        axs[i].plot(angles, h2[k], marker='.', color='red', label=r'$x^*='+f'{s_p2}'+'$')
        axs[i].legend(loc='upper center')

# # plot the inverse of covariance matrix
# divider = make_axes_locatable(axs[2])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# im2 = axs[2].imshow(np.linalg.inv(K), cmap='hot', interpolation='nearest')
# fig.colorbar(im2, cax=cax)
# axs[2].axis('off')
# axs[2].set_title(r"(c) inverse $[K^L(x,x')+\delta\mathbf{I}]$", y=-0.25)

# plt.tight_layout()
plt.savefig('fig_weight_function.png')
plt.savefig('fig_weight_function.pdf')
plt.show()