import nngp
import numpy as np
import interp_fast as interp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def relu(x):
        return x * (x > 0)

# variables
weight_var = 1.0
bias_var = 0.5
nonlin_fns = [relu, np.tanh]
configs = [
        {'depth': 3, 'color': 'red'},
        {'depth': 4, 'color': 'red'},
        {'depth': 5, 'color': 'green'},
        {'depth': 6, 'color': 'blue'},
        {'depth': 7, 'color': 'blue'},
]

# constants
grid_path = './grid_data'
n_gauss = 501
n_var = 501
n_corr = 500
max_gauss = 10
max_var = 100

params = {'legend.fontsize': 13,
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'xtick.labelsize':12,
        'ytick.labelsize':12}
matplotlib.rcParams.update(params)
fig, axs = plt.subplots(1, 2, figsize=(8,4))
fig.subplots_adjust(left=0.08, bottom=0.25, right=0.95, top=0.95, wspace=0.25)

for i, nonlin_fn in enumerate(nonlin_fns):
        for config in configs:
                ### depth = 4
                nngp_kernel = nngp.NNGPKernel(
                        depth=config['depth'],
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

                # plot NNGP kernel transform
                axs[i].plot(tx, ty+cov0, label=r"$L="+f"{config['depth']}" + r"$")

        '''
        Make figure
        '''
        # plot NNGP kernel transform
        axs[i].set_xlim((-1,1))
        axs[i].set_xticks([-1,-0.5, 0, 0.5, 1])
        axs[i].set_xlabel(r"$x\cdot x' / d_{in}$", labelpad=5)
        if i == 0:
                axs[i].set_ylabel(r"$K^L(x,x')$")
        if i == 0:
                axs[i].set_title('(a) ReLU', y=-0.35)
        else:
                axs[i].set_title('(b) Tanh', y=-0.35)
        axs[i].grid(True, which='major', alpha=.4)

plt.tight_layout()
plt.legend()
plt.savefig('fig_kernel_evolution.png')
plt.savefig('fig_kernel_evolution.pdf')
plt.show()