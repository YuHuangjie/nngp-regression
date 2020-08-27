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

r"""Run experiments with NNGP Kernel.

Usage:

python run_experiments.py \
        --num_train=100 \
        --num_eval=1000 \
        --nonlinearity=relu \
        --depth=10 \
        --weight_var=1.79 \
        --bias_var=0.83 \
        --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10

"""

import csv
import os.path
import time
import logging
import argparse
import json
import pickle

import numpy as np
import imageio
import matplotlib.pyplot as plt

import gpr_fast as gpr
import nngp
from load_dataset import *

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--nonlinearity', default='relu',
                    help='point-wise non-linearity function')
parser.add_argument('--weight_var', default=1.3, type=float,
                    help='initial value for the weight_variances parameter')
parser.add_argument('--bias_var', default=0.2, type=float,
                    help='initial value for the bias_variances parameter')
parser.add_argument('--depth', default=2, type=int,
                    help='number of hidden layers in corresponding NN')

parser.add_argument('--experiment_dir', default='/tmp/nngp',
                    help='Directory to put the experiment results.')
parser.add_argument('--data_dir', default='/tmp/nngp/data/',
                    help='Directory for data.')
parser.add_argument('--grid_path', default='./grid_data',
                    help='Directory to put or find the training data.')
parser.add_argument('--save_kernel', default=False, type=bool,
                    help='Save Kernel to disk')
parser.add_argument('--dataset', default='surface',
                    help='Which dataset to use ["mnist"]')

parser.add_argument('--n_gauss', default=501, type=int,
                    help='Number of gaussian integration grid. Choose odd integer.')
parser.add_argument('--n_var', default=501, type=int,
                    help='Number of variance grid points.')
parser.add_argument('--n_corr', default=500, type=int,
                    help='Number of correlation grid points.')
parser.add_argument('--max_var', default=100, type=int,
                    help='Max value for variance grid.')
parser.add_argument('--max_gauss', default=10, type=int,
                    help='Range for gaussian integration.')

parser.add_argument('--l_pts', default=0.01, type=float,
                    help='length-scale of position')
parser.add_argument('--l_dir', default=0.1, type=float,
                    help='length-scale of directions')
parser.add_argument('--gamma_pts', default=0.5, type=float,
                    help='exponential of position')
parser.add_argument('--gamma_dir', default=1.5, type=float,
                    help='exponential of directions')
parser.add_argument('--radius', default=2e-2, type=float,
                    help='threshold for euclidean dist of two samples')

def do_eval(args, model, x_data, y_data, mask=None, save_path=None):
    """Run evaluation."""

    gp_prediction = model.predict(x_data)
    gp_prediction = np.clip(gp_prediction, 0., 1.)

    mse = np.mean(np.mean((gp_prediction - y_data)**2, axis=1))+1e-10
    psnr = 10 * np.log10(1./mse)
    logging.info('PSNR: {:.4f}'.format(psnr))
    logging.info('MSE: {:.8f}'.format(mse))

    if save_path is not None and mask is not None:
        I = np.zeros((512*512,3))
        I[mask] = np.clip(gp_prediction, 0., 1.)
        imageio.imsave(save_path, np.reshape(I, (512,512,3)))

    return mse, psnr


def run_nngp_eval(args):
    """Runs experiments."""

    run_dir = args.experiment_dir
    os.makedirs(run_dir, exist_ok=True)
    hparams = {
        'nonlinearity': args.nonlinearity,
        'weight_var': args.weight_var,
        'bias_var': args.bias_var,
        'depth': args.depth,
        'l_pts': args.l_pts,
        'l_dir': args.l_dir,
        'gamma_pts': args.gamma_pts,
        'gamma_dir': args.gamma_dir,
        'radius': args.radius,
    }

    logging.info('Starting job.')
    logging.info('Hyperparameters')
    logging.info('---------------------')
    logging.info(hparams)
    logging.info('---------------------')
    logging.info('Loading data')

    # Get the sets of images and labels for training, validation, and
    # # test on dataset.
    if args.dataset == 'surface':
        train_paths = [os.path.join('data', f'{i}.npy') for i in range(100)]
        test_paths = [os.path.join('data', f'test{i}.npy') for i in range(200)]
        (train_x, train_y) = load_training_set(train_paths)
        (test_x, test_y, test_mask) = load_test_set(test_paths)
    else:
        raise NotImplementedError

    logging.info(f'training set size: {train_x.shape[0]}')
    logging.info('Building Model')

    if hparams['nonlinearity'] == 'tanh':
        nonlin_fn = np.tanh
    elif hparams['nonlinearity'] == 'relu':
        def relu(x):
            return x * (x > 0)
        nonlin_fn = relu
    else:
        raise NotImplementedError

    # Construct NNGP kernel
    nngp_kernel = nngp.NNGPKernel(
        depth=hparams['depth'],
        weight_var=hparams['weight_var'],
        bias_var=hparams['bias_var'],
        nonlin_fn=nonlin_fn,
        grid_path=args.grid_path,
        n_gauss=args.n_gauss,
        n_var=args.n_var,
        n_corr=args.n_corr,
        max_gauss=args.max_gauss,
        max_var=args.max_var,
        use_precomputed_grid=True)

    # Construct Gaussian Process Regression model
    model = gpr.GaussianProcessRegression(
        train_x, train_y, kern=nngp_kernel, l_pts=hparams['l_pts'], l_dir=hparams['l_dir'],
        gamma_pts=hparams['gamma_pts'], gamma_dir=hparams['gamma_dir'], radius=hparams['radius'])

    start_time = time.time()
    logging.info('Training')

    # For large number of training points, we do not evaluate on full set to
    # save on training evaluation time.
    mse_train, psnr_train = do_eval(args, model, train_x, train_y)
    logging.info('Evaluation of training set ({0} examples) took '
                 '{1:.3f} secs'.format(train_x.shape[0], time.time() - start_time))

    if args.save_kernel:
        essential = {
            'v': model.v,
            'hparams': hparams,
            'var_grid': nngp_kernel.var_aa_grid,
            'corr_grid': nngp_kernel.corr_ab_grid,
            'qab_grid': nngp_kernel.qab_grid,
            'qaa_grid': nngp_kernel.layer_qaa
        }
        with open(os.path.join(run_dir, 'essential.pkl'), 'wb') as f:
            pickle.dump(essential, f)

    start_time = time.time()
    logging.info('Test')
    psnr_test = [None]*len(test_x)
    for i, (tx, ty, tm) in enumerate(zip(test_x, test_y, test_mask)):
        _, psnr_test[i] = do_eval(args, model, tx, ty, mask=tm, save_path=f'{run_dir}/result{i}.png')
    logging.info('Evaluation of test set ({0} examples) took {1:.3f} secs'.format(
        test_x[0].shape[0]*len(test_x), time.time() - start_time))

    # Log results and hyper-parameters
    plt.plot(np.arange(len(test_x)), psnr_test)
    plt.savefig(f'{run_dir}/psnr_test.png')

    record_results = {
        'training_size': train_x.shape[0],
        'hparams': hparams,
        'psnr_test': psnr_test,
        'psnr_train': psnr_train
    }

    # Store data
    result_file = os.path.join(run_dir, 'results.json')
    with open(result_file, 'w') as f:
        f.write(json.dumps(record_results, indent=4))


if __name__ == '__main__':
    args = parser.parse_args()
    run_nngp_eval(args)
