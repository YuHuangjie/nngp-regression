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

import numpy as np

import gpr
import nngp
import load_dataset

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
parser.add_argument('--num_train', default=1000, type=int,
                    help='Number of training data.')
parser.add_argument('--num_eval', default=1000, type=int,
                    help='Number of evaluation data. Use 10_000 for full eval')
parser.add_argument('--seed', default=1234, type=int,
                    help='Random number seed for data shuffling')
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
parser.add_argument('--radius', default=1e-2, type=float,
                    help='threshold for euclidean dist of two samples')

def do_eval(args, model, x_data, y_data, save_pred=False):
    """Run evaluation."""

    gp_prediction = model.predict(x_data)
    gp_prediction = np.clip(gp_prediction, 0., 1.)

    mse = np.mean(np.mean((gp_prediction - y_data)**2, axis=1))+1e-10
    psnr = 10 * np.log10(1./mse)
    logging.info('PSNR: {:.4f}'.format(psnr))
    logging.info('MSE: {:.8f}'.format(mse))

    if save_pred:
        with open('data/test56.npy', 'rb') as f:
            _,_,mask = np.load(f), np.load(f), np.squeeze(np.load(f))
        I = np.ones((512*512,3))
        I[mask] = np.clip(gp_prediction, 0., 1.)
        import imageio
        imageio.imsave('result.png', np.reshape(I, (512,512,3)))

    return mse, psnr


def run_nngp_eval(args):
    """Runs experiments."""

    run_dir = args.experiment_dir
    os.makedirs(run_dir, exist_ok=True)
    hparams = {
        'nonlinearity': args.nonlinearity,
        'weight_var': args.weight_var,
        'bias_var': args.bias_var,
        'depth': args.depth
    }
    # Write hparams to experiment directory.
    with open(run_dir + '/hparams.txt', mode='w') as f:
        f.write(json.dumps(hparams))

    logging.info('Starting job.')
    logging.info('Hyperparameters')
    logging.info('---------------------')
    logging.info(hparams)
    logging.info('---------------------')
    logging.info('Loading data')

    # Get the sets of images and labels for training, validation, and
    # # test on dataset.
    if args.dataset == 'mnist':
        (train_x, train_y, valid_x, valid_y, test_x,
            test_y) = load_dataset.load_mnist(args, 
                num_train=args.num_train,
                mean_subtraction=True,
                random_roated_labels=False)
    elif args.dataset == 'surface':
        train_paths = [os.path.join('data', f'{i}.npy') for i in range(100)]
        # train_paths = [os.path.join('data', f'{i}.npy') for i in [
        #     55,13,20,28,29,63,
        #     65,70,76,84,89,98,
        #     0,1,2,3,4,5,
        #     6,7,8,9,10,11,
        #     12,14,15,16,17,18,
        #     19,21,22,23,24,25,
        #     26,27,30,31,32,33,
        #     34,35,36,37,38,39,
        #     40,41,42,43,44,45,
        #     46,47,48,49,50,51,
        #     52,53,54,56,57,58]]
        train_paths = [os.path.join('data', f'{i}.npy') for i in [
            55,13,20,28,29,63,
            65,70,76,84,89,98,]]
        test_paths = [os.path.join('data', f'test{i}.npy') for i in [56]]
        (train_x, train_y, test_x, test_y) = load_dataset.load_surface(train_paths, test_paths)
        # train_x = train_x[:, :3]
        # test_x = test_x[:, :3]
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
    elif hparams['nonlinearity'] == 'sin':
        nonlin_fn = np.sin
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
        train_x, train_y, kern=nngp_kernel, l=args.l_pts, radius=args.radius)

    start_time = time.time()
    logging.info('Training')

    # For large number of training points, we do not evaluate on full set to
    # save on training evaluation time.
    mse_train, psnr_train = do_eval(args, model, train_x, train_y)
    logging.info('Evaluation of training set ({0} examples) took '
                 '{1:.3f} secs'.format(train_x.shape[0], time.time() - start_time))

    start_time = time.time()
    logging.info('Test')
    mse_test, psnr_test = do_eval(args, model, test_x, test_y, save_pred=True)
    logging.info('Evaluation of test set ({0} examples) took {1:.3f} secs'.format(
        test_x.shape[0], time.time() - start_time))

    metrics = {
        'train_mse': float(mse_train),
        'train_psnr': float(psnr_train),
        'test_mse': float(mse_test),
        'test_psnr': float(psnr_test),
    }

    record_results = [
        args.num_train, hparams['nonlinearity'], hparams['weight_var'],
        hparams['bias_var'], hparams['depth'], psnr_train, psnr_test,
        mse_train, mse_test
    ]

    # Store data
    result_file = os.path.join(run_dir, 'results.csv')
    with open(result_file, 'a') as f:
        f.write(json.dumps(record_results))

    return metrics


if __name__ == '__main__':
    args = parser.parse_args()
    run_nngp_eval(args)
