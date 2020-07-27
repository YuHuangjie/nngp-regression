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
import torch
import torch.nn.functional as F

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
parser.add_argument('--dataset', default='mnist',
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



def do_eval(args, model, x_data, y_data, save_pred=False):
    """Run evaluation."""

    gp_prediction, stability_eps = model.predict(x_data)

    pred_1 = np.argmax(gp_prediction, axis=1)
    accuracy = np.sum(pred_1 == np.argmax(y_data, axis=1)) / float(len(y_data))
    mse = np.mean(np.mean((gp_prediction - y_data)**2, axis=1))
    pred_norm = np.mean(np.linalg.norm(gp_prediction, axis=1))
    logging.info('Accuracy: {:.4f}'.format(accuracy))
    logging.info('MSE: {:.8f}'.format(mse))

    if save_pred:
        with open(os.path.join(args.experiment_dir, 'gp_prediction_stats.npy'), 'w') as f:
            np.save(f, gp_prediction)

    return accuracy, mse, pred_norm, stability_eps


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
        (train_image, train_label, valid_image, valid_label, test_image,
            test_label) = load_dataset.load_mnist(args, 
                num_train=args.num_train,
                mean_subtraction=True,
                random_roated_labels=False)
    else:
        raise NotImplementedError

    logging.info('Building Model')

    if hparams['nonlinearity'] == 'tanh':
        nonlin_fn = F.tanh
    elif hparams['nonlinearity'] == 'relu':
        nonlin_fn = F.relu
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
        train_image, train_label, kern=nngp_kernel)

    start_time = time.time()
    logging.info('Training')

    # For large number of training points, we do not evaluate on full set to
    # save on training evaluation time.
    train_size = args.num_eval if args.num_train <= 5000 else 1000
    acc_train, mse_train, norm_train, final_eps = do_eval(
            args, model, train_image[:train_size], train_label[:train_size])
    logging.info('Evaluation of training set ({0} examples) took '
                 '{1:.3f} secs'.format(min(args.num_train, args.num_eval),
                 time.time() - start_time))

    start_time = time.time()
    logging.info('Validation')
    acc_valid, mse_valid, norm_valid, _ = do_eval(
        args, model, valid_image[:args.num_eval], valid_label[:args.num_eval])
    logging.info('Evaluation of valid set ({0} examples) took {1:.3f} secs'.format(
        args.num_eval, time.time() - start_time))

    start_time = time.time()
    logging.info('Test')
    acc_test, mse_test, norm_test, _ = do_eval(
        args, model, test_image[:args.num_eval], test_label[:args.num_eval])
    logging.info('Evaluation of valid set ({0} examples) took {1:.3f} secs'.format(
        args.num_eval, time.time() - start_time))

    metrics = {
        'train_acc': float(acc_train),
        'train_mse': float(mse_train),
        'train_norm': float(norm_train),
        'valid_acc': float(acc_valid),
        'valid_mse': float(mse_valid),
        'valid_norm': float(norm_valid),
        'test_acc': float(acc_test),
        'test_mse': float(mse_test),
        'test_norm': float(norm_test),
        'stability_eps': float(final_eps),
    }

    record_results = [
        args.num_train, hparams['nonlinearity'], hparams['weight_var'],
        hparams['bias_var'], hparams['depth'], acc_train, acc_valid, acc_test,
        mse_train, mse_valid, mse_test, final_eps
    ]

    # Store data
    result_file = os.path.join(run_dir, 'results.csv')
    with open(result_file, 'a') as f:
        f.write(json.dumps(record_results))

    return metrics


if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    run_nngp_eval(args)
