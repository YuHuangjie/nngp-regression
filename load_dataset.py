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

"""Data loader for NNGP experiments.

Loading MNIST dataset with train/valid/test split as numpy array.

Usage:
mnist_data = load_dataset.load_mnist(num_train=50000, use_float64=True,
                                     mean_subtraction=True)
"""

import numpy as np
import torch
import torchvision


def load_mnist(args,
            num_train=50000,
            use_float64=False,
            mean_subtraction=False,
            random_roated_labels=False):
    """Loads MNIST as numpy array."""

    data_dir = args.data_dir
    train = torchvision.datasets.MNIST(data_dir, 
            train=True, 
            download=True, 
            transform=torchvision.transforms.ToTensor())
    val = [train[i] for i in range(50000, len(train))]
    train = [train[i] for i in range(50000)]
    test = torchvision.datasets.MNIST(data_dir, 
            train=False, 
            download=True, 
            transform=torchvision.transforms.ToTensor())
    mnist_data = _select_mnist_subset(
        (train, val, test),
        num_train,
        use_float64=use_float64,
        mean_subtraction=mean_subtraction,
        random_roated_labels=random_roated_labels)

    return mnist_data

def _select_mnist_subset(datasets,
                         num_train=100,
                         digits=list(range(10)),
                         seed=9999,
                         sort_by_class=False,
                         use_float64=False,
                         mean_subtraction=False,
                         random_roated_labels=False):
    """Select subset of MNIST and apply preprocessing."""
    np.random.seed(seed)
    digits.sort()

    num_class = len(digits)
    num_per_class = num_train // num_class

    idx_list = np.array([], dtype='uint8')

    train, val, test = datasets
    ys = np.array([d[1] for d in train], dtype=np.int64)

    for digit in digits:
        if len(train) == num_train:
            idx_list = np.concatenate((idx_list, np.where(ys == digit)[0]))
        else:
            idx_list = np.concatenate((idx_list,
                                    np.where(ys == digit)[0][:num_per_class]))
    if not sort_by_class:
        np.random.shuffle(idx_list)

    data_precision = np.float64 if use_float64 else np.float32

    train_image = np.reshape(np.array([train[i][0].numpy() for i in idx_list]), 
        [idx_list.shape[0], -1]).astype(data_precision)
    train_label = np.array([train[i][1] for i in idx_list])
    train_label = np.eye(num_class, dtype=data_precision)[train_label]

    valid_image = np.reshape(np.array([val[i][0].numpy() for i in idx_list]), 
        [idx_list.shape[0], -1]).astype(data_precision)
    valid_label = np.array([val[i][1] for i in idx_list])
    valid_label = np.eye(num_class, dtype=data_precision)[valid_label]

    test_image = np.reshape(np.array([test[i][0].numpy() for i in range(len(test))]), 
        [len(test), -1]).astype(data_precision)
    test_label = np.array([test[i][1] for i in range(len(test))])
    test_label = np.eye(num_class, dtype=data_precision)[test_label]

    if sort_by_class:
        train_idx = np.argsort(np.argmax(train_label, axis=1))
        train_image = train_image[train_idx]
        train_label = train_label[train_idx]

    if mean_subtraction:
        train_image_mean = np.mean(train_image)
        train_label_mean = np.mean(train_label)
        train_image -= train_image_mean
        train_label -= train_label_mean
        valid_image -= train_image_mean
        valid_label -= train_label_mean
        test_image -= train_image_mean
        test_label -= train_label_mean

    if random_roated_labels:
        r, _ = np.linalg.qr(np.random.rand(10, 10))
        train_label = np.dot(train_label, r)
        valid_label = np.dot(valid_label, r)
        test_label = np.dot(test_label, r)

    return (train_image, train_label,
            valid_image, valid_label,
            test_image, test_label)

