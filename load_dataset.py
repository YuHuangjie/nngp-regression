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

def load_training_set(paths):
    x, y = [], []

    for p in paths:
        with open(p, 'rb') as f:
            x.append(np.squeeze(np.load(f)))
            y.append(np.squeeze(np.load(f)))

    x, y = np.vstack(x), np.vstack(y)

    return (x, y)

def load_test_set(paths):
    x, y, mask = [], [], []

    for p in paths:
        with open(p, 'rb') as f:
            x.append(np.squeeze(np.load(f)))
            y.append(np.squeeze(np.load(f)))
            mask.append(np.squeeze(np.load(f)))

    return (x, y, mask)
