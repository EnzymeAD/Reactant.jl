# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import numpy as np
import xarray as xr


def log1pexp(x: np.ndarray) -> np.ndarray:
    """
    Compute log(1 + exp(x)) in a numerically stable way,
    see https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf eqn 10
    :param x: numpy array of numbers
    :returns: log(1 + exp(x))
    """
    y = np.zeros_like(x)
    y[x < 18] = np.log1p(np.exp(x[x < 18]))
    y[x >= 18] = x[x >= 18] + np.exp(-x[x >= 18])
    return y


def log1mexpm(x: np.ndarray) -> np.ndarray:
    """
    Compute log(1 - exp(-x)) in a numerically stable way for x > 0,
    see https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf eqn 7
    :param x: numpy array of numbers >= 0
    :returns: log(1 - exp(-x))
    """
    y = np.zeros_like(x)
    x_low, x_high = x < 0.693, x >= 0.693
    y[x_low] = np.log(-(np.expm1(-x[x_low])))
    y[x_high] = np.log1p(-(np.exp(-x[x_high])))
    return y


def split_train_test(
    data: xr.Dataset, coord_name: str, train_frac: float
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Splice a dataset into two along the given coordinate
    and update n in the attributes.

    :param data: A dataset object which is to be split
    :param coord_name: The coordinate on which the data is going to be sliced
    :param train_frac: Fraction of data to be given to training
    :returns: The training and test datasets.
    """
    num_train = int(train_frac * len(data.coords[coord_name]))
    train = data[{coord_name: slice(None, num_train)}]
    test = data[{coord_name: slice(num_train, None)}]
    train.attrs = data.attrs.copy()
    train.attrs["n"] = num_train
    test.attrs = data.attrs.copy()
    test.attrs["n"] = data.attrs["n"] - train.attrs["n"]
    return train, test
