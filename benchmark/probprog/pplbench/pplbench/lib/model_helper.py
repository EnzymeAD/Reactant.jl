# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from types import SimpleNamespace
from typing import Tuple, Type

import xarray as xr

from ..models.base_model import BaseModel
from .utils import load_class_or_exit


LOGGER = logging.getLogger("pplbench")


def find_model_class(mconfig: SimpleNamespace) -> Type[BaseModel]:
    """
    Finds the model class in `pplbench.models` package.
    object of the class with the optional mconfig.args dict.
    :param: mconfig: An JSON object from the config file.
    :returns: An instantiated model class object.
    """
    # find the model
    package = getattr(mconfig, "package", "pplbench.models")
    class_name = getattr(mconfig, "class")
    mclass = load_class_or_exit(package + "." + class_name)
    return mclass


def simulate_data(
    mconfig: SimpleNamespace, model_cls: Type[BaseModel]
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    :param mconfig: benchrmark `model` config
    :param model_cls: model implementation class
    :returns: Training and test datasets
    """
    seed = getattr(mconfig, "seed", int(time.time()))
    LOGGER.info(f"model seed {seed}")
    kwargs = mconfig.args.__dict__ if hasattr(mconfig, "args") else {}
    kwargs["seed"] = seed
    train_data, test_data = model_cls.generate_data(**kwargs)
    LOGGER.debug(
        f"data generated: train n = {train_data.attrs['n']} test rows "
        f"= {test_data.attrs['n']} hyper params = '{train_data.attrs}'"
    )
    return train_data, test_data
