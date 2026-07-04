# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod


class BasePPLImplementation(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """
        Initialize the model representation in the PPL.
        """
        pass
