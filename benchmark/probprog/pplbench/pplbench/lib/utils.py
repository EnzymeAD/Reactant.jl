# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import io
import json
import logging
import os
import pydoc
import sys
import time
from types import SimpleNamespace
from typing import Any

import jsonargparse
import xarray as xr


LOGGER = logging.getLogger("pplbench")


def load_class_or_exit(class_name: str) -> Any:
    """
    Load the given `class_name` or exit the process.
    :param class_name: The class to be loaded.
    :returns: a class object
    """
    class_ = pydoc.locate(class_name)
    if class_ is None:
        LOGGER.error(f"class `{class_name}` not found. Exiting!`")
        sys.exit(1)
    LOGGER.debug(f"loaded class `{class_}`")
    return class_


class SimpleNamespaceEncoder(json.JSONEncoder):
    """define class for encoding config object"""

    def default(self, object):
        if isinstance(object, SimpleNamespace):
            return object.__dict__
        elif hasattr(object, "__dict__") and type(object).__name__ == "Namespace":
            return object.__dict__
        elif isinstance(object, jsonargparse.Path):
            return {}
        else:
            # call base class implementation which takes care of
            # raising exceptions for unsupported types
            return json.JSONEncoder.default(self, object)


def create_output_dir(config: SimpleNamespace) -> str:
    """
    Create an output directory for storing the benchmark results and write the config file to it.
    :param config: the experiment configuration
    :returns: a directory name for storing outputs
    """
    root_dir = getattr(config, "output_root_dir", os.path.join(".", "outputs"))
    if not os.path.isdir(root_dir):
        os.mkdir(os.path.join(".", "outputs"))
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
        "%Y-%m-%d_%H:%M:%S"
    )
    output_dir = create_subdir(root_dir, timestamp)
    # dump the config file in the output directory
    with open(os.path.join(output_dir, "config.json"), "w") as fp:
        fp.write(convert_to_readable_string(SimpleNamespaceEncoder().encode(config)))
    # redirect C stdout and stderr to files to avoid cluttering user's display
    # but keep Python stdout and stderr intact
    for sname in ["stdout", "stderr"]:
        py_stream = getattr(sys, sname)
        # save the current stream's file descriptor
        saved_fd = os.dup(py_stream.fileno())
        # redirect the current stream's file descriptor to a log file
        log_fd = os.open(
            os.path.join(output_dir, f"{sname}.txt"), os.O_WRONLY | os.O_CREAT
        )
        os.dup2(log_fd, py_stream.fileno())
        # now restor the Python stream to the saved file descriptor
        setattr(sys, sname, io.TextIOWrapper(os.fdopen(saved_fd, "wb")))
    return output_dir


def create_subdir(output_dir: str, name: str) -> str:
    """
    Create a subdirectory and return its name.
    :param output_dir: directory to make a subdir under
    :name: subdir name
    :returns: name of new subdirectory
    """
    subdir_name = os.path.join(output_dir, name)
    os.mkdir(subdir_name)
    LOGGER.debug(f"Created subdir {subdir_name}")
    return subdir_name


def save_dataset(output_dir: str, name_prefix: str, ds: xr.Dataset) -> None:
    """
    Saves the dataset in both NetCDF binary format as well as a readable CSV.
    The NetCDF files will be written to `output_dir/name_prefix.nc`
    A separate CSV file will be written to `output_dir/name_prefix/varname.csv`
    for each data variable `varname` in the dataset.
    :param output_dir: directory to save dataset
    :param name_prefix: prefix to be added before file names
    :param ds: dataset to be saved
    """
    # Ensure string coordinates use object dtype for netCDF4 compatibility
    # (numpy 2.x StringDType is not supported by netCDF4)
    for coord_name, coord_val in ds.coords.items():
        if coord_val.dtype.kind in ("U", "T") or "str" in str(coord_val.dtype).lower():
            ds = ds.assign_coords({coord_name: coord_val.values.astype(object)})
    ds.to_netcdf(os.path.join(output_dir, name_prefix + ".nc"))
    subdir = create_subdir(output_dir, name_prefix)
    for varname in ds.data_vars.keys():
        getattr(ds, str(varname)).to_dataframe().to_csv(
            os.path.join(subdir, f"{varname}.csv")
        )
    LOGGER.info(f"saved {name_prefix}")


def convert_to_readable_string(input: str):
    intermediate = ""
    for character in input:
        if character == " ":
            pass
        else:
            intermediate = intermediate + character

    indent = "   "
    indent_level = 0
    output = ""
    for character in intermediate:
        if character == "}" or character == "]":
            indent_level = indent_level - 1
            output = output + "\n" + indent * indent_level + character
        elif character == "{" or character == "[":
            indent_level = indent_level + 1
            output = output + character + "\n" + indent * indent_level
        elif character == ",":
            output = output + character + "\n" + indent * indent_level
        elif character == ":":
            output = output + character + " "
        else:
            output = output + character
    output = output + "\n"
    return output
