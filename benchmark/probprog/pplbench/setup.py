# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import sys

from setuptools import find_packages, setup


REQUIRED_MAJOR = 3
REQUIRED_MINOR = 7

PPLS_REQUIRE = [
    "pystan==2.19.1.1",
    "pymc3>=3.11.0",
    "pyro-ppl>=0.4.1",
    "numpyro>=0.3.0",
    "beanmachine>=0.2.0",
]
DEV_REQUIRE = PPLS_REQUIRE + ["black==22.3.0", "flake8", "mypy", "usort"]


# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need "
        "python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)


# get version string from module
current_dir = os.path.dirname(__file__)
init_file = os.path.join(current_dir, "pplbench", "__init__.py")
version_regexp = r"__version__ = ['\"]([^'\"]*)['\"]"
with open(init_file, "r") as f:
    version = re.search(version_regexp, f.read(), re.M).group(1)

# read in README.md as the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pplbench",
    version=version,
    description="Evaluation framework for probabilistic programming languages",
    author="Facebook, Inc.",
    license="MIT",
    project_urls={
        "Documentation": "https://pplbench.org",
        "Source": "https://github.com/facebookresearch/pplbench",
    },
    keywords=[
        "Probabilistic Programming Language",
        "Bayesian Inference",
        "Statistical Modeling",
        "MCMC",
        "PyTorch",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=[
        "jsonargparse>=4.0.0",
        "jsonschema>=3.2.0",
        "numpy>=1.18.5",
        "scipy>=1.5.0",
        "pandas>=1.0.1",
        "matplotlib>=3.1.3",
        "xarray>=0.16.0",
        "arviz>=0.11.0",
    ],
    packages=find_packages(),
    extras_require={"dev": DEV_REQUIRE, "ppls": PPLS_REQUIRE},
    entry_points={"console_scripts": ["pplbench=pplbench.__main__:console_main"]},
)
