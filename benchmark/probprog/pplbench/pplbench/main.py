# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from types import SimpleNamespace
from typing import List, Optional
from unittest import mock

from jsonargparse import ActionJsonSchema, ArgumentParser, dict_to_namespace

from .lib import model_helper, ppl_helper, reports, utils


# The following schema defines the experiment that PPLBench should run.
SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "type": "object",
            "properties": {
                "class": {"type": "string"},
                "args": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "minimum": 2},
                        "k": {"type": "integer", "minimum": 1},
                    },
                    "required": ["n"],
                    "additionalProperties": True,
                    "propertyNames": {"pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
                },
                "seed": {"type": "integer", "minimum": 1},
                "package": {"type": "string"},  # defaults to pplbench.models
            },
            "required": ["class"],
            "additionalProperties": False,
        },
        "iterations": {"type": "integer", "minimum": 1},
        "num_warmup": {"type": "integer", "minimum": 0},
        "trials": {"type": "integer", "minimum": 2},
        "profile": {"type": "boolean"},
        "profile_run": {"type": "boolean"},
        "num_profiled": {"type": "integer"},
        "strip_profiled_names": {"type": "boolean"},
        "profiling_tools_dir": {"type": "string"},
        "profiling_type": {"type": "string", "enum": ["deterministic", "statistical"]},
        "ppls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "inference": {
                        "type": "object",
                        "properties": {
                            "class": {"type": "string"},
                            "num_warmup": {"type": "integer", "minimum": 0},
                            "compile_args": {
                                "type": "object",
                                "propertyNames": {"pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
                                "additionalProperties": True,
                            },
                            "infer_args": {
                                "type": "object",
                                "propertyNames": {"pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
                                "additionalProperties": True,
                            },
                        },
                        "required": ["class"],
                        "additionalProperties": False,
                    },
                    "legend": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "color": {"type": "string"},
                        },
                    },
                    "seed": {"type": "integer", "minimum": 1},
                    "trials": {"type": "integer", "minimum": 1},
                    # package defaults to pplbench.ppls.<ppl name>
                    "package": {"type": "string"},
                    "optional": {"type": "boolean"},
                },
                "required": ["name", "inference"],
                "additionalProperties": False,
            },
        },
        "loglevels": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            },
        },
        "save_samples": {"type": "boolean"},  # default "false"
        "output_root_dir": {"type": "string"},  # defaults to "./outputs"
        "figures": {
            "type": "object",
            "properties": {
                "generate_pll": {"type": "boolean"},  # default "true"
                "suffix": {"type": "string"},  # default "png"
            },
            "additionalProperties": False,
        },
    },
    "required": ["model", "ppls", "iterations", "trials"],
    "additionalProperties": False,
}

LOGGER = logging.getLogger("pplbench")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def main(args: Optional[List[str]] = None) -> None:
    if args is None:
        args = sys.argv[1:]

    validate = "--validate" in args
    if validate:
        args = [a for a in args if a != "--validate"]

    ppls_filter = None
    include_optional = False
    if "--ppls-all" in args:
        idx = args.index("--ppls-all")
        ppls_filter = args[idx + 1].split(",")
        include_optional = True
        args = args[:idx] + args[idx + 2:]
    elif "--ppls" in args:
        idx = args.index("--ppls")
        ppls_filter = args[idx + 1].split(",")
        args = args[:idx] + args[idx + 2:]

    config = read_config(args)

    # Strip optional entries (e.g. "Impulse (no opt)") unless --ppls-all
    if not include_optional:
        config.ppls = [
            p for p in config.ppls
            if not getattr(p, "optional", False)
        ]

    if ppls_filter:
        _apply_ppls_filter(config, ppls_filter)

    if validate:
        _apply_validate_overrides(config)

    # Propagate PPLBENCH_PROFILE env var into config so ppl_helper
    # profiles trial 1 in server mode (in addition to the initial run
    # which harness.jl/numpyro_harness.py handle via the env var directly).
    if os.environ.get("PPLBENCH_PROFILE"):
        config.profile = True

    if os.environ.get("PPLBENCH_DUMP_MLIR"):
        config.trials = 1
        for ppl in config.ppls:
            if hasattr(ppl, "trials"):
                ppl.trials = 1

    output_dir = utils.create_output_dir(config)
    os.environ["PPLBENCH_OUTPUT_DIR"] = output_dir
    avail = len(os.sched_getaffinity(0))
    total = os.cpu_count()
    print(f"CPUs: {avail} available / {total} total")
    print(f"Output dir: {output_dir}")
    if validate:
        print("Validate mode: NumPyro vs Impulse (5 samples, no warmup, no adaptation)")
    if os.environ.get("PPLBENCH_DUMP_MLIR"):
        print(f"MLIR dump: {os.path.join(output_dir, 'mlir_dumps')}")
    if os.environ.get("PPLBENCH_PROFILE"):
        print(f"XLA profile: {os.path.join(output_dir, 'traces')}")
    configure_logging(config, output_dir)

    if hasattr(config, "profile_run") and config.profile_run:
        from .lib.ppl_profiler import PPL_Profiler
        ppl_profiler = PPL_Profiler(output_dir, config)

    model_cls = model_helper.find_model_class(config.model)
    all_ppl_details = ppl_helper.find_ppl_details(config)
    train_data, test_data = model_helper.simulate_data(config.model, model_cls)
    (
        all_variable_metrics_data,
        all_other_metrics_data,
    ) = ppl_helper.collect_samples_and_stats(config, model_cls, all_ppl_details, train_data, test_data, output_dir)

    if validate:
        _run_validation(output_dir, all_ppl_details)
    else:
        reports.generate_plots(
            output_dir, config, all_ppl_details,
            all_variable_metrics_data, all_other_metrics_data,
        )

    if hasattr(config, "profile_run") and config.profile_run:
        ppl_profiler.finish_profiling()

    LOGGER.info(f"Output saved in '{output_dir}'")


def _apply_ppls_filter(config, ppls_filter):
    """Keep only PPL entries whose name matches one of the filter values."""
    kept = [ppl for ppl in config.ppls if ppl.name in ppls_filter]
    if not kept:
        raise RuntimeError(
            f"--ppls filter {ppls_filter} matched none of the config's PPLs: "
            f"{[p.name for p in config.ppls]}"
        )
    config.ppls = kept


def _apply_validate_overrides(config):
    """Override config for validation: 5 samples, no warmup, no adaptation, 2 trials."""
    config.iterations = 5
    config.num_warmup = 0
    config.trials = 2
    # Keep only numpyro and impulse (first occurrence of each)
    kept = []
    seen = set()
    for ppl in config.ppls:
        if ppl.name in ("numpyro", "impulse") and ppl.name not in seen:
            seen.add(ppl.name)
            # Disable adaptation
            if hasattr(ppl.inference, "infer_args"):
                ppl.inference.infer_args.adapt_step_size = False
                ppl.inference.infer_args.adapt_mass_matrix = False
            kept.append(ppl)
    if len(kept) < 2:
        raise RuntimeError(
            "Validate mode requires both 'numpyro' and 'impulse' entries in config"
        )
    config.ppls = kept


def _run_validation(output_dir, all_ppl_details):
    """Compare NumPyro vs Impulse sample outputs for numerical agreement."""
    import glob
    import json
    import numpy as np

    sample_files = sorted(glob.glob(os.path.join(output_dir, "samples*.nc")))
    if not sample_files:
        LOGGER.warning("No sample files found for validation comparison")
        return

    import xarray as xr
    ds = xr.open_dataset(sample_files[0])

    names = [p.name for p in all_ppl_details]
    if len(names) < 2:
        LOGGER.warning("Need at least 2 PPLs for validation")
        return

    ppl_a, ppl_b = names[0], names[1]
    print(f"\n=== Validation: {ppl_a} vs {ppl_b} ===")
    all_close = True
    for var in ds.data_vars:
        a = ds[var].sel(ppl=ppl_a).values
        b = ds[var].sel(ppl=ppl_b).values
        if np.isnan(a).all() or np.isnan(b).all():
            continue
        close = np.allclose(a, b, atol=1e-4, rtol=1e-3)
        max_diff = np.nanmax(np.abs(a - b))
        status = "PASS" if close else "FAIL"
        print(f"  {var}: {status} (max_diff={max_diff:.6e})")
        if not close:
            all_close = False

    if all_close:
        print("Validation PASSED: all variables match within tolerance")
    else:
        print("Validation FAILED: some variables differ beyond tolerance")
        sys.exit(1)


def read_config(args: Optional[List[str]]) -> SimpleNamespace:
    """
    Parse command line arguments and return a JSON object.
    :returns: benchmark configuration.
    """
    parser = ArgumentParser()
    parser.add_argument("config", action=ActionJsonSchema(schema=SCHEMA), help="%s")
    config = parser.parse_args(args).config
    with mock.patch("jsonargparse.namespace.Namespace", SimpleNamespace):
        config = dict_to_namespace(config)

    # default num_warmup to half of num_sample
    if not hasattr(config, "num_warmup"):
        config.num_warmup = config.iterations // 2

    return config


def configure_logging(config: SimpleNamespace, output_dir: str) -> None:
    """
    Configure logging based on `config.loglevel` and add a stream handler.
    :param config: benchmark configuration
    :output_dir: directory to save the output
    """
    # set log level to INFO by default on root logger
    logging.getLogger().setLevel("INFO")
    # setup logging for all other requested loglevels
    if hasattr(config, "loglevels"):
        for key, val in config.loglevels.__dict__.items():
            logging.getLogger(key).setLevel(getattr(logging, val))
    # create a handler at the root level to display to stdout
    # and another to write to a log file
    for ch in [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(output_dir, "logging.txt"), encoding="utf-8"),
    ]:
        formatter = logging.Formatter(LOG_FORMAT)
        ch.setFormatter(formatter)
        logging.getLogger().addHandler(ch)
    LOGGER.debug(f"config - {str(config)}")
