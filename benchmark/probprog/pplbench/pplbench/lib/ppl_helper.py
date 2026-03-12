# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import logging
import os
import struct
import time
from types import SimpleNamespace
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

import arviz
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import is_color_like, to_rgb

from ..models.base_model import BaseModel
from ..ppls.base_ppl_impl import BasePPLImplementation
from ..ppls.base_ppl_inference import BasePPLInference
from .utils import load_class_or_exit, save_dataset


class PPLDetails(NamedTuple):
    name: str
    seed: int
    color: Tuple[float, float, float]
    impl_class: Type[BasePPLImplementation]
    inference_class: Type[BasePPLInference]
    num_warmup: Optional[int]
    compile_args: Dict
    infer_args: Dict
    trials: Optional[int]


LOGGER = logging.getLogger("pplbench")


def find_ppl_details(config: SimpleNamespace) -> List[PPLDetails]:
    """
    Returns information about each instance of PPL inference that is requested
    in the benchmark.
    This raises a Runtime exception if the names are not unique.
    :param config: The benchmark configuration object.
    :returns: A list of information objects one per ppl inference.
    """
    ret_val = []
    prev_names = set()
    model_class = getattr(config.model, "class")
    for ppl_config in config.ppls:
        package = getattr(ppl_config, "package", "pplbench.ppls." + ppl_config.name)
        impl_class = load_class_or_exit(f"{package}.{model_class}")
        inference_class = load_class_or_exit(
            f"{package}.{getattr(ppl_config.inference, 'class')}"
        )
        # create a unique name for the PPL inference if not given
        name = (
            ppl_config.legend.name
            if hasattr(ppl_config, "legend") and hasattr(ppl_config.legend, "name")
            else ppl_config.name + "-" + inference_class.__name__
        )
        if name in prev_names:
            raise RuntimeError(f"duplicate PPL inference {name}")
        prev_names.add(name)

        # we will generate a unique color for each ppl name
        # https://stackoverflow.com/questions/40351791/how-to-hash-strings-into-a-float-in-01
        def _hash(name):
            return (
                float(
                    struct.unpack(
                        "L", hashlib.sha256(bytes(name, "utf-8")).digest()[:8]
                    )[0]
                )
                / 2**64
            )

        def _get_color(name):
            return (_hash(name + "0"), _hash(name + "1"), _hash(name + "1"))

        if hasattr(ppl_config, "legend") and hasattr(ppl_config.legend, "color"):
            if is_color_like(ppl_config.legend.color):
                color = to_rgb(ppl_config.legend.color)
            else:
                raise RuntimeError(
                    f"invalid color '{ppl_config.legend.color}' for PPL inference '{name}'"
                )
        else:
            color = _get_color(name)
        # finally pick a default seed for the ppl
        seed = getattr(ppl_config, "seed", int(time.time() + 19))
        infer = ppl_config.inference
        ret_val.append(
            PPLDetails(
                name=name,
                seed=seed,
                color=color,
                impl_class=impl_class,
                inference_class=inference_class,
                num_warmup=getattr(infer, "num_warmup", None),
                compile_args=infer.compile_args.__dict__
                if hasattr(infer, "compile_args")
                else {},
                infer_args=infer.infer_args.__dict__
                if hasattr(infer, "infer_args")
                else {},
                trials=getattr(ppl_config, "trials", None),
            )
        )
        LOGGER.debug(f"added PPL inference '{str(ret_val[-1])}'")
    return ret_val


def _run_single_ppl(
    pplobj, config, model_cls, train_data, test_data, output_dir,
    compile_seed, shared_trial_seeds, max_trials,
    all_names, all_variable_metrics, all_pll, all_timing,
    all_samples, all_overall_neff, all_overall_neff_per_time,
):
    """Run all trials for a single PPL. Appends results to the provided lists.
    Raises on failure so the caller can catch and skip."""
    compile_t1 = time.time()
    infer_obj = pplobj.inference_class(pplobj.impl_class, train_data.attrs)
    infer_obj.compile(seed=compile_seed, **pplobj.compile_args)
    compile_time = time.time() - compile_t1
    LOGGER.info(f"compiling on `{pplobj.name}` took {compile_time:.2f} secs")

    if infer_obj.is_adaptive:
        num_warmup = (
            config.num_warmup if pplobj.num_warmup is None else pplobj.num_warmup
        )
        if num_warmup > config.iterations:
            raise ValueError(
                f"num_warmup ({num_warmup}) should be less than iterations "
                f"({config.iterations})"
            )
    else:
        if pplobj.num_warmup:
            raise ValueError(
                f"{pplobj.name} is not adaptive and does not accept a nonzero "
                "num_warmup as its parameter."
            )
        else:
            num_warmup = 0

    ppl_trials = pplobj.trials if pplobj.trials is not None else config.trials
    LOGGER.info(f"Running {ppl_trials} trials for {pplobj.name}")

    do_profile = getattr(config, "profile", False)
    trial_samples, trial_pll, trial_timing = [], [], []
    for trialnum in range(ppl_trials):
        trial_seed = shared_trial_seeds[trialnum]
        init_params = model_cls.generate_init_params(
            trial_seed, **train_data.attrs
        )
        profile_dir = None
        if do_profile and trialnum == 1:
            profile_dir = os.path.join(output_dir, "traces", pplobj.name)
            os.makedirs(profile_dir, exist_ok=True)
            LOGGER.info(
                f"Profiling trial {trialnum} for {pplobj.name} -> {profile_dir}"
            )
        infer_t1 = time.time()
        samples = infer_obj.infer(
            data=train_data,
            iterations=config.iterations,
            num_warmup=num_warmup,
            seed=trial_seed,
            init_params=init_params if init_params else None,
            profile_dir=profile_dir,
            **pplobj.infer_args,
        )
        infer_time = time.time() - infer_t1
        subprocess_run_time = getattr(infer_obj, 'last_run_time', None)
        subprocess_compile_time = getattr(infer_obj, 'last_compile_time', None)
        if subprocess_run_time is not None:
            infer_time = subprocess_run_time
            if subprocess_compile_time is not None:
                compile_time = subprocess_compile_time
        LOGGER.info(f"inference trial {trialnum} took {infer_time:.2f} secs")

        for var_name in sorted(samples.data_vars):
            vals = samples[var_name].values
            post = vals[num_warmup:]
            if post.ndim == 1:
                LOGGER.info(f"  {pplobj.name} trial {trialnum} {var_name} (first 5): {post[:5]}")
            else:
                LOGGER.info(f"  {pplobj.name} trial {trialnum} {var_name}[:,0] (first 5): {post[:5, 0]}")

        valid_samples = samples.dropna("draw")
        if samples.sizes["draw"] != config.iterations:
            raise RuntimeError(
                f"Expected {config.iterations} samples, but {samples.sizes['draw']}"
                f" samples are returned by {pplobj.name}"
            )

        persample_pll = model_cls.evaluate_posterior_predictive(
            valid_samples, test_data
        )
        pll = np.logaddexp.accumulate(persample_pll) - np.log(
            np.arange(valid_samples.sizes["draw"]) + 1
        )
        LOGGER.info(f"PLL = {str(pll)}")
        trial_samples.append(samples)

        padded_pll = np.full(config.iterations, np.nan)
        padded_pll[valid_samples.draw.data] = pll
        trial_pll.append(padded_pll)
        trial_timing.append([compile_time, infer_time])
        infer_obj.additional_diagnostics(output_dir, f"{pplobj.name}_{trialnum}")
    del infer_obj

    # Compute metrics from real trials only
    trial_samples_data = xr.concat(
        trial_samples, pd.Index(data=np.arange(ppl_trials), name="chain")
    )
    trial_samples_no_warmup = trial_samples_data.isel(draw=slice(num_warmup, None))

    neff_data = arviz.ess(trial_samples_no_warmup)
    rhat_data = arviz.rhat(trial_samples_no_warmup)
    LOGGER.info(f"Trials completed for {pplobj.name} ({ppl_trials} trials)")
    LOGGER.info("== n_eff ===")
    LOGGER.info(str(neff_data.data_vars))
    LOGGER.info("==  Rhat ===")
    LOGGER.info(str(rhat_data.data_vars))

    neff_values = np.concatenate([np.atleast_1d(neff_data[v].values).ravel() for v in neff_data.data_vars])
    overall_neff = [
        neff_values.min(),
        np.median(neff_values),
        neff_values.max(),
    ]
    trial_times = np.array(trial_timing)[:, 1]
    steady_times = trial_times[1:] if len(trial_times) > 1 else trial_times
    mean_inference_time = np.mean(steady_times)
    overall_neff_per_time = np.array(overall_neff) / mean_inference_time

    LOGGER.info("== all trial times (s) ===")
    LOGGER.info(str(trial_times))
    LOGGER.info("== steady-state trial times (excluding trial 0) ===")
    LOGGER.info(str(steady_times))
    LOGGER.info("== overall n_eff [min, median, max]===")
    LOGGER.info(str(overall_neff))
    LOGGER.info("== overall n_eff/s [min, median, max]===")
    LOGGER.info(str(overall_neff_per_time))

    # Pad to max_trials for uniform data dimensions across PPLs
    if ppl_trials < max_trials:
        trial_samples_data = trial_samples_data.reindex(
            chain=np.arange(max_trials)
        )
        while len(trial_pll) < max_trials:
            trial_pll.append(np.full(config.iterations, np.nan))
        while len(trial_timing) < max_trials:
            trial_timing.append([float('nan'), float('nan')])

    trial_variable_metrics_data = xr.concat(
        [neff_data, rhat_data], pd.Index(data=["n_eff", "Rhat"], name="metric")
    )
    # Only append to result lists after all trials succeed
    all_names.append(pplobj.name)
    all_variable_metrics.append(trial_variable_metrics_data)
    all_pll.append(trial_pll)
    all_timing.append(trial_timing)
    all_samples.append(trial_samples_data)
    all_overall_neff.append(overall_neff)
    all_overall_neff_per_time.append(overall_neff_per_time)


def collect_samples_and_stats(
    config: SimpleNamespace,
    model_cls: Type[BaseModel],
    all_ppl_details: List[PPLDetails],
    train_data: xr.Dataset,
    test_data: xr.Dataset,
    output_dir: str,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    :param confg: The benchmark configuration.
    :param model_cls: The model class
    :param ppl_details: For each ppl the the impl and inference classes etc.
    :param train_data: The training dataset.
    :param test_data: The held-out test dataset.
    :param output_dir: The directory for storing results.
    :returns: Two datasets:
        variable_metrics
            Coordinates: ppl, metric (n_eff, Rhat), others from model
            Data variables: from model
        other_metrics
            Coordinates: ppl, chain, draw, phase (compile, infer)
            Data variables: pll (ppl, chain, draw), timing (ppl, chain, phase)
    """
    all_variable_metrics, all_pll, all_timing, all_names = [], [], [], []
    all_samples, all_overall_neff, all_overall_neff_per_time = [], [], []

    # Compute max trials across all PPLs (per-PPL trials override global)
    max_trials = max(
        pplobj.trials if pplobj.trials is not None else config.trials
        for pplobj in all_ppl_details
    )

    # Pre-generate shared trial seeds so all PPLs get identical seeds per trial
    shared_seed = config.model.seed if hasattr(config.model, "seed") else int(time.time())
    shared_rand = np.random.RandomState(shared_seed)
    _compile_seed = shared_rand.randint(1, int(1e7))
    trial_seed = shared_rand.randint(1, int(1e7))
    shared_trial_seeds = [trial_seed] * max_trials
    LOGGER.info(f"Shared trial seeds (from model seed {shared_seed}): {shared_trial_seeds}")

    failed_ppls = []
    for pplobj in all_ppl_details:
        LOGGER.info(f"Starting inference on `{pplobj.name}` with seed {pplobj.seed}")
        try:
            _run_single_ppl(
                pplobj, config, model_cls, train_data, test_data, output_dir,
                _compile_seed, shared_trial_seeds, max_trials,
                all_names, all_variable_metrics, all_pll, all_timing,
                all_samples, all_overall_neff, all_overall_neff_per_time,
            )
        except Exception:
            LOGGER.exception(f"PPL `{pplobj.name}` failed, skipping")
            failed_ppls.append(pplobj.name)
            continue
    if failed_ppls:
        LOGGER.warning(f"Failed PPLs: {failed_ppls}")
    if not all_names:
        raise RuntimeError("All PPLs failed, no results to save")
    # merge the trial-level metrics at the PPL level
    all_variable_metrics_data = xr.concat(
        all_variable_metrics, pd.Index(data=all_names, name="ppl")
    )
    all_other_metrics_data = xr.Dataset(
        {
            "timing": (["ppl", "chain", "phase"], all_timing),
            "pll": (["ppl", "chain", "draw"], all_pll),
            "overall_neff": (["ppl", "percentile"], all_overall_neff),
            "overall_neff_per_time": (["ppl", "percentile"], all_overall_neff_per_time),
        },
        coords={
            "ppl": np.array(all_names),
            "chain": np.arange(max_trials),
            "phase": np.array(["compile", "infer"]),
            "draw": np.arange(config.iterations),
            "percentile": np.array(["min", "median", "max"]),
        },
    )
    all_samples_data = xr.concat(all_samples, pd.Index(data=all_names, name="ppl"))
    model_cls.additional_metrics(output_dir, all_samples_data, train_data, test_data)
    LOGGER.info("all benchmark samples and metrics collected")
    # save the samples data only if requested
    if getattr(config, "save_samples", False):
        save_dataset(output_dir, "samples", all_samples_data)
    # write out the metrics
    save_dataset(output_dir, "diagnostics", all_variable_metrics_data)
    save_dataset(output_dir, "metrics", all_other_metrics_data)
    return all_variable_metrics_data, all_other_metrics_data
