#!/usr/bin/env python3
"""
Collect benchmark data for the Impulse paper evaluation section.

Usage:
  # Run all benchmarks from scratch (takes a long time):
  python collect_paper_data.py

  # Run only standard benchmarks:
  python collect_paper_data.py --suite standard

  # Run only SICM benchmarks:
  python collect_paper_data.py --suite sicm

  # Read from existing PPLBench output directories:
  python collect_paper_data.py --from-outputs outputs/2026-03-07_12:00:00 outputs/...

  # Run only validation (correctness) data:
  python collect_paper_data.py --validate-only

  # Skip validation and only run performance benchmarks:
  python collect_paper_data.py --no-validate

Output: paper_data.json
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(os.path.dirname(sys.executable), "python")

# All benchmark configs grouped by category
STANDARD_CONFIGS = [
    "standard/logistic_regression",
    "standard/gaussian_process",
    "standard/robust_regression",
    "standard/n_schools",
]

SICM_CONFIGS = [
    "sicm/hierarchical_mvn",
    "sicm/scale_family_mvn",
    "sicm/gp_pois_regr_scaled",
    "sicm/gp_classify",
    "sicm/gp_fixed_lengthscale",
    "sicm/phylogenetic_regression",
    "sicm/linear_mixed_model",
    "sicm/stochastic_volatility",
    "sicm/car_spatial",
]

MOTIVATING_CONFIGS = [
    "motivating/gp_pois_regr",
]

ALL_BENCHMARK_CONFIGS = STANDARD_CONFIGS + MOTIVATING_CONFIGS + SICM_CONFIGS

VALIDATE_CONFIGS = [
    "standard/logistic_regression",
    "standard/gaussian_process",
    "standard/robust_regression",
    "standard/n_schools",
    "motivating/gp_pois_regr",
]

# Map model_class keys to config names for --from-outputs lookup
MODEL_CLASS_TO_CONFIG = {
    "logistic_regression.LogisticRegression": "standard/logistic_regression",
    "gaussian_process.GaussianProcess": "standard/gaussian_process",
    "robust_regression.RobustRegression": "standard/robust_regression",
    "n_schools.NSchools": "standard/n_schools",
    "gp_pois_regr.GPPoisRegr": "motivating/gp_pois_regr",
    "hierarchical_mvn.HierarchicalMVN": "sicm/hierarchical_mvn",
    "scale_family_mvn.ScaleFamilyMVN": "sicm/scale_family_mvn",
    "gp_pois_regr_scaled.GPPoisRegrScaled": "sicm/gp_pois_regr_scaled",
    "gp_classify.GPClassify": "sicm/gp_classify",
    "gp_fixed_lengthscale.GPFixedLengthscale": "sicm/gp_fixed_lengthscale",
    "phylogenetic_regression.PhylogeneticRegression": "sicm/phylogenetic_regression",
    "linear_mixed_model.LinearMixedModel": "sicm/linear_mixed_model",
    "stochastic_volatility.StochasticVolatility": "sicm/stochastic_volatility",
    "car_spatial.CARSpatial": "sicm/car_spatial",
}


def run_validate(config_name):
    """Run validation mode and parse output for max_diff values."""
    cmd = [os.path.join(SCRIPT_DIR, "run_pplbench.sh"), "--validate", config_name]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)

    stdout = result.stdout + result.stderr
    print(stdout)

    diffs = {}
    for line in stdout.split("\n"):
        line = line.strip()
        # Parse lines like: "  alpha: PASS (max_diff=0.000000e+00)"
        if "max_diff=" in line and ":" in line:
            parts = line.split(":")
            var_name = parts[0].strip()
            diff_str = line.split("max_diff=")[1].rstrip(")")
            diffs[var_name] = float(diff_str)

    return diffs


def read_metrics_from_output(output_dir):
    """Read timing metrics from a PPLBench output directory."""
    import xarray as xr

    metrics_path = os.path.join(output_dir, "metrics.nc")
    if not os.path.exists(metrics_path):
        print(f"Warning: {metrics_path} not found, skipping")
        return None

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    ds = xr.open_dataset(metrics_path)
    ppl_names = list(ds.coords["ppl"].values)
    timing = ds["timing"].values  # shape: (n_ppls, n_trials, 2)

    iterations = config["iterations"]
    num_warmup = config.get("num_warmup", iterations // 2)
    num_samples = iterations - num_warmup

    results = {}
    for i, name in enumerate(ppl_names):
        compile_times = timing[i, :, 0]
        infer_times = timing[i, :, 1]

        # compile_time: from trial 0 (where subprocess reports it)
        compile_time = compile_times[0]

        # per-iteration time: total infer / iterations, averaged over trials
        per_iter_ms = (infer_times / iterations) * 1000
        results[name] = {
            "compile_time_s": float(compile_time),
            "per_iter_ms_mean": float(np.mean(per_iter_ms)),
            "per_iter_ms_std": float(np.std(per_iter_ms)),
            "total_s_mean": float(np.mean(infer_times)),
            "total_s_std": float(np.std(infer_times)),
            "iterations": iterations,
            "num_warmup": num_warmup,
            "n_trials": len(infer_times),
        }

    # Extract n_eff and n_eff/s if available
    if "overall_neff" in ds:
        neff = ds["overall_neff"].values  # shape: (n_ppls, 3) for min/median/max
        neff_per_time = ds["overall_neff_per_time"].values
        for i, name in enumerate(ppl_names):
            if name in results:
                results[name]["neff_min"] = float(neff[i, 0])
                results[name]["neff_median"] = float(neff[i, 1])
                results[name]["neff_max"] = float(neff[i, 2])
                results[name]["neff_per_s_min"] = float(neff_per_time[i, 0])
                results[name]["neff_per_s_median"] = float(neff_per_time[i, 1])
                results[name]["neff_per_s_max"] = float(neff_per_time[i, 2])

    model_class = config["model"]["class"]
    model_args = config["model"].get("args", {})

    # Determine config name from model class
    config_name = MODEL_CLASS_TO_CONFIG.get(model_class, model_class)

    return {
        "config_name": config_name,
        "model_class": model_class,
        "model_args": model_args,
        "frameworks": results,
    }


def run_benchmark(config_name):
    """Run a full benchmark and return the output directory path."""
    cmd = [os.path.join(SCRIPT_DIR, "run_pplbench.sh"), config_name]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)

    stdout = result.stdout + result.stderr
    # Parse output dir from "Output dir: <path>"
    for line in stdout.split("\n"):
        if "Output dir:" in line:
            return line.split("Output dir:")[1].strip()

    print(f"Warning: could not find output dir in benchmark output")
    print(stdout[-500:])
    return None


def main():
    parser = argparse.ArgumentParser(description="Collect paper benchmark data")
    parser.add_argument(
        "--from-outputs", nargs="+",
        help="Read from existing PPLBench output directories instead of running benchmarks"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only run validation (correctness) mode, skip full benchmarks"
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip validation, only run performance benchmarks"
    )
    parser.add_argument(
        "--suite", choices=["all", "standard", "sicm", "motivating"],
        default="all",
        help="Which benchmark suite to run (default: all)"
    )
    parser.add_argument(
        "--output", default=os.path.join(SCRIPT_DIR, "paper_data.json"),
        help="Output JSON file path"
    )
    args = parser.parse_args()

    # Load existing data if present (to allow incremental updates)
    if os.path.exists(args.output):
        with open(args.output) as f:
            paper_data = json.load(f)
        # Ensure keys exist
        paper_data.setdefault("correctness", {})
        paper_data.setdefault("baselines", {})
        paper_data.setdefault("sicm", {})
    else:
        paper_data = {
            "correctness": {},
            "baselines": {},
            "sicm": {},
        }

    # Determine which configs to run
    if args.suite == "standard":
        benchmark_configs = STANDARD_CONFIGS
    elif args.suite == "sicm":
        benchmark_configs = SICM_CONFIGS
    elif args.suite == "motivating":
        benchmark_configs = MOTIVATING_CONFIGS
    else:
        benchmark_configs = ALL_BENCHMARK_CONFIGS

    # Correctness data (validate mode)
    if not args.no_validate:
        print("=" * 60)
        print("Collecting correctness data (validate mode)")
        print("=" * 60)
        for config_name in VALIDATE_CONFIGS:
            print(f"\n--- {config_name} ---")
            diffs = run_validate(config_name)
            paper_data["correctness"][config_name] = diffs
            for var, diff in sorted(diffs.items()):
                print(f"  {var}: {diff:.2e}")

    if args.validate_only:
        with open(args.output, "w") as f:
            json.dump(paper_data, f, indent=2)
        print(f"\nValidation data saved to {args.output}")
        return

    # Runtime data
    print("\n" + "=" * 60)
    print("Collecting runtime data")
    print("=" * 60)

    if args.from_outputs:
        for output_dir in args.from_outputs:
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(SCRIPT_DIR, output_dir)
            print(f"\nReading: {output_dir}")
            result = read_metrics_from_output(output_dir)
            if result:
                config_name = result["config_name"]
                # Route to baselines or sicm based on config category
                if config_name.startswith("sicm/"):
                    paper_data["sicm"][config_name] = result
                else:
                    paper_data["baselines"][config_name] = result
    else:
        for config_name in benchmark_configs:
            print(f"\n--- {config_name} ---")
            output_dir = run_benchmark(config_name)
            if output_dir:
                result = read_metrics_from_output(output_dir)
                if result:
                    if config_name.startswith("sicm/"):
                        paper_data["sicm"][config_name] = result
                    else:
                        paper_data["baselines"][config_name] = result

    with open(args.output, "w") as f:
        json.dump(paper_data, f, indent=2)
    print(f"\nAll data saved to {args.output}")


if __name__ == "__main__":
    main()
