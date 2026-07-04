#!/usr/bin/env python3
"""
Collect ablation benchmark results into ablation_data.json.

Usage:
  # Run benchmarks and collect:
  python collect_ablation_data.py

  # Read from existing output directories:
  python collect_ablation_data.py --from-outputs outputs/2026-03-13_* ...

  # Run a single model:
  python collect_ablation_data.py --model logistic_regression

Output: ablation_data.json
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)

ABLATION_MODELS = [
    "logistic_regression",
    "robust_regression",
    "n_schools",
    "gp_pois_regr_scaled",
    "gp_fixed_lengthscale",
    "linear_mixed_model",
    "stochastic_volatility",
]

# Legend name in config -> ablation column key
ABLATION_CONFIGS = ["Full", "No SICM", "No opt"]

# model_class -> ablation model name
MODEL_CLASS_TO_NAME = {
    "logistic_regression.LogisticRegression": "logistic_regression",
    "robust_regression.RobustRegression": "robust_regression",
    "n_schools.NSchools": "n_schools",
    "gp_pois_regr_scaled.GPPoisRegrScaled": "gp_pois_regr_scaled",
    "gp_fixed_lengthscale.GPFixedLengthscale": "gp_fixed_lengthscale",
    "linear_mixed_model.LinearMixedModel": "linear_mixed_model",
    "stochastic_volatility.StochasticVolatility": "stochastic_volatility",
}


def read_metrics_from_output(output_dir):
    """Read timing metrics from a PPLBench output directory."""
    import xarray as xr

    metrics_path = os.path.join(output_dir, "metrics.nc")
    if not os.path.exists(metrics_path):
        print(f"  Warning: {metrics_path} not found, skipping")
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

        compile_time = compile_times[0]
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

    if "overall_neff" in ds:
        neff = ds["overall_neff"].values
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
    model_name = MODEL_CLASS_TO_NAME.get(model_class)

    return {
        "model_name": model_name,
        "model_class": model_class,
        "model_args": config["model"].get("args", {}),
        "configs": results,
    }


def run_benchmark(model_name):
    """Run an ablation benchmark and return the output directory."""
    cmd = [os.path.join(BENCHMARK_DIR, "run_pplbench.sh"), f"ablation/{model_name}"]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BENCHMARK_DIR)

    stdout = result.stdout + result.stderr
    for line in stdout.split("\n"):
        if "Output dir:" in line:
            return line.split("Output dir:")[1].strip()

    print(f"  Warning: could not find output dir")
    print(stdout[-500:])
    return None


def main():
    parser = argparse.ArgumentParser(description="Collect ablation benchmark data")
    parser.add_argument(
        "--from-outputs", nargs="+",
        help="Read from existing PPLBench output directories"
    )
    parser.add_argument(
        "--model", type=str,
        help="Run only a single model (e.g., logistic_regression)"
    )
    parser.add_argument(
        "--output", default=os.path.join(SCRIPT_DIR, "ablation_data.json"),
        help="Output JSON file path"
    )
    args = parser.parse_args()

    # Load existing data for incremental updates
    if os.path.exists(args.output):
        with open(args.output) as f:
            ablation_data = json.load(f)
    else:
        ablation_data = {}

    models = [args.model] if args.model else ABLATION_MODELS

    if args.from_outputs:
        for output_dir in args.from_outputs:
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(BENCHMARK_DIR, output_dir)
            print(f"Reading: {output_dir}")
            result = read_metrics_from_output(output_dir)
            if result and result["model_name"]:
                ablation_data[result["model_name"]] = result
    else:
        for model_name in models:
            print(f"\n--- {model_name} ---")
            output_dir = run_benchmark(model_name)
            if output_dir:
                if not os.path.isabs(output_dir):
                    output_dir = os.path.join(BENCHMARK_DIR, output_dir)
                result = read_metrics_from_output(output_dir)
                if result and result["model_name"]:
                    ablation_data[result["model_name"]] = result

    with open(args.output, "w") as f:
        json.dump(ablation_data, f, indent=2)
    print(f"\nAblation data saved to {args.output}")


if __name__ == "__main__":
    main()
