import argparse
import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class ComparisonResult:
    algorithm: str
    numpyro_time_s: float
    reactant_time_s: float
    speedup: float
    correctness_passed: bool
    param_diff: float
    notes: str


def load_results(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def compare_params(
    numpyro_result: dict,
    reactant_result: dict,
    rtol: float = 0.1,
    atol: float = 0.5,
) -> tuple[bool, float, str]:
    notes = []
    max_diff = 0.0

    np_a = numpyro_result.get("param_a_final", 0.0)
    rx_a = reactant_result.get("param_a_final", 0.0)
    diff_a = abs(np_a - rx_a)
    max_diff = max(max_diff, diff_a)

    if not np.isclose(np_a, rx_a, rtol=rtol, atol=atol):
        notes.append(f"param_a: NumPyro={np_a:.6f}, Reactant={rx_a:.6f}")

    np_b = numpyro_result.get("param_b_final", 0.0)
    rx_b = reactant_result.get("param_b_final", 0.0)
    diff_b = abs(np_b - rx_b)
    max_diff = max(max_diff, diff_b)

    if not np.isclose(np_b, rx_b, rtol=rtol, atol=atol):
        notes.append(f"param_b: NumPyro={np_b:.6f}, Reactant={rx_b:.6f}")

    passed = len(notes) == 0
    notes_str = "; ".join(notes) if notes else "OK"

    return passed, max_diff, notes_str


def compare_results(
    numpyro_data: dict,
    reactant_data: dict,
    rtol: float = 0.1,
    atol: float = 0.5,
) -> list[ComparisonResult]:
    comparisons = []

    np_results = {r["algorithm"]: r for r in numpyro_data.get("results", [])}
    rx_results = {r["algorithm"]: r for r in reactant_data.get("results", [])}

    all_algorithms = set(np_results.keys()) | set(rx_results.keys())

    for algorithm in sorted(all_algorithms):
        if algorithm not in np_results:
            print(f"Warning: {algorithm} missing from NumPyro results")
            continue
        if algorithm not in rx_results:
            print(f"Warning: {algorithm} missing from Reactant results")
            continue

        np_result = np_results[algorithm]
        rx_result = rx_results[algorithm]

        np_time = np_result["run_time_s"]
        rx_time = rx_result["run_time_s"]
        speedup = np_time / rx_time if rx_time > 0 else float("inf")

        passed, max_diff, notes = compare_params(np_result, rx_result, rtol, atol)

        comparisons.append(
            ComparisonResult(
                algorithm=algorithm,
                numpyro_time_s=np_time,
                reactant_time_s=rx_time,
                speedup=speedup,
                correctness_passed=passed,
                param_diff=max_diff,
                notes=notes,
            )
        )

    return comparisons


def print_comparison_table(comparisons: list[ComparisonResult]):
    print("\n| Algorithm | NumPyro(s) | Reactant(s) | Speedup | Correctness |")
    print("|-----------|-----------|-------------|---------|---------|")

    for c in comparisons:
        correct_str = "✓" if c.correctness_passed else "✗"
        speedup_str = f"{c.speedup:.2f}x"
        print(f"| {c.algorithm} | {c.numpyro_time_s:.4f} | {c.reactant_time_s:.4f} | {speedup_str} | {correct_str} |")

    print()


def generate_benchmark_json(comparisons: list[ComparisonResult]) -> list[dict]:
    benchmarks = []

    for c in comparisons:
        # Reactant run time
        benchmarks.append(
            {
                "name": f"Reactant {c.algorithm} run_time",
                "unit": "seconds",
                "value": c.reactant_time_s,
            }
        )
        # NumPyro run time (for reference)
        benchmarks.append(
            {
                "name": f"NumPyro {c.algorithm} run_time",
                "unit": "seconds",
                "value": c.numpyro_time_s,
            }
        )

    return benchmarks


def main():
    parser = argparse.ArgumentParser(description="Compare NumPyro and Reactant benchmark results")
    parser.add_argument("--numpyro-results", type=str, required=True)
    parser.add_argument("--reactant-results", type=str, required=True)
    parser.add_argument("--output", type=str, default="comparison.json")
    parser.add_argument(
        "--benchmark-output", type=str, default=None, help="Output file for github-action-benchmark format"
    )
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--atol", type=float, default=0.5)
    args = parser.parse_args()

    print("Loading NumPyro results...")
    numpyro_data = load_results(args.numpyro_results)
    print(f"  Found {len(numpyro_data.get('results', []))} benchmark(s)")

    print("Loading Reactant results...")
    reactant_data = load_results(args.reactant_results)
    print(f"  Found {len(reactant_data.get('results', []))} benchmark(s)")

    comparisons = compare_results(numpyro_data, reactant_data, args.rtol, args.atol)

    print_comparison_table(comparisons)

    comparison_output = {
        "numpyro_metadata": {k: v for k, v in numpyro_data.items() if k != "results"},
        "reactant_metadata": {k: v for k, v in reactant_data.items() if k != "results"},
        "comparisons": [
            {
                "algorithm": c.algorithm,
                "numpyro_time_s": c.numpyro_time_s,
                "reactant_time_s": c.reactant_time_s,
                "speedup": c.speedup,
                "correctness_passed": c.correctness_passed,
                "param_diff": c.param_diff,
                "notes": c.notes,
            }
            for c in comparisons
        ],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(comparison_output, f, indent=2)
    print(f"\nComparison saved to {args.output}")

    if args.benchmark_output:
        benchmark_json = generate_benchmark_json(comparisons)
        benchmark_dir = os.path.dirname(args.benchmark_output)
        if benchmark_dir:
            os.makedirs(benchmark_dir, exist_ok=True)
        with open(args.benchmark_output, "w") as f:
            json.dump(benchmark_json, f, indent=2)
        print(f"Benchmark JSON saved to {args.benchmark_output}")

    failed = [c for c in comparisons if not c.correctness_passed]
    if failed:
        print(f"\nWARNING: {len(failed)} correctness check(s) failed!")
        for c in failed:
            print(f"  - {c.algorithm}: {c.notes}")

    print("\nComparison complete!")


if __name__ == "__main__":
    main()
