import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict

os.environ["JAX_ENABLE_X64"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, HMC, NUTS


@dataclass
class BenchmarkResult:
    name: str
    framework: str
    algorithm: str
    compile_time_s: float
    run_time_s: float
    param_a_final: float
    param_b_final: float


def seed_to_rbg_key(seed_u64: list[int]):
    seed_np = np.array(seed_u64, dtype=np.uint64)
    seed_u32 = seed_np.view(np.uint32)  # uint32[4]
    return jax.random.wrap_key_data(seed_u32, impl="rbg")


def model_normal_normal(xs, ys_a_obs=None, ys_b_obs=None):
    prior_std = 5.0
    likelihood_std = 0.5

    param_a = numpyro.sample("param_a", dist.Normal(0.0, prior_std))
    param_b = numpyro.sample("param_b", dist.Normal(0.0, prior_std))

    with numpyro.plate("obs_a", 5):
        numpyro.sample(
            "ys_a",
            dist.Normal(param_a + xs[:5], likelihood_std),
            obs=ys_a_obs
        )

    with numpyro.plate("obs_b", 5):
        numpyro.sample(
            "ys_b",
            dist.Normal(param_b + xs[5:], likelihood_std),
            obs=ys_b_obs
        )


def benchmark_hmc(step_size: float = 0.01, num_steps: int = 3) -> BenchmarkResult:
    xs = jnp.array([-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    ys_a = jnp.array([-2.3, -1.6, -0.4, 0.6, 1.4])
    ys_b = jnp.array([-2.6, -1.4, -0.6, 0.4, 1.6])

    key = seed_to_rbg_key([1, 5])

    init_params = {"param_a": jnp.array(0.0), "param_b": jnp.array(0.0)}

    kernel = HMC(
        model_normal_normal,
        step_size=step_size,
        trajectory_length=step_size * num_steps,
        adapt_step_size=False,
        adapt_mass_matrix=False,
        dense_mass=False,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=0,
        num_samples=1,
        progress_bar=False,
    )

    compile_start = time.perf_counter()
    mcmc.run(
        key,
        xs,
        ys_a_obs=ys_a,
        ys_b_obs=ys_b,
        init_params=init_params,
    )
    samples = mcmc.get_samples()
    jax.block_until_ready(samples)
    compile_time = time.perf_counter() - compile_start

    run_start = time.perf_counter()
    mcmc.run(
        key,
        xs,
        ys_a_obs=ys_a,
        ys_b_obs=ys_b,
        init_params=init_params,
    )
    samples = mcmc.get_samples()
    jax.block_until_ready(samples)
    run_time = time.perf_counter() - run_start

    return BenchmarkResult(
        name="NormalNormal/HMC",
        framework="NumPyro",
        algorithm="HMC",
        compile_time_s=compile_time,
        run_time_s=run_time,
        param_a_final=float(samples["param_a"][-1]),
        param_b_final=float(samples["param_b"][-1]),
    )


def benchmark_nuts(step_size: float = 0.001) -> BenchmarkResult:
    xs = jnp.array([-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    ys_a = jnp.array([-2.3, -1.6, -0.4, 0.6, 1.4])
    ys_b = jnp.array([-2.6, -1.4, -0.6, 0.4, 1.6])

    key = seed_to_rbg_key([1, 5])

    init_params = {"param_a": jnp.array(0.0), "param_b": jnp.array(0.0)}

    kernel = NUTS(
        model_normal_normal,
        step_size=step_size,
        adapt_step_size=False,
        adapt_mass_matrix=False,
        dense_mass=False,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=0,
        num_samples=1,
        progress_bar=False,
    )

    compile_start = time.perf_counter()
    mcmc.run(
        key,
        xs,
        ys_a_obs=ys_a,
        ys_b_obs=ys_b,
        init_params=init_params,
    )
    samples = mcmc.get_samples()
    jax.block_until_ready(samples)
    compile_time = time.perf_counter() - compile_start

    run_start = time.perf_counter()
    mcmc.run(
        key,
        xs,
        ys_a_obs=ys_a,
        ys_b_obs=ys_b,
        init_params=init_params,
    )
    samples = mcmc.get_samples()
    jax.block_until_ready(samples)
    run_time = time.perf_counter() - run_start

    return BenchmarkResult(
        name="NormalNormal/NUTS",
        framework="NumPyro",
        algorithm="NUTS",
        compile_time_s=compile_time,
        run_time_s=run_time,
        param_a_final=float(samples["param_a"][-1]),
        param_b_final=float(samples["param_b"][-1]),
    )


def run_benchmarks(test: str = "all") -> list[BenchmarkResult]:
    results = []

    print("=" * 70)
    print("NumPyro Benchmark (matching Reactant test configuration)")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"NumPyro version: {numpyro.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print("=" * 70)

    if test in ["hmc", "all"]:
        print("\n[HMC] step_size=0.01, num_steps=3")
        result = benchmark_hmc(step_size=0.01, num_steps=3)
        print(f"  compile={result.compile_time_s:.3f}s, run={result.run_time_s:.3f}s")
        print(f"  param_a={result.param_a_final:.6f}, param_b={result.param_b_final:.6f}")
        results.append(result)

    if test in ["nuts", "all"]:
        print("\n[NUTS] step_size=0.001")
        result = benchmark_nuts(step_size=0.001)
        print(f"  compile={result.compile_time_s:.3f}s, run={result.run_time_s:.3f}s")
        print(f"  param_a={result.param_a_final:.6f}, param_b={result.param_b_final:.6f}")
        results.append(result)

    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="NumPyro Benchmark")
    parser.add_argument("--test", type=str, choices=["hmc", "nuts", "all"], default="all")
    parser.add_argument("--output", type=str, default="numpyro_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results = run_benchmarks(args.test)

    output_data = {
        "framework": "NumPyro",
        "jax_version": jax.__version__,
        "numpyro_version": numpyro.__version__,
        "backend": jax.default_backend(),
        "results": [asdict(r) for r in results],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
