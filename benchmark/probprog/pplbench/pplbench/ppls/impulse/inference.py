import os
import sys
from typing import Dict, List, Type

from ..base_ppl_impl import BasePPLImplementation
from ..subprocess_inference import SubprocessMCMC, BENCHMARK_DIR
from .base_impulse_impl import BaseImpulseImplementation


class MCMC(SubprocessMCMC):
    _compile_time_attr = "impulse_compile_time"

    def __init__(
        self, impl_class: Type[BasePPLImplementation], model_attrs: Dict
    ) -> None:
        super().__init__(impl_class, model_attrs)
        self.impulse_compile_time = 0.0

    def _get_server_cmd(
        self, data_path: str, output_path: str,
        num_warmup: int, num_samples: int, seed: int,
        step_size: float, max_tree_depth: int,
        adapt_step_size: bool, adapt_mass_matrix: bool,
        **kwargs,
    ) -> List[str]:
        cmd = [
            "julia",
            "-t", os.environ.get("JULIA_NUM_THREADS", "32"),
            f"--project={BENCHMARK_DIR}",
            os.path.join(BENCHMARK_DIR, "harness.jl"),
            "--model", self.impl.julia_model_path,
            "--data", data_path,
            "--output", output_path,
            "--num-warmup", str(num_warmup),
            "--num-samples", str(num_samples),
            "--seed", str(seed),
            "--step-size", str(step_size),
            "--max-tree-depth", str(max_tree_depth),
            "--server",
        ]
        if adapt_step_size:
            cmd.append("--adapt-step-size")
        if adapt_mass_matrix:
            cmd.append("--adapt-mass-matrix")
        if kwargs.get("num_chains"):
            cmd += ["--num-chains", str(kwargs["num_chains"])]
        if kwargs.get("chain_method"):
            cmd += ["--chain-method", str(kwargs["chain_method"])]
        if kwargs.get("disable_opt", False):
            cmd.append("--disable-opt")
        if kwargs.get("disable_sicm", False):
            cmd.append("--disable-sicm")
        if kwargs.get("disable_region", False):
            cmd.append("--disable-region")
        if kwargs.get("no_opt", False):
            cmd.append("--no-opt")
        if kwargs.get("post_sicm", False):
            cmd.append("--post-sicm")
        return cmd

    def _extract_data(self, samples):
        return self.impl.extract_data_from_impulse(samples)


class SVI(SubprocessMCMC):
    """Stochastic Variational Inference driver.

    Non-adaptive: `iterations` (config) is the number of posterior draws K taken
    from the fitted guide (mirroring NumPyro Predictive over the guide); the SVI
    optimization budget is `num_steps` from infer_args. `step_size` is the guide
    learning rate. SICM on/off is controlled by the usual `disable_opt` /
    `disable_sicm` infer_args, reused from the MCMC path."""

    is_adaptive = False
    _compile_time_attr = "impulse_compile_time"

    def __init__(
        self, impl_class: Type[BasePPLImplementation], model_attrs: Dict
    ) -> None:
        super().__init__(impl_class, model_attrs)
        self.impulse_compile_time = 0.0

    def _get_server_cmd(
        self, data_path: str, output_path: str,
        num_warmup: int, num_samples: int, seed: int,
        step_size: float, max_tree_depth: int,
        adapt_step_size: bool, adapt_mass_matrix: bool,
        **kwargs,
    ) -> List[str]:
        cmd = [
            "julia",
            "-t", os.environ.get("JULIA_NUM_THREADS", "32"),
            f"--project={BENCHMARK_DIR}",
            os.path.join(BENCHMARK_DIR, "harness.jl"),
            "--model", self.impl.julia_model_path,
            "--data", data_path,
            "--output", output_path,
            "--method", "svi",
            "--num-samples", str(num_samples),
            "--seed", str(seed),
            "--step-size", str(step_size),
            "--num-steps", str(kwargs.get("num_steps", 300)),
            "--num-particles", str(kwargs.get("num_particles", 1)),
            "--init-scale", str(kwargs.get("init_scale", 0.1)),
            "--guide", str(kwargs.get("guide", "AutoNormal")),
            "--optimizer", str(kwargs.get("optimizer", "Adam")),
            "--server",
        ]
        if kwargs.get("disable_opt", False):
            cmd.append("--disable-opt")
        if kwargs.get("disable_sicm", False):
            cmd.append("--disable-sicm")
        if kwargs.get("disable_region", False):
            cmd.append("--disable-region")
        if kwargs.get("no_opt", False):
            cmd.append("--no-opt")
        if kwargs.get("post_sicm", False):
            cmd.append("--post-sicm")
        return cmd

    def _extract_data(self, samples):
        return self.impl.extract_data_from_impulse(samples)
