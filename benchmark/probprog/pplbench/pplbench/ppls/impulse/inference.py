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
        if kwargs.get("disable_opt", False):
            cmd.append("--disable-opt")
        return cmd

    def _extract_data(self, samples):
        return self.impl.extract_data_from_impulse(samples)
