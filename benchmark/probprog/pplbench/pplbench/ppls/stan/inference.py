import os
import sys
from typing import Dict, List, Type

from ..base_ppl_impl import BasePPLImplementation
from ..subprocess_inference import SubprocessMCMC, BENCHMARK_DIR
from .base_stan_impl import BaseStanImplementation


class MCMC(SubprocessMCMC):
    _compile_time_attr = "stan_compile_time"

    def __init__(
        self, impl_class: Type[BasePPLImplementation], model_attrs: Dict
    ) -> None:
        super().__init__(impl_class, model_attrs)
        self.stan_compile_time = 0.0

    def _get_server_cmd(
        self, data_path: str, output_path: str,
        num_warmup: int, num_samples: int, seed: int,
        step_size: float, max_tree_depth: int,
        adapt_step_size: bool, adapt_mass_matrix: bool,
        **kwargs,
    ) -> List[str]:
        module_name = self.impl.__class__.__module__.split(".")[-1]
        class_name = self.impl.__class__.__name__
        cmd = [
            sys.executable,
            os.path.join(BENCHMARK_DIR, "stan_harness.py"),
            "--model-class", f"{module_name}.{class_name}",
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
        return cmd

    def _extract_data(self, samples):
        return self.impl.extract_data_from_stan(samples)
