import os
from typing import Dict, List, Type

from ..base_ppl_impl import BasePPLImplementation
from ..subprocess_inference import SubprocessMCMC, BENCHMARK_DIR
from .base_turing_impl import BaseTuringImplementation


class MCMC(SubprocessMCMC):
    _compile_time_attr = "turing_compile_time"
    _is_jit = True  # JIT warmup separates most compilation, but keep as safety net

    def __init__(
        self, impl_class: Type[BasePPLImplementation], model_attrs: Dict
    ) -> None:
        super().__init__(impl_class, model_attrs)
        self.turing_compile_time = 0.0

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
            f"--project={os.path.join(BENCHMARK_DIR, 'turing_env')}",
            os.path.join(BENCHMARK_DIR, "turing_harness.jl"),
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
        return cmd

    def _extract_data(self, samples):
        return self.impl.extract_data_from_turing(samples)
