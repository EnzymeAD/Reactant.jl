import os
import sys
from typing import Dict, List, Type

from ..base_ppl_impl import BasePPLImplementation
from ..subprocess_inference import SubprocessMCMC, BENCHMARK_DIR
from .base_numpyro_impl import BaseNumPyroImplementation


class MCMC(SubprocessMCMC):
    _compile_time_attr = "numpyro_jit_overhead"
    _is_jit = True

    def __init__(
        self, impl_class: Type[BasePPLImplementation], model_attrs: Dict
    ) -> None:
        super().__init__(impl_class, model_attrs)
        self.numpyro_jit_overhead = 0.0

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
            os.path.join(BENCHMARK_DIR, "numpyro_harness.py"),
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
        if kwargs.get("num_chains"):
            cmd += ["--num-chains", str(kwargs["num_chains"])]
        if kwargs.get("chain_method"):
            cmd += ["--chain-method", str(kwargs["chain_method"])]
        if kwargs.get("matched_init", False):
            cmd.append("--matched-init")
        return cmd

    def _extract_data(self, samples):
        return self.impl.extract_data_from_numpyro(samples)


class SVI(SubprocessMCMC):
    """NumPyro SVI driver mirroring the Impulse SVI mode.

    Non-adaptive: `iterations` is the number of posterior draws K from the fitted
    guide (sample_posterior / Predictive over the guide); `num_steps` is the
    optimization budget; `step_size` is the learning rate. Guide/optimizer/
    num_particles/init_scale are matched to the Impulse arm via infer_args."""

    is_adaptive = False
    _compile_time_attr = "numpyro_jit_overhead"
    _is_jit = True

    def __init__(
        self, impl_class: Type[BasePPLImplementation], model_attrs: Dict
    ) -> None:
        super().__init__(impl_class, model_attrs)
        self.numpyro_jit_overhead = 0.0

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
            os.path.join(BENCHMARK_DIR, "numpyro_harness.py"),
            "--model-class", f"{module_name}.{class_name}",
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
        # Validate correctness mode: switch the SVI fit to the rbg key stream that
        # matches the Impulse arm's reparam noise (numpyro_harness.svi_main).
        if kwargs.get("matched_init", False):
            cmd.append("--matched-init")
        return cmd

    def _extract_data(self, samples):
        return self.impl.extract_data_from_numpyro(samples)
