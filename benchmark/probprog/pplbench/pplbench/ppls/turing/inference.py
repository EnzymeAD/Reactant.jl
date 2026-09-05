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
            "-t", os.environ.get("JULIA_NUM_THREADS", "auto"),
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
        if kwargs.get("matched_init", False):
            cmd.append("--matched-init")
        # AD backend: forwarddiff (Turing's default; low-dimensional models) or
        # reversediff (compiled tape; Turing's documented recommendation for
        # high-dimensional models). Set per model in the config's infer_args.
        cmd += ["--adtype", str(kwargs.get("adtype", "forwarddiff"))]
        return cmd

    def _extract_data(self, samples):
        return self.impl.extract_data_from_turing(samples)


class SVI(SubprocessMCMC):
    """Turing ADVI driver mirroring the Impulse / NumPyro SVI mode.

    Non-adaptive: `iterations` (config) is the number of posterior draws K taken
    from the fitted guide; `num_steps` is the ADVI optimization budget; both are
    threaded through to turing_harness.jl (--method svi), which runs a mean-field
    Gaussian ADVI (AdvancedVI KLMinRepGradDescent) with Adam(step_size)."""

    is_adaptive = False
    _compile_time_attr = "turing_compile_time"
    _is_jit = True

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
            "-t", os.environ.get("JULIA_NUM_THREADS", "auto"),
            f"--project={os.path.join(BENCHMARK_DIR, 'turing_env')}",
            os.path.join(BENCHMARK_DIR, "turing_harness.jl"),
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
        return cmd

    def _extract_data(self, samples):
        return self.impl.extract_data_from_turing(samples)
