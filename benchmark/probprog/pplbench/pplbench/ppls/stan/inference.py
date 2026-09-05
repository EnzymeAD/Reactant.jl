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
        impl_module = self.impl.__class__.__module__
        package, module_name = impl_module.rsplit(".", 1)
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
            "--metric", str(kwargs.get("metric", "dense_e")),
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
        if package != "pplbench.ppls.stan":
            # Variant packages (e.g. pplbench.ppls.stan_ladder.<variant>) reuse
            # this driver via a config-level `package` override; tell the
            # harness subprocess where to re-import the implementation from.
            cmd += ["--package", package]
        return cmd

    def _extract_data(self, samples):
        return self.impl.extract_data_from_stan(samples)


class SVI(SubprocessMCMC):
    """Stan ADVI driver mirroring the Impulse / NumPyro SVI mode.

    Non-adaptive: `iterations` (config) is the number of posterior draws K taken
    from the fitted variational approximation; K is threaded through as
    `output_samples`. Stan ADVI (CmdStan `variational`, algorithm=meanfield) runs
    its OWN adaptive stochastic optimizer and convergence test, so the matched
    step_size / num_steps / optimizer infer_args (used by the other arms) are
    recorded but not applied to Stan's optimizer. `num_particles` maps to
    `grad_samples` (one-particle ELBO gradient)."""

    is_adaptive = False
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
            "--method", "svi",
            "--seed", str(seed),
            "--output-samples", str(num_samples),
            # num_particles is the shared one-particle-ELBO protocol setting;
            # vi_grad_samples is a Stan-only override for ADVI ladder runs.
            "--grad-samples",
            str(kwargs.get("vi_grad_samples") or kwargs.get("num_particles", 1)),
            "--vi-algorithm", "meanfield",
            "--num-steps", str(kwargs.get("num_steps", 2000)),
            "--server",
        ]
        if kwargs.get("vi_iter"):
            cmd += ["--vi-iter", str(kwargs["vi_iter"])]
        # CmdStan ADVI knobs. Only forwarded when the config sets them, so an
        # unset knob keeps CmdStan's own default.
        for key, flag in (
            ("vi_elbo_samples", "--vi-elbo-samples"),
            ("vi_tol_rel_obj", "--vi-tol-rel-obj"),
            ("vi_eval_elbo", "--vi-eval-elbo"),
            ("vi_eta", "--vi-eta"),
            ("vi_adapt_iter", "--vi-adapt-iter"),
        ):
            if kwargs.get(key) is not None:
                cmd += [flag, str(kwargs[key])]
        if kwargs.get("vi_no_adapt", False):
            cmd.append("--vi-no-adapt")
        if kwargs.get("vi_show_console", False):
            cmd.append("--vi-show-console")
        if kwargs.get("require_converged", False):
            cmd.append("--require-converged")
        if kwargs.get("matched_init", False):
            cmd.append("--matched-init")
        return cmd

    def _extract_data(self, samples):
        return self.impl.extract_data_from_stan(samples)
