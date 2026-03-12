import json
import logging
import os
import subprocess
import tempfile
from typing import cast, Dict, List, Type

import numpy as np
import xarray as xr

from .base_ppl_impl import BasePPLImplementation
from .base_ppl_inference import BasePPLInference

LOGGER = logging.getLogger(__name__)

BENCHMARK_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


class SubprocessMCMC(BasePPLInference):
    is_adaptive = True
    _compile_time_attr = None
    _is_jit = False

    def __init__(
        self, impl_class: Type[BasePPLImplementation], model_attrs: Dict
    ) -> None:
        self.impl_class = impl_class
        self.impl = self.impl_class(**model_attrs)
        self._proc = None
        self._input_path = None
        self.last_run_time = None
        self.last_compile_time = None

    def compile(self, seed: int, **compile_args) -> None:
        pass

    def _get_server_cmd(
        self, data_path: str, output_path: str,
        num_warmup: int, num_samples: int, seed: int,
        step_size: float, max_tree_depth: int,
        adapt_step_size: bool, adapt_mass_matrix: bool,
        **kwargs,
    ) -> List[str]:
        raise NotImplementedError

    def _extract_data(self, samples: Dict) -> xr.Dataset:
        raise NotImplementedError

    def _serialize_data(self, data: xr.Dataset, init_params) -> Dict:
        result = {}
        for var in data.data_vars:
            result[var] = data[var].values.tolist()
        result["coords"] = {
            c: data.coords[c].values.tolist() for c in data.coords
        }
        result["dims"] = {
            var: list(data[var].dims) for var in data.data_vars
        }
        result["attrs"] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in data.attrs.items()
        }
        result["init_params"] = (
            {k: v.tolist() for k, v in init_params.items()}
            if init_params else None
        )
        return result

    def infer(
        self,
        data: xr.Dataset,
        iterations: int,
        num_warmup: int,
        seed: int,
        init_params=None,
        step_size: float = 0.1,
        max_tree_depth: int = 10,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        profile_dir: str = None,
        **infer_args,
    ) -> xr.Dataset:
        num_samples = iterations - num_warmup

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="pplbench_out_"
        ) as f_out:
            output_path = f_out.name

        try:
            if self._proc is None:
                self._start_server(
                    data, num_warmup, num_samples, seed, init_params,
                    step_size, max_tree_depth, adapt_step_size,
                    adapt_mass_matrix, output_path, infer_args,
                )
            else:
                self._run_trial(seed, init_params, output_path,
                                profile_dir=profile_dir)

            with open(output_path, "r") as f:
                output = json.load(f)

            compile_t = output.get("compile_time", 0)
            run_t = output.get("run_time", 0)
            self.last_compile_time = compile_t
            self.last_run_time = run_t
            if compile_t > 0 and self._compile_time_attr:
                setattr(self, self._compile_time_attr, compile_t)
            LOGGER.info(
                "Subprocess compile=%.2fs, run=%.2fs (total=%.2fs)",
                compile_t, run_t, compile_t + run_t,
            )

            samples = {}
            for key, values in output.items():
                if key in ("compile_time", "run_time"):
                    continue
                arr = np.array(values)
                if arr.ndim == 0:
                    continue
                pad = np.tile(arr[0:1], (num_warmup,) + (1,) * (arr.ndim - 1))
                samples[key] = np.concatenate([pad, arr], axis=0)

            return self._extract_data(samples)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def _start_server(
        self, data, num_warmup, num_samples, seed, init_params,
        step_size, max_tree_depth, adapt_step_size, adapt_mass_matrix,
        output_path, infer_args=None,
    ):
        data_dict = self._serialize_data(data, init_params)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="pplbench_in_"
        ) as f_in:
            json.dump(data_dict, f_in)
            self._input_path = f_in.name

        cmd = self._get_server_cmd(
            data_path=self._input_path,
            output_path=output_path,
            num_warmup=num_warmup,
            num_samples=num_samples,
            seed=seed,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            **(infer_args or {}),
        )

        LOGGER.info("Starting subprocess server: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )

        stdout_lines = self._read_until_marker("###READY###")
        if stdout_lines:
            LOGGER.info("Server stdout:\n%s", "\n".join(stdout_lines))

    def _run_trial(self, seed, init_params, output_path, profile_dir=None):
        req = {"seed": seed, "output": output_path}
        if init_params is not None:
            req["init_params"] = {k: v.tolist() for k, v in init_params.items()}
        if profile_dir is not None:
            req["profile"] = profile_dir

        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()

        stdout_lines = self._read_until_marker("###DONE###")
        if stdout_lines:
            LOGGER.info("Server stdout:\n%s", "\n".join(stdout_lines))

    def _read_until_marker(self, marker):
        lines = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                rc = self._proc.poll()
                stderr = self._proc.stderr.read()
                raise RuntimeError(
                    f"Subprocess exited (code={rc}):\n{stderr}"
                )
            stripped = line.rstrip("\n")
            lines.append(stripped)
            if stripped.strip().startswith(marker):
                break
        return lines

    def _cleanup(self):
        if self._proc is not None:
            try:
                self._proc.stdin.write("EXIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            finally:
                self._proc = None
        if self._input_path and os.path.exists(self._input_path):
            os.unlink(self._input_path)
            self._input_path = None

    def __del__(self):
        self._cleanup()
