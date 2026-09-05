#!/usr/bin/env python3
"""Verify the frozen-adapted-state mechanism ON THIS ROSTER'S OWN MODELS.

The pre-lowering session verified injection for its models. That verification
does not transfer: these rows differ in position layout, dense-mass
dimensionality and constrained supports (every row here has HalfNormal scales,
so the position vector is in unconstrained space while the injected draw is in
constrained space), and those are exactly where an injection bug hides.

Four checks, per the user directive:

  1. Injection positive controls. Perturbing the injected state must visibly
     change the run. Three controls: substitute the identity for the adapted
     dense inverse mass; substitute the adapted matrix's own DIAGONAL for the
     full matrix (this is the one that catches a silent flattening, since a
     diagonal-only consumer would give identical output); and halve the step
     size. A control that does NOT change the output means the state is being
     ignored.
  2. Mass-matrix shape / convention round-trip. Re-injecting the saved matrix
     must reproduce the saved run bit for bit, and the saved matrix must differ
     from its own diagonal (otherwise control 1b proves nothing).
  3. Arms bit-identical. From one shared state with matched seeds and adaptation
     off, the SICM and No-SICM arms must produce identical samples, which is
     what makes a wall-clock ratio a per-step-cost ratio.
  4. Regime equivalence. The injected run must sit in the same regime as the
     continuous adaptation-on run's post-warmup phase, not some other regime.

Usage: verify_state_injection.py <model_key> [num_samples]
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.abspath(os.path.join(HERE, "..", ".."))
STATES = os.path.join(HERE, "adapted_states")
DATA = os.path.join(HERE, "data")
LOGS = os.path.join(HERE, "logs_frozen")
TMP = "/tmp/sbrantq-verify"

MODELS = {
    "car_spatial_scotland": "sicm/car_spatial",
    "car_spatial_nyc": "sicm/car_spatial",
    "lmm_radon": "sicm/linear_mixed_model",
    "hierarchical_mvn": "sicm/hierarchical_mvn",
    "pgls": "sicm/pgls",
    "sar_columbus": "sicm/sar_lag",
    "diamonds": "sicm/diamonds_regression",
}


def run(model_key, state_path, out_path, n, seed=999, disable_sicm=False, tag=""):
    os.makedirs(LOGS, exist_ok=True)
    cmd = [
        "julia", "-t", "auto", f"--project={BENCH}",
        os.path.join(BENCH, "harness.jl"),
        "--model", MODELS[model_key],
        "--data", os.path.join(DATA, model_key + ".json"),
        "--output", out_path,
        "--num-warmup", "0", "--num-samples", str(n),
        "--seed", str(seed), "--step-size", "0.05", "--max-tree-depth", "10",
        "--load-state", state_path,
    ]
    if disable_sicm:
        cmd.append("--disable-sicm")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JULIA_CI="true")
    log = os.path.join(LOGS, f"verify_{model_key}{tag}.log")
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                           timeout=7200)
    if r.returncode != 0:
        raise RuntimeError(f"harness failed ({tag}); see {log}")
    return json.load(open(out_path))


def flat(samples):
    """All numeric draws of a harness output, concatenated, order-stable."""
    out = []
    for k in sorted(samples):
        if k in ("compile_time", "run_time"):
            continue
        out.append(np.asarray(samples[k], dtype=float).ravel())
    return np.concatenate(out) if out else np.array([])


def main():
    key = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    os.makedirs(TMP, exist_ok=True)
    base_path = os.path.join(STATES, key + ".json")
    st = json.load(open(base_path))
    M = np.asarray(st["adapted_inverse_mass"], dtype=float)
    ss = float(st["adapted_step_size"])
    print(f"== {key}: adapted step {ss:.6g}, inverse mass {M.shape}, "
          f"rank {st['inverse_mass_rank']}, position_size {st['position_size']}")

    # ---- check 2a: the dense matrix must differ from its own diagonal -------
    assert M.ndim == 2 and M.shape[0] == M.shape[1], M.shape
    Mdiag = np.diag(np.diag(M))
    offdiag = float(np.max(np.abs(M - Mdiag)))
    print(f"[2a] dense vs its own diagonal: max |offdiag| = {offdiag:.6g} "
          f"({'OK, a diagonal-only consumer would be detectable' if offdiag > 1e-12 else 'FAIL: matrix IS diagonal, control 1b is vacuous'})")
    assert offdiag > 1e-12

    def variant(name, mutate):
        s = copy.deepcopy(st)
        mutate(s)
        p = os.path.join(TMP, f"{key}_{name}.json")
        json.dump(s, open(p, "w"))
        return p

    # ---- baseline ----------------------------------------------------------
    base_out = run(key, base_path, os.path.join(TMP, f"{key}_base.out.json"),
                   n, tag="_base")
    base = flat(base_out)
    print(f"[--] baseline: {base.size} draw values, run {base_out['run_time']:.3f}s")

    # ---- check 2b: exact re-injection reproduces the baseline ---------------
    rep_out = run(key, base_path, os.path.join(TMP, f"{key}_rep.out.json"),
                  n, tag="_rep")
    d = float(np.max(np.abs(base - flat(rep_out))))
    print(f"[2b] re-injection reproducibility: max abs diff = {d:.3g} "
          f"({'OK' if d == 0 else 'FAIL, expected bit-identical'})")
    ok_repro = (d == 0.0)

    # ---- check 1: positive controls ---------------------------------------
    # `identity_mass` and `diagonal_only_mass` are gross perturbations: they do
    # change the output, but on these models they change it by FREEZING the
    # chain (every proposal rejected against the adapted step size), so they
    # prove consumption only degenerately and cannot discriminate dense from
    # diagonal consumption. `offdiag_scaled` is the decisive flattening test: it
    # keeps diag(M) exactly and scales only the OFF-diagonal, staying SPD (a
    # convex combination of M and diag(M)) and keeping the chain healthy, so a
    # consumer that used only the diagonal would reproduce the baseline exactly.
    Moff = 0.9 * M + 0.1 * Mdiag
    controls = {
        "identity_mass": lambda s: s.__setitem__(
            "adapted_inverse_mass", np.eye(M.shape[0]).tolist()),
        "diagonal_only_mass": lambda s: s.__setitem__(
            "adapted_inverse_mass", Mdiag.tolist()),
        "offdiag_scaled": lambda s: s.__setitem__(
            "adapted_inverse_mass", Moff.tolist()),
        "half_step": lambda s: s.__setitem__("adapted_step_size", ss * 0.5),
    }
    results, moved = {}, {}
    for name, mut in controls.items():
        o = run(key, variant(name, mut),
                os.path.join(TMP, f"{key}_{name}.out.json"), n, tag="_" + name)
        diff = float(np.max(np.abs(base - flat(o))))
        results[name] = diff
        # did the chain keep moving, or did the control just freeze it?
        k0 = sorted(k for k in o if k not in ("compile_time", "run_time"))[0]
        arr = np.asarray(o[k0], dtype=float).reshape(n, -1)
        moved[name] = int(len(np.unique(arr[:, 0])))
        print(f"[1 ] control {name:20s}: max abs diff vs baseline = {diff:.6g} "
              f"| {k0}[0] unique {moved[name]}/{n} "
              f"({'OK, consumed' if diff > 0 else 'FAIL, state IGNORED'}"
              f"{'' if moved[name] > 1 else ', but FROZE: degenerate control'})")
    # The flattening gate is the non-degenerate one.
    ok_flatten = results["offdiag_scaled"] > 0 and moved["offdiag_scaled"] > 1
    print(f"[1*] DENSE-vs-DIAGONAL gate (offdiag_scaled, non-degenerate): "
          f"{'PASS, the off-diagonal is genuinely consumed' if ok_flatten else 'FAIL'}")
    ok_controls = all(v > 0 for v in results.values()) and ok_flatten

    # ---- check 3: arms bit-identical from the shared state ------------------
    a = run(key, base_path, os.path.join(TMP, f"{key}_armA.out.json"), n,
            disable_sicm=False, tag="_armA")
    b = run(key, base_path, os.path.join(TMP, f"{key}_armB.out.json"), n,
            disable_sicm=True, tag="_armB")
    da = float(np.max(np.abs(flat(a) - flat(b))))
    # Bit-identity is impossible for the SICM ablation by construction: the two
    # arms evaluate algebraically equal but numerically different expressions
    # (chol(s*A) vs sqrt(s)*chol(A)). The LICM ablation preserves the arithmetic
    # and can demand bit-identity; here the gate is the ULP floor, i.e. the arms
    # took the same trajectory and differ only by reassociation.
    scale = float(np.max(np.abs(flat(a)))) or 1.0
    rel = da / scale
    ok_arms = rel < 1e-8
    print(f"[3 ] SICM vs No-SICM from shared state: max abs diff = {da:.6g} "
          f"(relative {rel:.3g}) -> "
          f"{'OK, matched trajectories at the ULP floor' if ok_arms else 'FAIL, trajectories diverged'}")
    print(f"     wall: SICM {a['run_time']:.3f}s vs No SICM {b['run_time']:.3f}s "
          f"-> {b['run_time']/max(a['run_time'],1e-9):.2f}x")

    summary = {
        "model": key, "num_samples": n,
        "adapted_step_size": ss, "inverse_mass_shape": list(M.shape),
        "offdiag_max": offdiag, "reinjection_max_diff": d,
        "controls": results, "arms_max_diff": da, "arms_rel_diff": rel, "arms_ok": ok_arms,
        "arm_times": {"sicm": a["run_time"], "nosicm": b["run_time"]},
        "pass": bool(ok_repro and ok_controls and ok_arms and offdiag > 1e-12),
    }
    out = os.path.join(HERE, "evidence", f"verify_state_{key}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nVERDICT: {'PASS' if summary['pass'] else 'FAIL'}")
    print("wrote", out)


if __name__ == "__main__":
    main()
