#!/usr/bin/env python3
"""
Generate LaTeX tables for the Impulse paper from paper_data.json.

Usage:
  python generate_tables.py [paper_data.json]

Output:
  ../../../paper/tables/correctness.tex   - Numerical agreement (Impulse vs NumPyro)
  ../../../paper/tables/baselines.tex     - Standard model performance comparison
  ../../../paper/tables/sicm.tex          - SICM optimization speedup table
"""
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_TABLES_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "paper", "tables")

# ─── Display names and parameters for all models ───

MODEL_DISPLAY_NAMES = {
    # Standard
    "standard/logistic_regression": "Logistic Regr.",
    "standard/gaussian_process": "Gaussian Proc.",
    "standard/robust_regression": "Robust Regr.",
    "standard/n_schools": "N-Schools",
    # Motivating
    "motivating/gp_pois_regr": "GP Pois.\\ Regr.",
    # SICM
    "sicm/hierarchical_mvn": "Hierarchical MVN",
    "sicm/scale_family_mvn": "Scale-Family MVN",
    "sicm/gp_pois_regr_scaled": "GP Pois.\\ Regr.\\ (scaled)",
    "sicm/gp_classify": "GP Classify",
    "sicm/gp_fixed_lengthscale": "GP Fixed Lengthscale",
    "sicm/phylogenetic_regression": "Phylogenetic Regr.",
    "sicm/linear_mixed_model": "Linear Mixed Model",
    "sicm/stochastic_volatility": "Stochastic Volatility",
    "sicm/car_spatial": "CAR Spatial",
}

MODEL_PARAMS = {
    "standard/logistic_regression": "$n{=}10000, k{=}10$",
    "standard/gaussian_process": "$n{=}60$",
    "standard/robust_regression": "$n{=}10000, k{=}10$",
    "standard/n_schools": "$n{=}10000$",
    "motivating/gp_pois_regr": "$n{=}11$",
    "sicm/hierarchical_mvn": "$n{=}60, K{=}20, J{=}10$",
    "sicm/scale_family_mvn": "$n{=}60$",
    "sicm/gp_pois_regr_scaled": "$n{=}30$",
    "sicm/gp_classify": "$n{=}30$",
    "sicm/gp_fixed_lengthscale": "$n{=}60$",
    "sicm/phylogenetic_regression": "$n{=}50, k{=}3$",
    "sicm/linear_mixed_model": "$n{=}100, k{=}3, q{=}20$",
    "sicm/stochastic_volatility": "$n{=}20$",
    "sicm/car_spatial": "$n{=}50$",
}

FRAMEWORK_ORDER = ["NumPyro", "Impulse", "Impulse (no opt)", "Turing.jl", "Stan", "PyMC"]

# Config ordering for tables
BASELINES_ORDER = [
    "standard/logistic_regression",
    "standard/robust_regression",
    "standard/gaussian_process",
    "standard/n_schools",
    "motivating/gp_pois_regr",
]

SICM_ORDER = [
    "sicm/hierarchical_mvn",
    "sicm/scale_family_mvn",
    "sicm/gp_pois_regr_scaled",
    "sicm/gp_classify",
    "sicm/gp_fixed_lengthscale",
    "sicm/phylogenetic_regression",
    "sicm/linear_mixed_model",
    "sicm/stochastic_volatility",
    "sicm/car_spatial",
]

# Map old model_class keys to config names (backwards compat with old paper_data.json)
MODEL_CLASS_TO_CONFIG = {
    "logistic_regression.LogisticRegression": "standard/logistic_regression",
    "gaussian_process.GaussianProcess": "standard/gaussian_process",
    "robust_regression.RobustRegression": "standard/robust_regression",
    "n_schools.NSchools": "standard/n_schools",
    "gp_pois_regr.GPPoisRegr": "motivating/gp_pois_regr",
    "hierarchical_mvn.HierarchicalMVN": "sicm/hierarchical_mvn",
    "scale_family_mvn.ScaleFamilyMVN": "sicm/scale_family_mvn",
    "gp_pois_regr_scaled.GPPoisRegrScaled": "sicm/gp_pois_regr_scaled",
    "gp_classify.GPClassify": "sicm/gp_classify",
    "gp_fixed_lengthscale.GPFixedLengthscale": "sicm/gp_fixed_lengthscale",
    "phylogenetic_regression.PhylogeneticRegression": "sicm/phylogenetic_regression",
    "linear_mixed_model.LinearMixedModel": "sicm/linear_mixed_model",
    "stochastic_volatility.StochasticVolatility": "sicm/stochastic_volatility",
    "car_spatial.CARSpatial": "sicm/car_spatial",
}


# ─── Formatting helpers ───

def format_diff(val):
    if val == 0.0:
        return "$0.0$"
    exp = f"{val:.1e}"
    mantissa, power = exp.split("e")
    power = int(power)
    return f"${mantissa} \\times 10^{{{power}}}$"


def format_time(val, unit="s"):
    if val < 0.01:
        return f"${val*1000:.1f}$\\,ms"
    if unit == "ms":
        return f"${val:.2f}$"
    return f"${val:.1f}$"


def format_time_with_std(mean, std, unit="s"):
    if unit == "ms":
        return f"${mean:.2f} \\pm {std:.2f}$"
    return f"${mean:.1f} \\pm {std:.1f}$"


def fw_display_name(fw_name):
    if fw_name == "Impulse":
        return r"\tool{}"
    elif fw_name == "Impulse (no opt)":
        return r"\tool{} (no opt)"
    return fw_name


def format_speedup(val):
    if val >= 10:
        return f"${val:.0f}\\times$"
    return f"${val:.1f}\\times$"


# ─── Normalize baselines data ───

def normalize_baselines(data):
    """Convert baselines keyed by model_class (old format) to config_name keys."""
    baselines = data.get("baselines", {})
    normalized = {}
    for key, entry in baselines.items():
        # If key is already a config path (new format), use as-is
        if "/" in key and not "." in key.split("/")[-1].split(".")[0]:
            normalized[key] = entry
        else:
            # Old format: model_class key
            config_name = MODEL_CLASS_TO_CONFIG.get(key, key)
            normalized[config_name] = entry
    return normalized


# ─── Table 1: Correctness ───

def generate_correctness_table(data):
    correctness = data.get("correctness", {})
    if not correctness:
        print("Warning: no correctness data found")
        return ""

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(
        r"\caption{Numerical agreement between \tool{} and NumPyro. "
        r"Both systems run NUTS with identical inputs (same seed, initial parameters, "
        r"step size) and all adaptation disabled. Differences are either exactly zero "
        r"or at machine epsilon (${\sim}10^{-14}$) from floating-point non-associativity "
        r"in matrix operations.}"
    )
    lines.append(r"\label{tab:correctness}")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{@{}l l r@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Variable} & \textbf{Max $|\Delta|$} \\")
    lines.append(r"\midrule")

    config_order = [c for c in BASELINES_ORDER if c in correctness and correctness[c]]

    for i, config_name in enumerate(config_order):
        diffs = correctness[config_name]
        display_name = MODEL_DISPLAY_NAMES.get(config_name, config_name)
        var_names = sorted(diffs.keys())
        n_vars = len(var_names)

        for j, var in enumerate(var_names):
            var_escaped = var.replace("_", r"\_")
            diff_str = format_diff(diffs[var])
            if j == 0:
                lines.append(
                    f"\\multirow{{{n_vars}}}{{*}}{{{display_name}}}"
                )
            lines.append(f"  & \\texttt{{{var_escaped}}} & {diff_str} \\\\")

        if i < len(config_order) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ─── Table 2: Standard Baselines ───

def generate_baselines_table(data):
    baselines = normalize_baselines(data)
    if not baselines:
        return _generate_placeholder_baselines()

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(
        r"\caption{Per-iteration NUTS performance (ms) on standard PPLBench models. "
        r"All frameworks use identical adaptation settings (5000 warmup, 10000 sampling iterations, dense mass matrix). "
        r"\tool{} compilation takes 18--25\,s (one-time AOT cost, not included in per-iteration times). "
        r"\textbf{Bold}: fastest per-iteration time for each model.}"
    )
    lines.append(r"\label{tab:pplbench-baselines}")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{@{}l l r r r r r r@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Params} & "
        r"\textbf{\tool{}} & \textbf{NumPyro} & \textbf{Stan} & "
        r"\textbf{Turing.jl} & \textbf{PyMC} & "
        r"\textbf{vs.\ NumPyro} \\"
    )
    lines.append(
        r" & & \multicolumn{5}{c}{\footnotesize (ms/iter)} & \\"
    )
    lines.append(r"\midrule")

    active_configs = [(c, baselines[c]) for c in BASELINES_ORDER if c in baselines]

    for idx, (config_name, entry) in enumerate(active_configs):
        display_name = MODEL_DISPLAY_NAMES.get(config_name, config_name)
        params = MODEL_PARAMS.get(config_name, "")
        frameworks = entry["frameworks"]

        impulse = frameworks.get("Impulse", {})
        numpyro = frameworks.get("NumPyro", {})
        stan = frameworks.get("Stan", {})
        turing = frameworks.get("Turing.jl", {})
        pymc = frameworks.get("PyMC", {})

        impulse_ms = impulse.get("per_iter_ms_mean", float("nan"))
        numpyro_ms = numpyro.get("per_iter_ms_mean", float("nan"))
        stan_ms = stan.get("per_iter_ms_mean", float("nan"))
        turing_ms = turing.get("per_iter_ms_mean", float("nan"))
        pymc_ms = pymc.get("per_iter_ms_mean", float("nan"))

        # Find minimum for bolding
        all_ms = {}
        if impulse: all_ms["impulse"] = impulse_ms
        if numpyro: all_ms["numpyro"] = numpyro_ms
        if stan: all_ms["stan"] = stan_ms
        if turing: all_ms["turing"] = turing_ms
        if pymc: all_ms["pymc"] = pymc_ms
        min_ms = min(all_ms.values()) if all_ms else float("nan")

        def fmt_ms(val, key):
            if not val or val != val:  # nan check
                return "---"
            s = f"{val:.2f}" if val >= 0.1 else f"{val:.2f}"
            if key in all_ms and abs(all_ms[key] - min_ms) < 1e-6:
                return f"$\\mathbf{{{s}}}$"
            return f"${s}$"

        impulse_str = fmt_ms(impulse_ms, "impulse") if impulse else "---"
        numpyro_str = fmt_ms(numpyro_ms, "numpyro") if numpyro else "---"
        stan_str = fmt_ms(stan_ms, "stan") if stan else "---"
        turing_str = fmt_ms(turing_ms, "turing") if turing else "---"
        pymc_str = fmt_ms(pymc_ms, "pymc") if pymc else "---"

        # vs NumPyro: numpyro / impulse
        if impulse and numpyro and impulse_ms > 0:
            vs_numpyro = numpyro_ms / impulse_ms
            vs_numpyro_str = format_speedup(vs_numpyro)
        else:
            vs_numpyro_str = "---"

        lines.append(
            f"{display_name} & {params} & "
            f"{impulse_str} & {numpyro_str} & {stan_str} & "
            f"{turing_str} & {pymc_str} & "
            f"{vs_numpyro_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ─── Table 3: SICM Optimization ───

def generate_sicm_table(data):
    sicm_data = data.get("sicm", {})
    # Also check baselines for SICM entries (backwards compat)
    baselines = normalize_baselines(data)
    for key in list(baselines.keys()):
        if key.startswith("sicm/") and key not in sicm_data:
            sicm_data[key] = baselines[key]

    if not sicm_data:
        print("Warning: no SICM data found, skipping sicm.tex")
        return ""

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(
        r"\caption{Per-iteration NUTS performance (ms) on models with sample-invariant structure. "
        r"\tool{} applies SICM automatically; other frameworks execute the model as written. "
        r"All frameworks use identical adaptation settings (500 warmup, 1000 sampling iterations, dense mass matrix). "
        r"\tool{} compilation takes 25--48\,s (one-time AOT cost, not included in per-iteration times). "
        r"\textbf{Bold}: fastest per-iteration time for each model.}"
    )
    lines.append(r"\label{tab:sicm}")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{@{}l l r r r r r r@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Params} & "
        r"\textbf{\tool{}} & \textbf{NumPyro} & \textbf{Stan} & "
        r"\textbf{Turing.jl} & \textbf{PyMC} & "
        r"\textbf{vs.\ NumPyro} \\"
    )
    lines.append(
        r" & & \multicolumn{5}{c}{\footnotesize (ms/iter)} & \\"
    )
    lines.append(r"\midrule")

    active_configs = [(c, sicm_data[c]) for c in SICM_ORDER if c in sicm_data]

    for idx, (config_name, entry) in enumerate(active_configs):
        display_name = MODEL_DISPLAY_NAMES.get(config_name, config_name)
        params = MODEL_PARAMS.get(config_name, "")
        frameworks = entry["frameworks"]

        impulse = frameworks.get("Impulse", {})
        numpyro = frameworks.get("NumPyro", {})
        stan = frameworks.get("Stan", {})
        turing = frameworks.get("Turing.jl", {})
        pymc = frameworks.get("PyMC", {})

        impulse_ms = impulse.get("per_iter_ms_mean", float("nan"))
        numpyro_ms = numpyro.get("per_iter_ms_mean", float("nan"))
        stan_ms = stan.get("per_iter_ms_mean", float("nan"))
        turing_ms = turing.get("per_iter_ms_mean", float("nan"))
        pymc_ms = pymc.get("per_iter_ms_mean", float("nan"))

        # Find minimum for bolding
        all_ms = {}
        if impulse: all_ms["impulse"] = impulse_ms
        if numpyro: all_ms["numpyro"] = numpyro_ms
        if stan: all_ms["stan"] = stan_ms
        if turing: all_ms["turing"] = turing_ms
        if pymc: all_ms["pymc"] = pymc_ms
        min_ms = min(all_ms.values()) if all_ms else float("nan")

        def fmt_ms(val, key):
            if not val or val != val:  # nan check
                return "---"
            s = f"{val:.2f}" if val >= 0.1 else f"{val:.2f}"
            if key in all_ms and abs(all_ms[key] - min_ms) < 1e-6:
                return f"$\\mathbf{{{s}}}$"
            return f"${s}$"

        impulse_str = fmt_ms(impulse_ms, "impulse") if impulse else "---"
        numpyro_str = fmt_ms(numpyro_ms, "numpyro") if numpyro else "---"
        stan_str = fmt_ms(stan_ms, "stan") if stan else "---"
        turing_str = fmt_ms(turing_ms, "turing") if turing else "---"
        pymc_str = fmt_ms(pymc_ms, "pymc") if pymc else "---"

        # vs NumPyro: numpyro / impulse
        if impulse and numpyro and impulse_ms > 0:
            vs_numpyro = numpyro_ms / impulse_ms
            vs_numpyro_str = format_speedup(vs_numpyro)
        else:
            vs_numpyro_str = "---"

        lines.append(
            f"{display_name} & {params} & "
            f"{impulse_str} & {numpyro_str} & {stan_str} & "
            f"{turing_str} & {pymc_str} & "
            f"{vs_numpyro_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ─── Placeholder ───

def _generate_placeholder_baselines():
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(
        r"\caption{NUTS performance comparison on standard PPLBench models. (Placeholder)}"
    )
    lines.append(r"\label{tab:pplbench-baselines}")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{@{}l l r r r r r@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Framework} & "
        r"\textbf{Compile (s)} & \textbf{Per-iter (ms)} & \textbf{Total (s)} & "
        r"\textbf{$n_\mathrm{eff}$} & \textbf{$n_\mathrm{eff}/s$} \\"
    )
    lines.append(r"\midrule")
    for config_name in BASELINES_ORDER:
        display_name = MODEL_DISPLAY_NAMES.get(config_name, config_name)
        params = MODEL_PARAMS.get(config_name, "")
        lines.append(
            f"\\multirow{{5}}{{*}}{{\\shortstack[l]{{{display_name}\\\\({params})}}}}"
        )
        for fw in FRAMEWORK_ORDER:
            lines.append(f"  & {fw_display_name(fw)} & --- & --- & --- & --- & --- \\\\")
        lines.append(r"\midrule")
    # Remove last midrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ─── Main ───

def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "paper_data.json")
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Run collect_paper_data.py first.")
        print("Generating placeholder tables instead...")
        data = {"correctness": {}, "baselines": {}, "sicm": {}}
    else:
        with open(data_file) as f:
            data = json.load(f)

    os.makedirs(PAPER_TABLES_DIR, exist_ok=True)

    # Correctness table
    correctness_tex = generate_correctness_table(data)
    if correctness_tex:
        correctness_path = os.path.join(PAPER_TABLES_DIR, "correctness.tex")
        with open(correctness_path, "w") as f:
            f.write(correctness_tex + "\n")
        print(f"Written: {correctness_path}")

    # Baselines table (standard + motivating)
    baselines_tex = generate_baselines_table(data)
    baselines_path = os.path.join(PAPER_TABLES_DIR, "baselines.tex")
    with open(baselines_path, "w") as f:
        f.write(baselines_tex + "\n")
    print(f"Written: {baselines_path}")

    # SICM table
    sicm_tex = generate_sicm_table(data)
    if sicm_tex:
        sicm_path = os.path.join(PAPER_TABLES_DIR, "sicm.tex")
        with open(sicm_path, "w") as f:
            f.write(sicm_tex + "\n")
        print(f"Written: {sicm_path}")


if __name__ == "__main__":
    main()
