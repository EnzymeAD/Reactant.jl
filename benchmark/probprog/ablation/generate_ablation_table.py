#!/usr/bin/env python3
"""
Generate the ablation study LaTeX table from ablation_data.json.

Usage:
  python generate_ablation_table.py [ablation_data.json]

Output:
  ../../../paper/tables/ablation.tex
"""
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_TABLES_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "paper", "tables")

# Ordered model list matching the paper table
ABLATION_ORDER = [
    # Standard benchmarks (no SICM structure)
    "logistic_regression",
    "robust_regression",
    "n_schools",
    # SICM-eligible models
    "gp_pois_regr_scaled",
    "gp_fixed_lengthscale",
    "linear_mixed_model",
    "stochastic_volatility",
]

# Display names for the table
MODEL_DISPLAY = {
    "logistic_regression": ("Logistic Regr.", r"$n{=}10\text{k}, k{=}10$"),
    "robust_regression": ("Robust Regr.", r"$n{=}10\text{k}, k{=}10$"),
    "n_schools": ("N-Schools", r"$n{=}10\text{k}$"),
    "gp_pois_regr_scaled": ("GP Pois.\\ (scaled)", r"$n{=}30$"),
    "gp_fixed_lengthscale": ("GP Fixed LS", r"$n{=}60$"),
    "linear_mixed_model": ("Linear Mixed", r"$n{=}100, q{=}20$"),
    "stochastic_volatility": ("Stoch.\\ Vol.", r"$n{=}20$"),
}

# Column keys (legend names from the JSON configs)
COLUMNS = ["Full", "No SICM", "No opt"]


def format_cell(entry):
    """Format a single ablation cell as 'time_s | neff/s'."""
    if entry is None:
        return r"\PROBLEM{}"

    total_s = entry.get("total_s_mean")
    neff_per_s = entry.get("neff_per_s_median")

    if total_s is None:
        return r"\PROBLEM{}"

    # Format time
    if total_s < 1.0:
        time_str = f"{total_s * 1000:.0f}\\,ms"
    elif total_s < 10.0:
        time_str = f"{total_s:.1f}\\,s"
    else:
        time_str = f"{total_s:.0f}\\,s"

    # Format neff/s
    if neff_per_s is not None and neff_per_s == neff_per_s:  # not NaN
        if neff_per_s >= 1000:
            neff_str = f"{neff_per_s / 1000:.1f}\\text{{k}}"
        elif neff_per_s >= 10:
            neff_str = f"{neff_per_s:.0f}"
        else:
            neff_str = f"{neff_per_s:.1f}"
        return f"${time_str}$\\,$|$\\,${neff_str}$"
    else:
        return f"${time_str}$"


def format_compile_time(entry):
    """Format compile time."""
    if entry is None:
        return "---"
    ct = entry.get("compile_time_s", 0)
    if ct <= 0:
        return "---"
    return f"${ct:.0f}$"


def generate_ablation_table(data):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(
        r"\caption{Ablation study: contribution of each compiler design layer. "
        r"``Full'' includes all optimizations. "
        r"``No SICM'' disables algebraic decomposition and sample-invariant hoisting. "
        r"``No opt'' disables all compiler optimizations. "
        r"Each cell shows total wall-clock time\,$|$\,$n_\mathrm{eff}/s$. "
        r"Compile time is reported separately (one-time AOT cost).}"
    )
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\centering\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{@{}l l ccc c@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Params} & "
        r"Full & No SICM & No opt & "
        r"\textbf{Compile} \\"
    )
    lines.append(r"\midrule")

    standard_models = ["logistic_regression", "robust_regression", "n_schools"]
    sicm_models = [
        "gp_pois_regr_scaled", "gp_fixed_lengthscale",
        "linear_mixed_model", "stochastic_volatility",
    ]

    for model_group in [standard_models, sicm_models]:
        for model_name in model_group:
            display_name, params = MODEL_DISPLAY.get(
                model_name, (model_name, "")
            )
            entry = data.get(model_name, {})
            configs = entry.get("configs", {})

            cells = []
            compile_time = None
            for col in COLUMNS:
                col_data = configs.get(col)
                cells.append(format_cell(col_data))
                # Use Full config compile time
                if col == "Full" and col_data:
                    compile_time = col_data.get("compile_time_s")

            ct_str = f"${compile_time:.0f}$\\,s" if compile_time else r"\PROBLEM{}"

            lines.append(
                f"{display_name} & \\footnotesize {params} &"
            )
            lines.append(
                f"  {cells[0]} & {cells[1]} & {cells[2]} & {ct_str} \\\\"
            )

        # Midrule between groups (not after last)
        if model_group is not sicm_models:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    data_file = (
        sys.argv[1] if len(sys.argv) > 1
        else os.path.join(SCRIPT_DIR, "ablation_data.json")
    )

    if os.path.exists(data_file):
        with open(data_file) as f:
            data = json.load(f)
    else:
        print(f"Warning: {data_file} not found, generating placeholder table")
        data = {}

    table_tex = generate_ablation_table(data)

    os.makedirs(PAPER_TABLES_DIR, exist_ok=True)
    output_path = os.path.join(PAPER_TABLES_DIR, "ablation.tex")
    with open(output_path, "w") as f:
        f.write(table_tex + "\n")
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
