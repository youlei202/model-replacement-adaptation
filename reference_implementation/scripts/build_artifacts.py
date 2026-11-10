from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernel_isqed.contracts import ADAPTERS, LAMBDA_GRID, MATCHED_BUDGETS
from kernel_isqed.audit import paired_bootstrap_interval
from kernel_isqed.io import (
    CANONICAL_DIR,
    TABLES_DIR,
    read_json,
    sha256_file,
    write_json,
    write_table,
)
from kernel_isqed.response import coordinates_to_probabilities, total_variation
from kernel_isqed.planted import (
    PLANTED_SCOPE_ID,
    PRIMARY_BUDGET,
    PRIMARY_TV_STRENGTH,
)


FIGURE_DIR = PROJECT_ROOT / "figures"
LATEX_DIR = PROJECT_ROOT / "tables"
VIEW_DIR = PROJECT_ROOT / "views"

COLORS = {
    "mean_all": "#666666",
    "fixed_peer": "#666666",
    "fixed_pair_mean": "#666666",
    "convex": "#0072B2",
    "linear": "#009E73",
    "rff": "#E69F00",
    "krr": "#D55E00",
    "mlp": "#CC79A7",
    "val_select_peer": "#333333",
    "val_select_pair_mean": "#333333",
}

DISPLAY_NAMES = {
    "mean_all": "Mean of peers",
    "fixed_peer": "Fixed peer",
    "fixed_pair_mean": "Fixed pair mean",
    "convex": "Convex",
    "linear": "Linear",
    "rff": "RFF",
    "krr": "KRR",
    "mlp": "MLP",
    "val_select_peer": "Validation-selected peer",
    "val_select_pair_mean": "Validation-selected pair",
}

MODEL_DISPLAY_NAMES = {
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "allenai/Olmo-3-7B-Instruct": "OLMo-3-7B",
    "mistralai/Ministral-3-8B-Instruct-2512-BF16": "Ministral-3-8B",
    "Qwen/Qwen3.5-9B": "Qwen3.5-9B",
    "google/gemma-4-12B-it": "Gemma-4-12B",
    "ibm-granite/granite-3.3-8b-instruct": "Granite-3.3-8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "tiiuae/Falcon3-7B-Instruct": "Falcon3-7B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
}

TASK_DISPLAY_NAMES = {
    "hellaswag": "HellaSwag",
    "mmlu_pro": "MMLU-Pro",
    "rewardbench": "RewardBench",
}


def display_audit_cell(value: object) -> str:
    model_id, task_id = str(value).rsplit("__", 1)
    canonical_model_id = model_id.replace("--", "/", 1)
    model = MODEL_DISPLAY_NAMES.get(canonical_model_id, model_id)
    task = TASK_DISPLAY_NAMES.get(task_id, task_id.replace("_", " "))
    return f"{model} · {task}"


ARTIFACTS = (
    ("F02", "figures/main/fig02_controlled_information_paths.pdf", "views/fig02_controlled_information_paths.parquet"),
    ("F03", "figures/main/fig03_adapter_sensitivity.pdf", "views/fig03_adapter_sensitivity.parquet"),
    ("F04", "figures/main/fig04_state_disclosure_and_swap.pdf", "views/fig04_state_disclosure_and_swap.parquet"),
    ("F05", "figures/main/fig05_operational_frontier.pdf", "views/fig05_operational_frontier.parquet"),
    ("T02", "tables/main/tab02_audit_contracts.tex", "views/tab02_audit_contracts.csv"),
    ("T03", "tables/main/tab03_core_results.tex", "views/tab03_core_results.csv"),
    ("FA01", "figures/appendix/figA01_full_residual_ledger.pdf", "views/figA01_full_residual_ledger.parquet"),
    ("FA02", "figures/appendix/figA02_disaggregated_frontiers.pdf", "views/figA02_disaggregated_frontiers.parquet"),
    ("FA03", "figures/appendix/figA03_generalization.pdf", "views/figA03_generalization.parquet"),
    ("FA04", "figures/appendix/figA04_information_source_controls.pdf", "views/figA04_information_source_controls.parquet"),
    ("FA05", "figures/appendix/figA05_scaling_and_conditioning.pdf", "views/figA05_scaling_and_conditioning.parquet"),
    ("FA06", "figures/appendix/figA06_kernel_gauge.pdf", "views/figA06_kernel_gauge.parquet"),
    ("FA07", "figures/appendix/figA07_runtime_and_failures.pdf", "views/figA07_runtime_and_failures.parquet"),
    ("FA08", "figures/appendix/figA08_adapter_comparisons.pdf", "views/figA08_adapter_comparisons.parquet"),
    ("TA01", "tables/appendix/tabA01_model_inventory.tex", "views/tabA01_model_inventory.csv"),
    ("TA02", "tables/appendix/tabA02_data_splits.tex", "views/tabA02_data_splits.csv"),
    ("TA03", "tables/appendix/tabA03_adapter_state_contracts.tex", "views/tabA03_adapter_state_contracts.csv"),
    ("TA04", "tables/appendix/tabA04_full_results.tex", "views/tabA04_full_results.parquet"),
    ("TA05", "tables/appendix/tabA05_reconstruction_frontier.tex", "views/tabA05_reconstruction_frontier.parquet"),
    ("TA06", "tables/appendix/tabA06_control_results.tex", "views/tabA06_control_results.parquet"),
    ("TA07", "tables/appendix/tabA07_statistical_contrasts.tex", "views/tabA07_statistical_contrasts.csv"),
    ("TA08", "tables/appendix/tabA08_compute_failures_provenance.tex", "views/tabA08_compute_failures_provenance.parquet"),
    ("TA09", "tables/appendix/tabA09_adapter_comparisons.tex", "views/tabA09_adapter_comparisons.csv"),
    ("TA10", "tables/appendix/tabA10_ood_domain_summary.tex", "views/tabA10_ood_domain_summary.csv"),
)


def result(name: str) -> pd.DataFrame:
    path = TABLES_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "run_mode" not in frame or set(frame["run_mode"].dropna()) != {"full"}:
        raise RuntimeError("paper artifacts require full runs only: %s" % path)
    return frame


def save_view(frame: pd.DataFrame, filename: str) -> Path:
    suffix = Path(filename).suffix
    return write_table(
        frame,
        VIEW_DIR / filename
        if suffix in (".csv", ".parquet")
        else VIEW_DIR / (filename + ".parquet"),
    )


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_latex(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        frame.to_latex(index=False, escape=True, na_rep="N/A"), encoding="utf-8"
    )


def permutation_robustness_view() -> pd.DataFrame:
    protocol = read_json(CANONICAL_DIR / "protocol_lock.json")
    expected_models = {
        model["model_id"]: protocol["revisions"]["models"][model["model_id"]]
        for model in protocol["peers"] + protocol["targets"]
    }
    frames = []
    for model_id, revision in expected_models.items():
        path = (
            CANONICAL_DIR
            / "robustness_cache"
            / (model_id.replace("/", "--") + ".parquet")
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        if set(frame["revision"].astype(str)) != {revision}:
            raise RuntimeError("robustness cache revision mismatch: %s" % path)
        frames.append(frame)
    robust = pd.concat(frames, ignore_index=True)
    canonical = pd.read_parquet(CANONICAL_DIR / "matched_responses.parquet")
    coordinate_columns = ["coord_%d" % index for index in range(9)]
    canonical = canonical[["model_id", "task_id", "item_id"] + coordinate_columns]
    merged = robust.merge(
        canonical,
        on=["model_id", "task_id", "item_id"],
        suffixes=("_robust", "_canonical"),
        validate="many_to_one",
    )
    dimensions = {"mmlu_pro": 9, "hellaswag": 3, "rewardbench": 1}
    rows = []
    for row in merged.itertuples(index=False):
        dimension = dimensions[row.task_id]
        robust_coordinates = np.asarray(
            [getattr(row, "coord_%d_robust" % index) for index in range(dimension)]
        )
        canonical_coordinates = np.asarray(
            [getattr(row, "coord_%d_canonical" % index) for index in range(dimension)]
        )
        robust_probability = coordinates_to_probabilities(robust_coordinates)[0]
        canonical_probability = coordinates_to_probabilities(canonical_coordinates)[0]
        rows.append(
            {
                "model_id": row.model_id,
                "role": row.role,
                "task_id": row.task_id,
                "item_id": row.item_id,
                "view_id": row.view_id,
                "permutation_tv": float(
                    total_variation(
                        robust_probability[None, :], canonical_probability[None, :]
                    )[0]
                ),
                "status": "completed",
            }
        )
    return pd.DataFrame(rows)


def figure_02() -> None:
    clone = result("exp1_exact_peer_clone.csv")
    selector = result("exp2_convex_selector_worlds.csv")
    truth = result("exp3_random_truth_table_worlds.csv")
    planted = result("exp4_planted_bit_llm_geometry.csv")
    view = pd.concat(
        [
            clone.assign(panel="A", source="clone"),
            selector.assign(panel="C", source="selector"),
            truth.assign(panel="B", source="truth_table"),
            planted.assign(panel="D", source="planted_bits"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_view(view, "fig02_controlled_information_paths.parquet")
    selector_summary = selector.groupby("codec_bits", as_index=False).agg(
        distortion=("distortion", "mean"),
        world_decoding_error=("world_decoding_error", "first"),
        fano_lower_bound=("fano_lower_bound", "first"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].scatter(
        [0, 1, 2],
        [
            0,
            selector_summary.loc[
                selector_summary["codec_bits"] == 6, "distortion"
            ].iloc[0],
            truth[
                (truth["row_kind"] == "analytic")
                & (truth["stored_bits"] == truth["stored_bits"].max())
            ]["distortion"].iloc[0],
        ],
        color=["#666666", "#0072B2", "#D55E00"],
    )
    axes[0, 0].set_xticks([0, 1, 2], ["Peer clone", "64-way selector", "Truth table"])
    axes[0, 0].set_ylabel("Deployment distortion")
    axes[0, 0].set_ylim(0, 0.5)
    axes[0, 0].set_title("(a) Low-distortion controlled paths")
    analytic = truth[truth["row_kind"] == "analytic"].sort_values("distortion")
    axes[0, 1].plot(
        analytic["distortion"], analytic["rate_distortion_lower_bound"], color="black"
    )
    axes[0, 1].scatter(analytic["distortion"], analytic["stored_bits"], color="#D55E00")
    axes[0, 1].set_xlabel("Hamming distortion")
    axes[0, 1].set_ylabel("Complete coded bits")
    axes[0, 1].set_title("(b) Truth-table rate-distortion")
    axes[1, 0].plot(
        selector_summary["world_decoding_error"],
        selector_summary["fano_lower_bound"],
        color="#0072B2",
        marker="o",
    )
    axes[1, 0].set_xlabel("World decoding error")
    axes[1, 0].set_ylabel("Fano lower bound")
    axes[1, 0].set_title("(c) Selector identification")
    summary = planted[
        (planted["row_kind"] == "summary")
        & (planted["status"] == "completed")
        & (planted["inference_tier"] == "simultaneous_primary")
    ].copy()
    if (
        set(summary["world_law_scope"].astype(str)) != {PLANTED_SCOPE_ID}
        or set(summary["tv_strength"].astype(float)) != {PRIMARY_TV_STRENGTH}
        or set(summary["budget_cap"].astype(int)) != {PRIMARY_BUDGET}
    ):
        raise RuntimeError("Figure 2D received the wrong response-geometry family")
    for adapter_index, (adapter, values) in enumerate(summary.groupby("adapter")):
        color = COLORS[adapter]
        axes[1, 1].scatter(
            values["bits_per_planted_bit"],
            values["plugin_information_per_bit"],
            color="#999999",
            alpha=0.45,
            s=18,
            label="Plug-in descriptive" if adapter_index == 0 else None,
        )
        axes[1, 1].scatter(
            values["bits_per_planted_bit"],
            values["information_lower_bound_per_bit"],
            color=color,
            alpha=0.6,
            s=24,
            label=adapter.upper(),
        )
        median_x = float(values["bits_per_planted_bit"].median())
        median_y = float(values["information_lower_bound_per_bit"].median())
        lower, upper = np.quantile(
            values["information_lower_bound_per_bit"], [0.25, 0.75]
        )
        axes[1, 1].errorbar(
            median_x,
            median_y,
            yerr=[[median_y - lower], [upper - median_y]],
            color=color,
            marker="o",
            markersize=8,
            linewidth=2,
            capsize=3,
        )
    positive_x = summary["bits_per_planted_bit"].to_numpy(dtype=float)
    boundary_x = np.geomspace(positive_x.min(), positive_x.max(), 256)
    axes[1, 1].plot(
        boundary_x,
        np.minimum(1.0, boundary_x),
        color="black",
        linestyle="--",
        linewidth=1,
        label="Accounting boundary",
    )
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("Complete retained-state bits / planted bit")
    axes[1, 1].set_ylabel("Simultaneous information lower bound / bit")
    axes[1, 1].set_title(
        r"(d) Incremental planted information conditioned on oracle $Y_0$"
    )
    axes[1, 1].legend(fontsize=7)
    save_figure(fig, FIGURE_DIR / "main/fig02_controlled_information_paths.pdf")


def figure_03() -> None:
    frame = result("exp5_llm_adapter_sensitivity.csv")
    save_view(frame, "fig03_adapter_sensitivity.parquet")
    order = ["mean_all", "convex", "linear", "rff", "krr", "mlp"]
    display_order = [DISPLAY_NAMES[adapter] for adapter in order]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
    for _, cell in frame.groupby("audit_cell"):
        values = cell.set_index("adapter").reindex(order)
        axes[0].plot(
            display_order,
            values["tv"],
            color="#777777",
            alpha=0.4,
            marker="o",
            ms=3,
        )
    axes[0].plot(
        display_order,
        frame.groupby("adapter")["tv"].median().reindex(order),
        color="black",
        lw=2.5,
        marker="o",
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("IID test TV")
    axes[0].set_title("(a) Unmatched-state diagnostic")
    axes[0].tick_params(axis="x", rotation=22)
    for label in axes[0].get_xticklabels():
        label.set_ha("right")
    convex = frame[frame["adapter"] == "convex"][["audit_cell", "tv"]].rename(
        columns={"tv": "convex_tv"}
    )
    delta = frame.merge(convex, on="audit_cell")
    delta = delta[delta["adapter"].isin(["linear", "rff", "krr", "mlp"])]
    delta["delta"] = delta["tv"] - delta["convex_tv"]
    delta["audit_cell_label"] = delta["audit_cell"].map(display_audit_cell)
    sns.stripplot(
        data=delta,
        x="delta",
        y="audit_cell_label",
        hue="adapter",
        hue_order=["linear", "rff", "krr", "mlp"],
        palette={adapter: COLORS[adapter] for adapter in ("linear", "rff", "krr", "mlp")},
        legend=False,
        ax=axes[1],
    )
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_title("(b) Paired difference vs Convex")
    axes[1].set_ylabel("Target/task")
    epsilons = (0.01, 0.02, 0.05, 0.10, 0.20)
    for adapter in ("linear", "rff", "krr", "mlp"):
        selected = delta[delta["adapter"] == adapter]
        flips = [
            np.mean((selected["convex_tv"] <= epsilon) != (selected["tv"] <= epsilon))
            for epsilon in epsilons
        ]
        axes[2].plot(epsilons, flips, marker="o", label=adapter, color=COLORS[adapter])
    axes[2].set_xlabel("Residual threshold")
    axes[2].set_ylabel("Flip fraction")
    axes[2].set_title("(c) Residual-only verdict flips")
    axes[2].legend()
    save_figure(fig, FIGURE_DIR / "main/fig03_adapter_sensitivity.pdf")


def figure_04() -> None:
    inverse = result("exp6_krr_inverse_reconstruction.csv")
    disclosure = result("exp7_quantized_state_disclosure.csv")
    swap = result("exp8_target_state_swap.csv")
    view = pd.concat(
        [
            inverse.assign(panel="A"),
            disclosure.assign(panel="B_D"),
            swap.assign(panel="C"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_view(view, "fig04_state_disclosure_and_swap.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    adapter_order = [
        adapter for adapter in ADAPTERS if adapter in set(disclosure["adapter"])
    ]
    palette = {adapter: COLORS[adapter] for adapter in adapter_order}
    inverse_main = inverse[inverse["codec"].isin(("native_fp16", "native_fp32"))]
    sns.boxplot(
        data=inverse_main,
        x="codec",
        y="relative_l2_error",
        color="#D55E00",
        ax=axes[0, 0],
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("(a) KRR inverse recovery")
    axes[0, 0].set_xlabel("Native state codec")
    axes[0, 0].set_ylabel("Relative reconstruction error")
    axes[0, 0].text(
        0.02,
        0.03,
        "fp64 omitted: below the numerical reporting threshold",
        transform=axes[0, 0].transAxes,
        fontsize=8,
        color="#555555",
    )
    sns.scatterplot(
        data=disclosure,
        x="bits_per_fit_row",
        y="fit_recovery_mae",
        hue="adapter",
        hue_order=adapter_order,
        palette=palette,
        s=28,
        alpha=0.65,
        linewidth=0,
        legend=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("(b) Declared-decoder recovery")
    axes[0, 1].set_xlabel("Retained-state bits per fit row")
    axes[0, 1].set_ylabel("Fit recovery MAE")
    donor_swap = swap[swap["condition"] == "other_target"]
    sns.scatterplot(
        data=donor_swap,
        x="delta_swap",
        y="donor_target_tv_change",
        hue="adapter",
        hue_order=adapter_order,
        palette=palette,
        s=30,
        alpha=0.7,
        linewidth=0,
        legend=False,
        ax=axes[1, 0],
    )
    axes[1, 0].axhline(0, color="black", lw=1)
    axes[1, 0].axvline(0, color="black", lw=1)
    axes[1, 0].set_title("(c) State-swap direction")
    axes[1, 0].set_xlabel("Recipient error change")
    axes[1, 0].set_ylabel("Donor-target TV change (negative = toward donor)")
    bit_scale = np.log2(disclosure["complete_coded_bits"].clip(lower=2).astype(float))
    bit_scale = 18 + 28 * (bit_scale - bit_scale.min()) / max(
        bit_scale.max() - bit_scale.min(), 1.0
    )
    for adapter in adapter_order:
        selected = disclosure[disclosure["adapter"] == adapter]
        axes[1, 1].scatter(
            selected["fit_recovery_mae"],
            selected["iid_test_tv"],
            s=bit_scale.loc[selected.index],
            color=palette[adapter],
            alpha=0.55,
            linewidth=0,
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("(d) Disclosure and generalization")
    axes[1, 1].set_xlabel("Fit recovery MAE")
    axes[1, 1].set_ylabel("IID test TV")
    axes[1, 1].text(
        0.98,
        0.96,
        "marker area scales with log state size",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=palette[adapter],
            markeredgecolor="none",
            label=adapter,
        )
        for adapter in adapter_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(legend_handles),
        frameon=False,
        title="Adapter",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.0, w_pad=1.8)
    save_figure(fig, FIGURE_DIR / "main/fig04_state_disclosure_and_swap.pdf")


def figure_05() -> None:
    frame = result("exp9_error_state_query_frontier.csv")
    frame["pareto_efficient"] = False
    for _, values in frame[frame["status"] == "completed"].groupby(["audit_cell", "q"]):
        bits = values["complete_coded_bits"].to_numpy(dtype=float)
        errors = values["iid_test_tv"].to_numpy(dtype=float)
        efficient = []
        for index in range(len(values)):
            dominated = np.any(
                (bits <= bits[index])
                & (errors <= errors[index])
                & ((bits < bits[index]) | (errors < errors[index]))
            )
            efficient.append(not dominated)
        frame.loc[values.index, "pareto_efficient"] = efficient
    save_view(frame, "fig05_operational_frontier.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=True)
    adapter_order = [
        "mean_all",
        "fixed_peer",
        "fixed_pair_mean",
        "val_select_peer",
        "val_select_pair_mean",
        "convex",
        "linear",
        "rff",
        "krr",
        "mlp",
    ]
    baseline_styles = {
        "mean_all": ("-", "s"),
        "fixed_peer": ("--", "^"),
        "fixed_pair_mean": (":", "v"),
        "val_select_peer": ("-.", "P"),
        "val_select_pair_mean": ((0, (3, 1, 1, 1)), "X"),
    }
    display_names = {
        "mean_all": "mean all",
        "fixed_peer": "fixed peer",
        "fixed_pair_mean": "fixed pair mean",
        "val_select_peer": "validation-selected peer",
        "val_select_pair_mean": "validation-selected pair",
    }
    for q, ax in zip((1, 2, 3), axes):
        part = frame[(frame["q"] == q) & (frame["status"] == "completed")].copy()
        cell_paths = part.drop_duplicates(
            ["audit_cell", "adapter", "complete_coded_bits", "iid_test_tv"]
        )
        for _, cell_values in cell_paths.groupby(["audit_cell", "adapter"]):
            cell_values = cell_values.sort_values("complete_coded_bits")
            ax.plot(
                cell_values["complete_coded_bits"],
                cell_values["iid_test_tv"],
                color="#6f6f6f",
                alpha=0.10,
                lw=0.5,
                zorder=1,
            )
        for adapter in adapter_order:
            values = part[part["adapter"] == adapter]
            if values.empty:
                continue
            color = COLORS.get(adapter, "#777777")
            linestyle, marker = baseline_styles.get(adapter, ("-", "o"))
            summary = (
                values.groupby("budget_cap", as_index=False)
                .agg(
                    bits=("complete_coded_bits", "median"),
                    median=("iid_test_tv", "median"),
                    lower=("iid_test_tv", lambda x: x.quantile(0.25)),
                    upper=("iid_test_tv", lambda x: x.quantile(0.75)),
                )
                .sort_values("bits")
                .drop_duplicates("bits")
            )
            bits = summary["bits"].to_numpy(dtype=float)
            median = summary["median"].to_numpy(dtype=float)
            lower = summary["lower"].to_numpy(dtype=float)
            upper = summary["upper"].to_numpy(dtype=float)
            ax.plot(
                bits,
                median,
                marker=marker,
                markersize=4.5,
                markevery=max(1, len(bits) // 7),
                color=color,
                linestyle=linestyle,
                lw=2.0 if adapter not in baseline_styles else 1.6,
                label=display_names.get(adapter, adapter),
                zorder=3,
            )
            ax.fill_between(bits, lower, upper, color=color, alpha=0.12, linewidth=0)
            pareto = values[values["pareto_efficient"]]
            ax.scatter(
                pareto["complete_coded_bits"],
                pareto["iid_test_tv"],
                facecolors="none",
                edgecolors=color,
                s=16,
                linewidths=0.55,
                alpha=0.45,
                zorder=2,
            )
        ax.set_xscale("symlog", base=2, linthresh=2, linscale=0.75)
        ax.set_xticks(
            [0, 2, 2 ** 10, 2 ** 14, 2 ** 18], ["0", "2", "1K", "16K", "256K"]
        )
        ax.set_ylim(0, 1)
        ax.set_title("%d deployment peer%s" % (q, "" if q == 1 else "s"))
        ax.set_xlabel("Complete retained-state bits")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("IID test TV")
    available_adapters = set(frame.loc[frame["status"] == "completed", "adapter"])
    handles = []
    labels = []
    for adapter in adapter_order:
        if adapter not in available_adapters:
            continue
        color = COLORS.get(adapter, "#777777")
        linestyle, marker = baseline_styles.get(adapter, ("-", "o"))
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4.5,
                linewidth=2.0 if adapter not in baseline_styles else 1.6,
            )
        )
        labels.append(display_names.get(adapter, adapter))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        title="Deployment rule",
    )
    fig.text(
        0.995,
        0.985,
        "Thin gray trajectories show individual audit cells; bands show the interquartile range.",
        ha="right",
        va="top",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.96), w_pad=1.1)
    save_figure(fig, FIGURE_DIR / "main/fig05_operational_frontier.pdf")


def appendix_figures() -> None:
    frontier = result("exp9_error_state_query_frontier.csv")
    generalization = result("exp11_fit_size_generalization.csv")
    controls = result("exp10_peer_source_controls.csv")
    robustness = permutation_robustness_view()
    swap = result("exp8_target_state_swap.csv")
    inverse = result("exp6_krr_inverse_reconstruction.csv")
    gauge = result("exp12_kernel_scale_gauge.csv")
    completed = frontier[frontier["status"] == "completed"]
    save_view(completed, "figA01_full_residual_ledger.parquet")
    pivot = (
        completed.groupby(["audit_cell", "adapter", "q"], as_index=False)["iid_test_tv"]
        .min()
        .pivot(index="audit_cell", columns=["adapter", "q"], values="iid_test_tv")
    )
    display_pivot = pivot.copy()
    display_pivot.index = [display_audit_cell(value) for value in display_pivot.index]
    display_pivot.columns = [
        f"{DISPLAY_NAMES.get(adapter, adapter)} · q={q}"
        for adapter, q in display_pivot.columns
    ]
    fig, ax = plt.subplots(figsize=(16, 7.5))
    sns.heatmap(
        display_pivot,
        vmin=0,
        vmax=1,
        cmap="viridis",
        annot=True,
        fmt=".3f",
        annot_kws={"size": 7},
        linewidths=0.4,
        linecolor="white",
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xlabel("Deployment rule and queried peers")
    ax.set_ylabel("Target/task")
    ax.set_title("Full residual ledger: minimum IID test TV", fontsize=11)
    save_figure(fig, FIGURE_DIR / "appendix/figA01_full_residual_ledger.pdf")
    q3 = completed[(completed["q"] == 3) & (completed["complete_coded_bits"] > 0)]
    save_view(q3, "figA02_disaggregated_frontiers.parquet")
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True, sharey=True)
    for ax, (cell_id, values) in zip(axes.flat, q3.groupby("audit_cell")):
        for adapter, part in values.groupby("adapter"):
            ax.scatter(
                part["complete_coded_bits"],
                part["iid_test_tv"],
                s=20,
                label=adapter,
                color=COLORS.get(adapter),
            )
            ax.set_xscale("log", base=2)
        ax.set_title(display_audit_cell(cell_id), fontsize=8)
        ax.set_ylim(0, 1)
    adapter_order = [adapter for adapter in ADAPTERS if adapter in set(q3["adapter"])]
    fig.legend(
        [
            Line2D([0], [0], marker="o", linestyle="", color=COLORS[adapter])
            for adapter in adapter_order
        ],
        [DISPLAY_NAMES[adapter] for adapter in adapter_order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=len(adapter_order),
        frameon=False,
        title="Adapter",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, FIGURE_DIR / "appendix/figA02_disaggregated_frontiers.pdf")
    save_view(generalization, "figA03_generalization.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    adapter_order = [
        adapter for adapter in ADAPTERS if adapter in set(generalization["adapter"])
    ]
    split_order = ["fit", "validation", "iid_test", "ood_test"]
    split_dashes = {
        "fit": "",
        "validation": (4, 2),
        "iid_test": (1, 1),
        "ood_test": (6, 2, 1, 2),
    }
    sns.lineplot(
        data=generalization,
        x="fit_size",
        y="tv",
        hue="adapter",
        hue_order=adapter_order,
        palette={adapter: COLORS[adapter] for adapter in adapter_order},
        style="split",
        style_order=split_order,
        dashes=split_dashes,
        errorbar=None,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_title("(a) Fit-size trajectories")
    axes[0].set_xlabel("Fit-set size")
    axes[0].set_ylabel("Total variation")
    gaps = generalization.pivot_table(
        index=["audit_cell", "adapter", "fit_size"], columns="split", values="tv"
    ).reset_index()
    gaps["iid_fit_gap"] = gaps["iid_test"] - gaps["fit"]
    sns.stripplot(
        data=gaps, x="adapter", y="iid_fit_gap", hue="adapter", legend=False, ax=axes[1]
    )
    axes[1].set_title("(b) IID fit-to-test gap")
    axes[1].set_xlabel("Adapter")
    axes[1].set_ylabel("IID test TV - fit TV")
    sns.scatterplot(
        data=gaps,
        x="iid_test",
        y="ood_test",
        hue="adapter",
        legend=False,
        ax=axes[2],
    )
    axes[2].set_title("(c) IID and OOD error")
    axes[2].set_xlabel("IID test TV")
    axes[2].set_ylabel("OOD test TV")
    fig.legend(
        [Line2D([0], [0], color=COLORS[adapter], lw=2) for adapter in adapter_order],
        [DISPLAY_NAMES[adapter] for adapter in adapter_order],
        loc="lower center",
        bbox_to_anchor=(0.31, 0.075),
        ncol=len(adapter_order),
        frameon=False,
    )
    fig.legend(
        [
            Line2D([0], [0], color="black", lw=2, dashes=split_dashes[split])
            for split in split_order
        ],
        ["Fit", "Validation", "IID test", "OOD test"],
        loc="lower center",
        bbox_to_anchor=(0.78, 0.075),
        ncol=len(split_order),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=2.0)
    save_figure(fig, FIGURE_DIR / "appendix/figA03_generalization.pdf")
    control_view = pd.concat(
        [
            controls.assign(source="features"),
            swap.assign(source="state"),
            robustness.assign(source="prompt_permutation"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_view(control_view, "figA04_information_source_controls.parquet")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)
    swap_order = [adapter for adapter in ADAPTERS if adapter in set(swap["adapter"])]
    control_order = [
        adapter for adapter in ADAPTERS if adapter in set(controls["adapter"])
    ]
    sns.barplot(
        data=swap,
        x="condition",
        y="controlled_tv",
        hue="adapter",
        hue_order=swap_order,
        palette={adapter: COLORS[adapter] for adapter in swap_order},
        errorbar=None,
        ax=axes[0],
    )
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(
        data=controls[controls["q"] == 3],
        x="condition",
        y="iid_test_tv",
        hue="adapter",
        hue_order=control_order,
        palette={adapter: COLORS[adapter] for adapter in control_order},
        errorbar=None,
        ax=axes[1],
    )
    axes[1].tick_params(axis="x", rotation=20)
    sns.barplot(
        data=controls,
        x="condition",
        y="iid_test_tv",
        hue="q",
        errorbar=None,
        ax=axes[2],
    )
    axes[2].tick_params(axis="x", rotation=20)
    sns.boxplot(
        data=robustness,
        x="task_id",
        y="permutation_tv",
        hue="role",
        showfliers=False,
        ax=axes[3],
    )
    axes[3].tick_params(axis="x", rotation=20)
    axes[0].set_title("(a) Swapped state source")
    axes[0].set_xlabel("State source")
    axes[0].set_ylabel("Controlled TV")
    axes[1].set_title("(b) Feature-source controls")
    axes[1].set_xlabel("Feature condition")
    axes[1].set_ylabel("IID test TV")
    axes[2].set_title("(c) Query-count controls")
    axes[2].set_xlabel("Feature condition")
    axes[2].set_ylabel("IID test TV")
    axes[3].set_title("(d) Prompt permutations")
    axes[3].set_xlabel("Task")
    axes[3].set_ylabel("Permutation TV")
    task_tick_labels = [
        TASK_DISPLAY_NAMES.get(label.get_text(), label.get_text())
        for label in axes[3].get_xticklabels()
    ]
    axes[3].set_xticks(axes[3].get_xticks(), task_tick_labels, rotation=20, ha="right")
    save_figure(fig, FIGURE_DIR / "appendix/figA04_information_source_controls.pdf")
    scaling = pd.concat(
        [
            frontier.assign(source="frontier"),
            generalization.assign(source="generalization"),
            inverse.assign(source="conditioning"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_view(scaling, "figA05_scaling_and_conditioning.parquet")
    adapter_order = [
        adapter for adapter in ADAPTERS if adapter in set(completed["adapter"])
    ]
    adapter_palette = {adapter: COLORS[adapter] for adapter in adapter_order}
    trained_frontier = completed[completed["adapter"].isin(adapter_order)]
    fixed_reference = completed[
        completed["adapter"].isin(("fixed_peer", "fixed_pair_mean", "mean_all"))
    ].assign(reference_rule="Fixed roster")
    selected_reference = completed[
        completed["adapter"].isin(
            ("val_select_peer", "val_select_pair_mean", "mean_all")
        )
    ].assign(reference_rule="Validation-selected roster")
    reference_frontier = pd.concat(
        [fixed_reference, selected_reference], ignore_index=True
    )
    reference_styles = {
        "Fixed roster": {"color": "#666666", "linestyle": "--", "marker": "D"},
        "Validation-selected roster": {
            "color": "#222222",
            "linestyle": ":",
            "marker": "X",
        },
    }
    generalization_iid = generalization[
        generalization["split"] == "iid_test"
    ]
    codec_order = [
        codec
        for codec in ("native_fp16", "native_fp32", "native_fp64")
        if codec in set(inverse["codec"])
    ]
    codec_colors = dict(
        zip(codec_order, sns.color_palette("deep", n_colors=len(codec_order)))
    )
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    sns.lineplot(
        data=trained_frontier,
        x="q",
        y="iid_test_tv",
        hue="adapter",
        hue_order=adapter_order,
        palette=adapter_palette,
        errorbar=None,
        legend=False,
        ax=axes[0],
    )
    for reference_rule, style in reference_styles.items():
        summary = (
            reference_frontier[
                reference_frontier["reference_rule"] == reference_rule
            ]
            .groupby("q", as_index=False)["iid_test_tv"]
            .median()
            .sort_values("q")
        )
        axes[0].plot(
            summary["q"],
            summary["iid_test_tv"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4,
            linewidth=1.5,
            zorder=2,
        )
    sns.lineplot(
        data=generalization_iid,
        x="fit_size",
        y="tv",
        hue="adapter",
        hue_order=adapter_order,
        palette=adapter_palette,
        errorbar=None,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_xscale("log", base=2)
    sns.scatterplot(
        data=inverse,
        x="condition_number",
        y="relative_l2_error",
        hue="codec",
        hue_order=codec_order,
        palette=codec_colors,
        legend=False,
        ax=axes[2],
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    sns.scatterplot(
        data=inverse,
        x="condition_number",
        y="probability_tv",
        hue="codec",
        hue_order=codec_order,
        palette=codec_colors,
        legend=False,
        ax=axes[3],
    )
    axes[3].set_xscale("log")
    axes[0].set_title("(a) Query dimension")
    axes[0].set_xlabel("Queried peers (q)")
    axes[0].set_ylabel("IID test TV")
    axes[1].set_title("(b) Fit-set size")
    axes[1].set_xlabel("Fit-set size")
    axes[1].set_ylabel("IID test TV")
    axes[2].set_title("(c) State reconstruction")
    axes[2].set_xlabel("Condition number")
    axes[2].set_ylabel("Relative L2 error")
    axes[3].set_title("(d) Prediction reconstruction")
    axes[3].set_xlabel("Condition number")
    axes[3].set_ylabel("Prediction TV")
    fig.legend(
        [Line2D([0], [0], color=COLORS[adapter], lw=2) for adapter in adapter_order],
        [DISPLAY_NAMES[adapter] for adapter in adapter_order],
        loc="lower center",
        bbox_to_anchor=(0.22, 0.035),
        ncol=len(adapter_order),
        frameon=False,
        title="Adapter",
    )
    fig.legend(
        [
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4,
                linewidth=1.5,
            )
            for style in reference_styles.values()
        ],
        list(reference_styles),
        loc="lower center",
        bbox_to_anchor=(0.58, 0.035),
        ncol=len(reference_styles),
        frameon=False,
        title="Reference rule",
    )
    fig.legend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=codec_colors[codec],
            )
            for codec in codec_order
        ],
        [codec.replace("native_", "").upper() for codec in codec_order],
        loc="lower center",
        bbox_to_anchor=(0.87, 0.035),
        ncol=len(codec_order),
        frameon=False,
        title="State codec",
    )
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.5)
    save_figure(fig, FIGURE_DIR / "appendix/figA05_scaling_and_conditioning.pdf")
    save_view(gauge, "figA06_kernel_gauge.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    sns.lineplot(
        data=gauge,
        x="kernel_scale",
        y="max_prediction_difference",
        errorbar=None,
        ax=axes[0],
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("symlog", linthresh=1e-16)
    axes[0].set_title("(a) Prediction invariance")
    axes[0].set_xlabel("Kernel scale")
    axes[0].set_ylabel("Maximum prediction difference")
    sns.lineplot(
        data=gauge,
        x="kernel_scale",
        y="inverse_lambda_prime",
        errorbar=None,
        ax=axes[1],
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Inverse regularization")
    axes[1].set_xlabel("Kernel scale")
    axes[1].set_ylabel("Inverse effective regularization")
    sns.lineplot(
        data=gauge, x="kernel_scale", y="complete_coded_bits", errorbar=None, ax=axes[2]
    )
    axes[2].set_xscale("log")
    axes[2].set_title("(c) Serialized state size")
    axes[2].set_xlabel("Kernel scale")
    axes[2].set_ylabel("Complete coded bits")
    save_figure(fig, FIGURE_DIR / "appendix/figA06_kernel_gauge.pdf")
    runtime_frames = [
        pd.read_parquet(path)
        for path in (CANONICAL_DIR / "preflight").glob("*.parquet")
    ]
    runtime = (
        pd.concat(runtime_frames, ignore_index=True)
        if runtime_frames
        else pd.DataFrame(
            columns=["model_id", "seconds_per_view", "peak_allocated_bytes", "status"]
        )
    )
    save_view(runtime, "figA07_runtime_and_failures.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), sharey=True)
    if not runtime.empty:
        runtime = runtime.assign(
            model_label=runtime["model_id"].map(MODEL_DISPLAY_NAMES)
        )
        if runtime["model_label"].isna().any():
            unknown = sorted(
                runtime.loc[runtime["model_label"].isna(), "model_id"].unique()
            )
            raise RuntimeError("missing runtime display names: %s" % unknown)
        model_order = [
            MODEL_DISPLAY_NAMES[model_id]
            for model_id in MODEL_DISPLAY_NAMES
            if MODEL_DISPLAY_NAMES[model_id] in set(runtime["model_label"])
        ]
        sns.barplot(
            data=runtime,
            y="model_label",
            x="seconds_per_view",
            order=model_order,
            errorbar=None,
            ax=axes[0],
        )
        sns.barplot(
            data=runtime,
            y="model_label",
            x="peak_allocated_bytes",
            order=model_order,
            errorbar=None,
            ax=axes[1],
        )
        failure = (
            runtime.assign(failed=runtime["status"] != "completed")
            .groupby("model_label", as_index=False)["failed"]
            .mean()
        )
        sns.barplot(
            data=failure,
            y="model_label",
            x="failed",
            order=model_order,
            ax=axes[2],
        )
        if float(failure["failed"].max()) == 0.0:
            axes[2].set_xlim(0, 1)
            axes[2].text(
                0.5,
                0.5,
                "No recorded failures",
                transform=axes[2].transAxes,
                ha="center",
                va="center",
                color="#555555",
            )
    axes[0].set_title("(a) Inference time")
    axes[0].set_xlabel("Seconds per view")
    axes[0].set_ylabel("Model")
    axes[1].set_title("(b) Peak allocated memory")
    axes[1].set_xlabel("Bytes")
    axes[1].set_ylabel("")
    axes[2].set_title("(c) Failure fraction")
    axes[2].set_xlabel("Fraction")
    axes[2].set_ylabel("")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "appendix/figA07_runtime_and_failures.pdf")


def adapter_comparison_views():
    roster_labels = {
        "primary_targets": "Primary grid",
        "independent_instruction_pool": "Independent roster",
        "reference_instruction_pool": "Reference roster",
        "cross_role_instruction_pool": "Cross-role roster",
    }
    family_labels = {
        "primary_targets": "Primary targets",
        "peer_roster_replication": "Peer-roster replication",
        "target_role_replication": "Target-role replication",
    }
    specifications = (
        (
            "primary_targets",
            result("exp5_llm_adapter_sensitivity.csv"),
            "tv",
            ("linear", "rff", "krr", "mlp"),
        ),
        (
            "peer_roster_replication",
            result("exp13_peer_roster_replication.csv"),
            "iid_test_tv",
            ("rff", "krr"),
        ),
        (
            "target_role_replication",
            result("exp14_target_role_replication.csv"),
            "iid_test_tv",
            ("rff", "krr"),
        ),
    )
    effect_rows = []
    for family, frame, value_column, adapters in specifications:
        selected = frame.copy()
        if "analysis" in selected:
            selected = selected[selected["analysis"] == "native"].copy()
        if "roster_id" not in selected:
            selected["roster_id"] = "primary_targets"
        index_columns = [
            "roster_id",
            "audit_cell",
            "target_id",
            "task_id",
        ]
        wide = selected.pivot(
            index=index_columns, columns="adapter", values=value_column
        ).reset_index()
        required = {"convex", *adapters}
        if not required.issubset(wide.columns):
            raise RuntimeError("adapter comparison is missing registered methods")
        for row in wide.itertuples(index=False):
            target_label = MODEL_DISPLAY_NAMES.get(row.target_id, row.target_id)
            task_label = TASK_DISPLAY_NAMES.get(row.task_id, row.task_id)
            for adapter in adapters:
                effect_rows.append(
                    {
                        "family": family,
                        "family_label": family_labels[family],
                        "roster_id": row.roster_id,
                        "roster_label": roster_labels[row.roster_id],
                        "audit_cell": row.audit_cell,
                        "audit_cell_label": f"{target_label} · {task_label}",
                        "target_id": row.target_id,
                        "task_id": row.task_id,
                        "adapter": adapter,
                        "delta_tv_vs_convex": float(
                            getattr(row, adapter) - getattr(row, "convex")
                        ),
                    }
                )
    effects = pd.DataFrame(effect_rows)
    summary_rows = []
    for (family, roster_id, adapter), values in effects.groupby(
        ["family", "roster_id", "adapter"], sort=False
    ):
        delta = values["delta_tv_vs_convex"].to_numpy(dtype=float)
        leave_one_out = np.asarray(
            [np.delete(delta, index).mean() for index in range(len(delta))]
        )
        summary_rows.append(
            {
                "family": family,
                "family_label": family_labels[family],
                "roster_id": roster_id,
                "roster_label": roster_labels[roster_id],
                "adapter": adapter,
                "mean_delta_tv": delta.mean(),
                "median_delta_tv": np.median(delta),
                "improved_cells": int((delta < 0).sum()),
                "total_cells": len(delta),
                "improved_fraction": float((delta < 0).mean()),
                "best_cell_delta": delta.min(),
                "worst_cell_delta": delta.max(),
                "loo_mean_min": leave_one_out.min(),
                "loo_mean_max": leave_one_out.max(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return effects, summary


def figure_a08_adapter_comparisons() -> None:
    effects, _ = adapter_comparison_views()
    save_view(effects, "figA08_adapter_comparisons.parquet")
    families = (
        "primary_targets",
        "peer_roster_replication",
        "target_role_replication",
    )
    titles = (
        "(a) Primary targets",
        "(b) Peer-roster replication",
        "(c) Target-role replication",
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), sharey=True)
    for index, (ax, family, title) in enumerate(zip(axes, families, titles)):
        panel = effects[
            (effects["family"] == family)
            & effects["adapter"].isin(("rff", "krr"))
        ]
        sns.stripplot(
            data=panel,
            x="roster_label",
            y="delta_tv_vs_convex",
            hue="adapter",
            hue_order=("rff", "krr"),
            palette=COLORS,
            dodge=True,
            jitter=0.14,
            size=5,
            alpha=0.8,
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=12)
        if index == 0:
            ax.set_ylabel("IID test TV difference vs Convex")
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, [label.upper() for label in labels], title="Adapter")
        else:
            ax.set_ylabel("")
            ax.get_legend().remove()
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "appendix/figA08_adapter_comparisons.pdf")


def ood_domain_summary() -> pd.DataFrame:
    split = pd.read_parquet(CANONICAL_DIR / "split_manifest.parquet")
    domains = (
        split[split["split"] == "ood_test"]
        .groupby(["task_id", "domain"], as_index=False)
        .size()
        .rename(columns={"size": "items"})
    )
    summary = domains.groupby("task_id", as_index=False).agg(
        ood_domains=("domain", "nunique"),
        ood_items=("items", "sum"),
        smallest_domain=("items", "min"),
        largest_domain=("items", "max"),
    )
    generalization = result("exp11_fit_size_generalization.csv")
    bootstrap = (
        generalization[generalization["split"] == "ood_test"]
        .groupby("task_id")["bootstrap_unit"]
        .agg(lambda values: "|".join(sorted(set(values.astype(str)))))
        .rename("bootstrap_unit")
        .reset_index()
    )
    return summary.merge(bootstrap, on="task_id", validate="one_to_one")


def _paired_contrast_row(
    contrast_id: str,
    hypothesis: str,
    label: str,
    left: np.ndarray,
    right: np.ndarray,
    clusters: np.ndarray,
    sampling_unit: str = "item_cluster",
    **keys: object,
) -> Dict[str, object]:
    estimate, lower, upper = paired_bootstrap_interval(left, right, clusters=clusters)
    cluster_delta = (
        pd.DataFrame(
            {
                "cluster": np.asarray(clusters).astype(str),
                "delta": np.asarray(left, dtype=float) - np.asarray(right, dtype=float),
            }
        )
        .groupby("cluster", as_index=False)["delta"]
        .mean()["delta"]
    )
    return {
        "contrast_id": contrast_id,
        "hypothesis": hypothesis,
        "contrast": label,
        **keys,
        "estimate": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "paired_items": len(left),
        "paired_clusters": len(cluster_delta),
        "better": int((cluster_delta < -1e-6).sum()),
        "tied": int((cluster_delta.abs() <= 1e-6).sum()),
        "worse": int((cluster_delta > 1e-6).sum()),
        "sampling_unit": sampling_unit,
    }


def statistical_contrasts() -> pd.DataFrame:
    loss_dir = TABLES_DIR / "item_losses"
    required = {
        "native": loss_dir / "exp5_native_item_losses.parquet",
        "swap": loss_dir / "exp8_swap_item_losses.parquet",
        "frontier": loss_dir / "exp9_frontier_item_losses.parquet",
        "controls": loss_dir / "exp10_control_item_losses.parquet",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = []
    native = pd.read_parquet(required["native"])
    for cell_id, values in native[native["adapter"].isin(("convex", "krr"))].groupby(
        "audit_cell"
    ):
        pivot = values.pivot(
            index=["item_id", "item_cluster"], columns="adapter", values="tv_loss"
        ).dropna()
        rows.append(
            _paired_contrast_row(
                "C1",
                "adapter_sensitivity",
                "KRR - Convex (native)",
                pivot["krr"].to_numpy(),
                pivot["convex"].to_numpy(),
                pivot.index.get_level_values("item_cluster").to_numpy(),
                audit_cell=cell_id,
                q=3,
                budget_cap="native",
            )
        )

    frontier = pd.read_parquet(required["frontier"])
    selected = frontier[
        frontier["adapter"].isin(("convex", "krr"))
        & frontier["budget_cap"].isin(MATCHED_BUDGETS)
    ]
    for (cell_id, q, budget), values in selected.groupby(
        ["audit_cell", "q", "budget_cap"]
    ):
        if set(values["adapter"]) != {"convex", "krr"}:
            continue
        pivot = values.pivot(
            index=["item_id", "item_cluster"], columns="adapter", values="tv_loss"
        ).dropna()
        rows.append(
            _paired_contrast_row(
                "C2",
                "matched_budget_frontier",
                "budget-matched KRR - Convex",
                pivot["krr"].to_numpy(),
                pivot["convex"].to_numpy(),
                pivot.index.get_level_values("item_cluster").to_numpy(),
                audit_cell=cell_id,
                q=int(q),
                budget_cap=int(budget),
            )
        )

    swap = pd.read_parquet(required["swap"])
    for (adapter, condition), values in swap.groupby(["adapter", "condition"]):
        clusters = (
            values["task_id"].astype(str)
            + "|"
            + values["recipient"].astype(str)
            + "|"
            + values["donor"].astype(str)
            + "|"
            + values["item_cluster"].astype(str)
        ).to_numpy()
        rows.append(
            _paired_contrast_row(
                "C3",
                "state_dependence",
                "controlled state - correct state",
                values["controlled_tv_loss"].to_numpy(),
                values["correct_tv_loss"].to_numpy(),
                clusters,
                audit_cell="pooled_fixed_cells",
                adapter=adapter,
                condition=condition,
                q=3,
            )
        )
        if condition == "other_target":
            rows.append(
                _paired_contrast_row(
                    "C3b",
                    "state_dependence",
                    "donor-state prediction - correct-state prediction to donor target",
                    values["donor_target_controlled_tv_loss"].to_numpy(),
                    values["donor_target_correct_tv_loss"].to_numpy(),
                    clusters,
                    audit_cell="pooled_fixed_cells",
                    adapter=adapter,
                    condition="donor_target_tv_change",
                    q=3,
                )
            )

    controls = pd.read_parquet(required["controls"])
    q3 = controls[controls["q"] == 3]
    for (cell_id, adapter), values in q3.groupby(["audit_cell", "adapter"]):
        pivot = values.pivot(
            index=["item_id", "item_cluster"], columns="condition", values="tv_loss"
        ).dropna()
        for condition in ("row_shuffled", "channel_permuted"):
            rows.append(
                _paired_contrast_row(
                    "C4",
                    "peer_source_dependence",
                    "%s peers - actual peers" % condition,
                    pivot[condition].to_numpy(),
                    pivot["actual"].to_numpy(),
                    pivot.index.get_level_values("item_cluster").to_numpy(),
                    audit_cell=cell_id,
                    adapter=adapter,
                    condition=condition,
                    q=3,
                )
            )

    role_rotation = result("exp14_target_role_replication.csv")
    role_native = role_rotation[role_rotation["analysis"] == "native"]
    for (roster_id, adapter), values in role_native[
        role_native["adapter"].isin(("rff", "krr"))
    ].groupby(["roster_id", "adapter"]):
        convex = role_native[
            (role_native["roster_id"] == roster_id)
            & (role_native["adapter"] == "convex")
        ][["audit_cell", "iid_test_tv"]].rename(columns={"iid_test_tv": "convex"})
        paired = values[["audit_cell", "iid_test_tv"]].merge(
            convex, on="audit_cell", validate="one_to_one"
        )
        rows.append(
            _paired_contrast_row(
                "C6",
                "target_role_replication",
                "%s - Convex (secondary targets, native)" % adapter.upper(),
                paired["iid_test_tv"].to_numpy(),
                paired["convex"].to_numpy(),
                paired["audit_cell"].to_numpy(),
                sampling_unit="audit_cell",
                audit_cell="secondary_targets_pooled",
                roster_id=roster_id,
                adapter=adapter,
                q=3,
                budget_cap="native",
            )
        )

    role_frontier = role_rotation[
        (role_rotation["analysis"] == "frontier")
        & (role_rotation["status"] == "completed")
    ]
    for (roster_id, adapter, budget), values in role_frontier[
        role_frontier["adapter"].isin(("rff", "krr"))
    ].groupby(["roster_id", "adapter", "budget_cap"]):
        convex = role_frontier[
            (role_frontier["roster_id"] == roster_id)
            & (role_frontier["adapter"] == "convex")
            & (role_frontier["budget_cap"] == budget)
        ][["audit_cell", "iid_test_tv"]].rename(columns={"iid_test_tv": "convex"})
        paired = values[["audit_cell", "iid_test_tv"]].merge(
            convex, on="audit_cell", validate="one_to_one"
        )
        if paired.empty:
            continue
        rows.append(
            _paired_contrast_row(
                "C7",
                "target_role_replication",
                "%s - Convex (secondary targets, matched budget)" % adapter.upper(),
                paired["iid_test_tv"].to_numpy(),
                paired["convex"].to_numpy(),
                paired["audit_cell"].to_numpy(),
                sampling_unit="audit_cell",
                audit_cell="secondary_targets_pooled",
                roster_id=roster_id,
                adapter=adapter,
                q=3,
                budget_cap=int(budget),
            )
        )

    role_swaps = role_rotation[role_rotation["analysis"] == "state_swap"]
    for (roster_id, adapter), values in role_swaps.groupby(
        ["roster_id", "adapter"]
    ):
        swap_ids = (
            values["task_id"].astype(str)
            + "|"
            + values["recipient"].astype(str)
            + "|"
            + values["donor"].astype(str)
        ).to_numpy()
        rows.append(
            _paired_contrast_row(
                "C8",
                "target_role_replication",
                "controlled state - correct state (secondary targets)",
                values["controlled_tv"].to_numpy(),
                values["correct_tv"].to_numpy(),
                swap_ids,
                sampling_unit="directed_target_swap",
                audit_cell="secondary_targets_pooled",
                roster_id=roster_id,
                adapter=adapter,
                q=3,
            )
        )

    planted = result("exp4_planted_bit_llm_geometry.csv")
    primary = planted[
        (planted["row_kind"] == "summary")
        & (planted["inference_tier"] == "simultaneous_primary")
    ]
    for row in primary.itertuples(index=False):
        rows.append(
            {
                "contrast_id": "C5",
                "hypothesis": "controlled_information",
                "contrast": "planted-bit information lower bound",
                "audit_cell": row.audit_cell,
                "adapter": row.adapter,
                "q": 3,
                "budget_cap": int(row.budget_cap),
                "condition": "semantic_tv=%s; delta=%s"
                % (row.tv_strength, row.delta),
                "estimate": row.information_lower_bound_per_bit,
                "ci_lower": row.information_lower_bound_per_bit,
                "ci_upper": min(1.0, row.bits_per_planted_bit),
                "bit_error": row.bit_error,
                "bit_error_upper_bound": row.bit_error_upper_bound,
                "minimum_success": row.minimum_success,
                "strong_point": row.strong_point,
                "primary_family_pass_count": row.primary_family_pass_count,
                "primary_family_success": row.primary_family_success,
                "paired_items": np.nan,
                "paired_clusters": int(row.world_count),
                "better": np.nan,
                "tied": np.nan,
                "worse": np.nan,
                "sampling_unit": "precommitted_world",
            }
        )
    return pd.DataFrame(rows)


def tables() -> None:
    protocol = read_json(CANONICAL_DIR / "protocol_lock.json")
    freeze = read_json(CANONICAL_DIR / "state_freeze.json")
    split = pd.read_parquet(CANONICAL_DIR / "split_manifest.parquet")
    frontier = result("exp9_error_state_query_frontier.csv")
    contract_rows = []
    for target in protocol["targets"]:
        for task in protocol["tasks"]:
            counts = (
                split[split["task_id"] == task["task_id"]]
                .groupby("split")
                .size()
                .to_dict()
            )
            contract_rows.append(
                {
                    "target": target["model_id"],
                    "model_revision": protocol["revisions"]["models"][
                        target["model_id"]
                    ],
                    "task": task["task_id"],
                    "dataset_revision": protocol["revisions"]["datasets"][
                        task["dataset_id"]
                    ],
                    "fit": counts.get("fit", 0),
                    "validation": counts.get("validation", 0),
                    "iid_test": counts.get("iid_test", 0),
                    "ood_test": counts.get("ood_test", 0),
                    "prompt_manifest_sha256": sha256_file(
                        CANONICAL_DIR / "prompt_manifest.parquet"
                    ),
                    "protocol_contract_hash": protocol["contract_hash"],
                    "source_tree_hash": freeze["source_tree_hash"],
                    "requirements_sha256": freeze["requirements_sha256"],
                    "transformers_commit": protocol["transformers_commit"],
                    "python": "3.11.9",
                    "loss": "TV",
                    "q_grid": "1,2,3",
                    "validation_rule": "lowest validation TV under complete-bit cap",
                }
            )
    tab02 = pd.DataFrame(contract_rows)
    save_view(tab02, "tab02_audit_contracts.csv")
    save_latex(tab02, LATEX_DIR / "main/tab02_audit_contracts.tex")
    selected_frontier = frontier[
        (frontier["q"] == 3) & (frontier["budget_cap"].isin(MATCHED_BUDGETS))
    ]
    core = selected_frontier.groupby(
        ["adapter", "q", "budget_cap", "status"], as_index=False
    ).agg(
        completed_cells=("audit_cell", "nunique"),
        median_test_tv=("iid_test_tv", "median"),
        median_bits=("complete_coded_bits", "median"),
    )
    convex = selected_frontier[selected_frontier["adapter"] == "convex"][
        ["audit_cell", "budget_cap", "iid_test_tv"]
    ].rename(columns={"iid_test_tv": "convex_tv"})
    paired = selected_frontier.merge(
        convex, on=["audit_cell", "budget_cap"], how="left"
    )
    paired["delta_vs_convex"] = paired["iid_test_tv"] - paired["convex_tv"]
    paired["better"] = paired["delta_vs_convex"] < -1e-6
    paired["tied"] = paired["delta_vs_convex"].abs() <= 1e-6
    paired["worse"] = paired["delta_vs_convex"] > 1e-6
    flip_threshold = float(paired["iid_test_tv"].median())
    paired["verdict_flip_median"] = (
        paired["iid_test_tv"] <= flip_threshold
    ) != (
        paired["convex_tv"] <= flip_threshold
    )
    comparison = paired.groupby(["adapter", "q", "budget_cap"], as_index=False).agg(
        paired_delta_vs_convex=("delta_vs_convex", "median"),
        better=("better", "sum"),
        tied=("tied", "sum"),
        worse=("worse", "sum"),
        verdict_flips_at_median=("verdict_flip_median", "sum"),
    )
    core = core.merge(comparison, on=["adapter", "q", "budget_cap"], how="left")
    save_view(core, "tab03_core_results.csv")
    save_latex(core, LATEX_DIR / "main/tab03_core_results.tex")
    inventory_rows = []
    for path in (CANONICAL_DIR / "preflight").glob("*_inventory.json"):
        inventory_rows.append(read_json(path))
    inventory = pd.DataFrame(inventory_rows)
    save_view(inventory, "tabA01_model_inventory.csv")
    save_latex(inventory, LATEX_DIR / "appendix/tabA01_model_inventory.tex")
    split_summary = (
        split.groupby(["task_id", "split", "domain"], as_index=False)
        .size()
        .rename(columns={"size": "items"})
    )
    save_view(split_summary, "tabA02_data_splits.csv")
    save_latex(split_summary, LATEX_DIR / "appendix/tabA02_data_splits.tex")
    adapter_contract = pd.DataFrame(
        [
            {
                "adapter": adapter,
                "objects_in_C": "peer table, public transforms and seeds",
                "objects_in_state": "weights, selected metadata and scales",
                "lambda_grid": len(LAMBDA_GRID),
                "q": "1,2,3",
                "decoder": "canonical bitstream decoder",
            }
            for adapter in ADAPTERS
        ]
    )
    save_view(adapter_contract, "tabA03_adapter_state_contracts.csv")
    save_latex(
        adapter_contract, LATEX_DIR / "appendix/tabA03_adapter_state_contracts.tex"
    )
    full_results = pd.concat(
        [
            result("exp5_llm_adapter_sensitivity.csv"),
            frontier,
            result("exp11_fit_size_generalization.csv"),
            result("exp13_peer_roster_replication.csv"),
            result("exp14_target_role_replication.csv"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_view(full_results, "tabA04_full_results.parquet")
    save_latex(full_results, LATEX_DIR / "appendix/tabA04_full_results.tex")
    reconstruction = pd.concat(
        [
            result("exp6_krr_inverse_reconstruction.csv"),
            result("exp7_quantized_state_disclosure.csv"),
            frontier,
        ],
        ignore_index=True,
        sort=False,
    )
    save_view(reconstruction, "tabA05_reconstruction_frontier.parquet")
    save_latex(
        reconstruction, LATEX_DIR / "appendix/tabA05_reconstruction_frontier.tex"
    )
    controls = pd.concat(
        [
            result("exp1_exact_peer_clone.csv"),
            result("exp2_convex_selector_worlds.csv"),
            result("exp3_random_truth_table_worlds.csv"),
            result("exp4_planted_bit_llm_geometry.csv"),
            result("exp8_target_state_swap.csv"),
            result("exp10_peer_source_controls.csv"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_view(controls, "tabA06_control_results.parquet")
    save_latex(controls, LATEX_DIR / "appendix/tabA06_control_results.tex")
    contrasts = statistical_contrasts()
    save_view(contrasts, "tabA07_statistical_contrasts.csv")
    save_latex(contrasts, LATEX_DIR / "appendix/tabA07_statistical_contrasts.tex")
    runtime_frames = [
        pd.read_parquet(path)
        for path in (CANONICAL_DIR / "preflight").glob("*.parquet")
    ]
    runtime = (
        pd.concat(runtime_frames, ignore_index=True)
        if runtime_frames
        else pd.DataFrame(columns=["model_id", "status"])
    )
    save_view(runtime, "tabA08_compute_failures_provenance.parquet")
    save_latex(runtime, LATEX_DIR / "appendix/tabA08_compute_failures_provenance.tex")
    _, adapter_summary = adapter_comparison_views()
    save_view(adapter_summary, "tabA09_adapter_comparisons.csv")
    save_latex(
        adapter_summary, LATEX_DIR / "appendix/tabA09_adapter_comparisons.tex"
    )
    domain_summary = ood_domain_summary()
    save_view(domain_summary, "tabA10_ood_domain_summary.csv")
    save_latex(
        domain_summary, LATEX_DIR / "appendix/tabA10_ood_domain_summary.tex"
    )


def manifest() -> None:
    rows = []
    expected_keys = CANONICAL_DIR / "expected_run_keys.parquet"
    source = PROJECT_ROOT / "scripts/build_artifacts.py"
    for artifact_id, output_name, view_name in ARTIFACTS:
        output = PROJECT_ROOT / output_name
        view = PROJECT_ROOT / view_name
        for path in (output, view, expected_keys):
            if not path.is_file():
                raise FileNotFoundError(path)
        rows.append(
            {
                "artifact_id": artifact_id,
                "status": "rendered",
                "canonical_view": view_name,
                "canonical_view_sha256": sha256_file(view),
                "source_query": "scripts/build_artifacts.py",
                "source_query_sha256": sha256_file(source),
                "render_script": "scripts/build_artifacts.py",
                "render_script_sha256": sha256_file(source),
                "expected_keys_view": str(expected_keys.relative_to(PROJECT_ROOT)),
                "expected_keys_view_sha256": sha256_file(expected_keys),
                "output_file": output_name,
                "output_sha256": sha256_file(output),
            }
        )
    write_json(
        PROJECT_ROOT / "artifact_manifest.json", {"artifacts": rows, "count": len(rows)}
    )


def main() -> None:
    sns.set_theme(style="whitegrid", palette="colorblind")
    figure_02()
    figure_03()
    figure_04()
    figure_05()
    appendix_figures()
    figure_a08_adapter_comparisons()
    tables()
    manifest()
    print("rendered %d artifacts" % len(ARTIFACTS))


if __name__ == "__main__":
    main()
