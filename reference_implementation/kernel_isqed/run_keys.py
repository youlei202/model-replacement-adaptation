from __future__ import annotations

import secrets
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .contracts import (
    PEERS,
    PREFLIGHT_GENERATIONS,
    RESPONSE_MODELS,
    ROLE_ROTATION_TARGETS,
    TARGETS,
    TASKS,
)
from .controlled import pedersen_commit, world_opening_path
from .data import select_preflight_items
from .io import CANONICAL_DIR, stable_hash, write_table


KEY_FAMILIES: Tuple[str, ...] = (
    "MODEL_MAIN",
    "MODEL_ROBUSTNESS",
    "PREFLIGHT",
    "EXP1_CLONE",
    "EXP2_SELECTOR",
    "EXP3_TRUTH",
    "EXP4_PLANTED",
    "EXP5_NATIVE",
    "EXP6_INVERSE",
    "EXP7_DISCLOSURE",
    "EXP8_SWAP",
    "EXP9_FRONTIER",
    "EXP10_CONTROLS",
    "EXP11_GENERALIZATION",
    "EXP12_GAUGE",
    "EXP13_ROSTER_REPLICATION",
    "EXP14_ROLE_ROTATION",
    "RFF_STABILITY",
    "RUNTIME_STATE",
)


def audit_cells() -> Tuple[Tuple[str, str, str], ...]:
    return tuple(
        (
            "%s__%s" % (target.model_id.replace("/", "--"), task.task_id),
            target.model_id,
            task.task_id,
        )
        for target in TARGETS
        for task in TASKS
    )


def role_rotation_audit_cells() -> Tuple[Tuple[str, str, str], ...]:
    return tuple(
        (
            "%s__%s" % (target.model_id.replace("/", "--"), task.task_id),
            target.model_id,
            task.task_id,
        )
        for target in ROLE_ROTATION_TARGETS
        for task in TASKS
    )


def experiment_keys(grid: Tuple[Tuple[str, str, str], ...]) -> List[Dict[str, object]]:
    from experiments.exp1_exact_peer_clone import expected_run_keys as exp1
    from experiments.exp2_convex_selector_worlds import expected_run_keys as exp2
    from experiments.exp3_random_truth_table_worlds import expected_run_keys as exp3
    from experiments.exp4_planted_bit_llm_geometry import expected_run_keys as exp4
    from experiments.exp5_llm_adapter_sensitivity import expected_run_keys as exp5
    from experiments.exp6_krr_inverse_reconstruction import expected_run_keys as exp6
    from experiments.exp7_quantized_state_disclosure import expected_run_keys as exp7
    from experiments.exp8_target_state_swap import expected_run_keys as exp8
    from experiments.exp9_error_state_query_frontier import expected_run_keys as exp9
    from experiments.exp10_peer_source_controls import expected_run_keys as exp10
    from experiments.exp11_fit_size_generalization import expected_run_keys as exp11
    from experiments.exp12_kernel_scale_gauge import expected_run_keys as exp12
    from experiments.exp13_peer_roster_replication import expected_run_keys as exp13
    from experiments.exp14_target_role_replication import expected_run_keys as exp14

    rows = exp1() + exp2() + exp3() + exp8()
    for builder in (exp4, exp5, exp6, exp7, exp9, exp10, exp11, exp12, exp13):
        rows.extend(builder(grid))
    rows.extend(exp14(role_rotation_audit_cells()))
    return rows


def _key_id(family: str, values: Dict[str, object]) -> str:
    return stable_hash([family, [(key, values[key]) for key in sorted(values)]])


def _opening_path(output_path: Path) -> Path:
    if output_path == CANONICAL_DIR / "expected_run_keys.parquet":
        return world_opening_path()
    return output_path.with_name(output_path.stem + "_openings.parquet")


def build_expected_run_keys(
    split_manifest: pd.DataFrame,
    state_manifest: Optional[pd.DataFrame] = None,
    output_path: Path = CANONICAL_DIR / "expected_run_keys.parquet",
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add(family: str, **values: object) -> None:
        row = {"key_family": family, **values}
        row["key_id"] = _key_id(family, values)
        rows.append(row)

    models = tuple(RESPONSE_MODELS)
    eligible = split_manifest[
        split_manifest["split"].isin(("fit", "validation", "iid_test", "ood_test"))
    ].drop_duplicates(["task_id", "item_id"])
    for model in models:
        for item in eligible.itertuples(index=False):
            add(
                "MODEL_MAIN",
                model_id=model.model_id,
                task_id=item.task_id,
                item_id=item.item_id,
                split=item.split,
                view_id="canonical",
            )

    robustness_views = {"mmlu_pro": 3, "hellaswag": 3, "rewardbench": 1}
    for task in TASKS:
        selected = eligible[eligible["task_id"] == task.task_id].copy()
        selected["order"] = selected["item_id"].map(
            lambda value: stable_hash([task.task_id, value, "robustness"])
        )
        selected = selected.sort_values("order").head(512)
        for model in PEERS + TARGETS:
            for item in selected.itertuples(index=False):
                for view_index in range(robustness_views[task.task_id]):
                    add(
                        "MODEL_ROBUSTNESS",
                        model_id=model.model_id,
                        task_id=task.task_id,
                        item_id=item.item_id,
                        split=item.split,
                        view_id="permutation_%d" % (view_index + 1),
                    )

    tokenizer_audit_path = CANONICAL_DIR / "tokenizer_length_audit.parquet"
    if not tokenizer_audit_path.is_file():
        raise RuntimeError("expected keys require the frozen tokenizer-length audit")
    tokenizer_audit = pd.read_parquet(tokenizer_audit_path)
    preflight_items = select_preflight_items(split_manifest, tokenizer_audit)
    generation_quotas = {
        TASKS[0].task_id: 11,
        TASKS[1].task_id: 11,
        TASKS[2].task_id: 10,
    }
    if sum(generation_quotas.values()) != PREFLIGHT_GENERATIONS:
        raise RuntimeError("generation-sanity quotas do not match the protocol")
    sanity_items = []
    for task in TASKS:
        part = preflight_items[preflight_items["task_id"] == task.task_id].copy()
        part["generation_order"] = part["item_id"].map(
            lambda value: stable_hash([task.task_id, value, "preflight_generation"])
        )
        sanity_items.extend(
            part.sort_values("generation_order")
            .head(generation_quotas[task.task_id])
            .to_dict("records")
        )
    for model in models:
        for item in preflight_items.to_dict("records"):
            add(
                "PREFLIGHT",
                check="stratified_item",
                model_id=model.model_id,
                task_id=item["task_id"],
                item_id=item["item_id"],
                length_bin=int(item["length_bin"]),
                tail_anchor=bool(item["tail_anchor"]),
            )
        for draw_id, item in enumerate(sanity_items):
            add(
                "PREFLIGHT",
                check="sanity_generation",
                model_id=model.model_id,
                task_id=item["task_id"],
                item_id=item["item_id"],
                draw_id=draw_id,
            )
    for registered in experiment_keys(audit_cells()):
        family = str(registered["key_family"])
        add(
            family,
            **{key: value for key, value in registered.items() if key != "key_family"}
        )

    for cell_id, target_id, task_id in audit_cells():
        for seed_index in range(1, 5):
            add(
                "RFF_STABILITY",
                audit_cell=cell_id,
                target_id=target_id,
                task_id=task_id,
                adapter="rff",
                seed_id="public_%d" % seed_index,
                q=3,
            )

    planned = list(rows)
    if state_manifest is not None and not state_manifest.empty:
        for state in state_manifest.itertuples(index=False):
            for measurement in ("encode", "decode", "adapter_latency"):
                add(
                    "RUNTIME_STATE",
                    runtime_kind="state_measurement",
                    state_key=getattr(state, "state_key", state.bitstream_digest),
                    bitstream_digest=state.bitstream_digest,
                    measurement=measurement,
                )
    for planned_row in planned:
        add(
            "RUNTIME_STATE",
            runtime_kind="failure_status",
            planned_family=planned_row["key_family"],
            planned_key_id=planned_row["key_id"],
        )

    frame = pd.DataFrame(rows)
    opening_path = _opening_path(output_path)
    opening_columns = (
        "commitment_scope",
        "seed_hex",
        "commitment_id",
        "commitment_value",
        "blinding",
    )
    if opening_path.is_file():
        openings = pd.read_parquet(opening_path)
    else:
        openings = pd.DataFrame(columns=opening_columns)
    commitments = dict(
        zip(
            openings["commitment_scope"].astype(str),
            openings["commitment_id"].astype(str),
        )
    )
    opening_rows = openings.to_dict("records")
    scopes = frame["commitment_scope"].dropna().astype(str).unique()
    for scope in scopes:
        if scope not in commitments:
            seed = secrets.randbits(256)
            commitment = pedersen_commit(seed)
            commitments[scope] = commitment.commitment_id
            opening_rows.append(
                {
                    "commitment_scope": scope,
                    "seed_hex": "%064x" % seed,
                    "commitment_id": commitment.commitment_id,
                    "commitment_value": str(commitment.value),
                    "blinding": str(commitment.blinding),
                }
            )
    write_table(pd.DataFrame(opening_rows, columns=opening_columns), opening_path)
    frame["commitment_id"] = frame["commitment_scope"].map(commitments)
    frame = frame.sort_values(["key_family", "key_id"]).reset_index(drop=True)
    write_table(frame, output_path)
    return frame
