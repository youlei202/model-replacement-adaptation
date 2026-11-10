from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_DIR = PROJECT_ROOT / "results" / ".freeze_checkpoint"

from experiments.exp7_quantized_state_disclosure import (
    registered_codecs,
    retained_ranks,
)
from experiments.exp8_target_state_swap import SWAP_ADAPTERS, neutral_state
from experiments.exp10_peer_source_controls import (
    ALTERNATIVE_CONDITIONS,
    CONTROL_ADAPTERS,
    PEER_CONDITIONS,
    STATE_CAP,
    alternative_feature_cell,
    shuffled_peer_cell,
)
from experiments.exp11_fit_size_generalization import fit_prefix_cell
from kernel_isqed.adapters import (
    AdapterState,
    fit_adapter,
    peer_subsets,
    select_native_configuration,
)
from kernel_isqed.audit import AuditCell, load_audit_cells
from kernel_isqed.codec import EncodedState, encode_state
from kernel_isqed.contracts import (
    ADAPTERS,
    BIT_CAPS,
    CROSS_ROLE_PEERS,
    CROSS_ROLE_ROSTER_ID,
    FIT_SIZES,
    GLOBAL_SEED,
    MATCHED_BUDGETS,
    PEERS,
    PROTOCOL_VERSION,
    REPLICATION_PEERS,
    REPLICATION_ROSTER_ID,
    REFERENCE_ROSTER_ID,
    ROLE_ROTATION_TARGETS,
    ROSTER_REPLICATION_ADAPTERS,
)
from kernel_isqed.deployment import (
    OperatingPoint,
    enumerate_operating_points,
    select_budget_point,
    select_under_budget,
)
from kernel_isqed.controls import fit_tfidf_state
from kernel_isqed.io import (
    CANONICAL_DIR,
    STATES_DIR,
    sha256_file,
    stable_hash,
    write_json,
    write_table,
)
from kernel_isqed.frozen import fit_validation_response_hash, source_tree_hash
from kernel_isqed.planted import (
    CONFIG_SELECTION_RULE,
    DELTA_CALIBRATION_DRAWS,
    DELTA_CALIBRATION_MAX,
    DELTA_CALIBRATION_STEPS,
    DESCRIPTIVE_WORLD_COUNT,
    INFERENCE_RULE,
    MIN_RESPONSE_GAIN,
    PERSUASIVE_PRIMARY_POINTS,
    PLANTED_ADAPTERS,
    PLANTED_SCOPE_ID,
    PLANTED_BUDGETS,
    PLANTED_LAW,
    PLANTED_TV_STRENGTHS,
    PRIMARY_BUDGET,
    PRIMARY_TV_STRENGTH,
    PRIMARY_WORLD_COUNT,
    RECOVERY_RULE,
    SIMULTANEOUS_ALPHA,
    STRONG_BIT_ERROR_THRESHOLD,
    freeze_planted_geometry,
)
from kernel_isqed.run_keys import build_expected_run_keys


def _codec_rank(state, context) -> int:
    return len(context.fit_features) if state.adapter == "krr" else int(state.rank)


def _persist(
    rows: List[Dict[str, object]],
    cell: AuditCell,
    family: str,
    adapter: str,
    encoded: Optional[EncodedState],
    validation_tv: float,
    condition: str = "natural",
    feature_state_path: str = "",
    feature_state_sha256: str = "",
    **fields: object,
) -> None:
    row: Dict[str, object] = {
        "audit_cell": cell.cell_id,
        "target_id": cell.target_id,
        "task_id": cell.task_id,
        "state_family": family,
        "adapter": adapter,
        "condition": condition,
        "validation_tv": validation_tv,
        "status": "frozen" if encoded is not None else "no_feasible_configuration",
        "state_path": "",
        "bitstream_digest": "",
        "complete_coded_bits": float("nan"),
        "codec": "",
        "rank": 0,
        "feature_state_path": feature_state_path,
        "feature_state_sha256": feature_state_sha256,
    }
    row.update(fields)
    if encoded is not None:
        path = (
            STATES_DIR
            / "frozen"
            / family
            / cell.cell_id
            / (encoded.bitstream_digest + ".bin")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded.bitstream)
        row.update(
            {
                "state_path": str(path.relative_to(PROJECT_ROOT)),
                "bitstream_digest": encoded.bitstream_digest,
                "complete_coded_bits": encoded.complete_bits,
                "codec": encoded.codec,
                "rank": encoded.rank,
            }
        )
    rows.append(row)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _checkpoint_path(cell_id: str) -> Path:
    return CHECKPOINT_DIR / (cell_id + ".json")


def _read_cell_checkpoint(cell_id: str) -> Optional[Dict[str, object]]:
    path = _checkpoint_path(cell_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("cell_id") != cell_id:
        return None
    return payload


def _write_cell_checkpoint(
    cell_id: str,
    rows: List[Dict[str, object]],
    completed_stages: List[str],
    done: bool,
    contract: Dict[str, str],
) -> None:
    _atomic_write_json(
        _checkpoint_path(cell_id),
        {
            "cell_id": cell_id,
            "contract": contract,
            "completed_stages": completed_stages,
            "rows": rows,
            "done": done,
        },
    )


def _map_ordered(executor, tasks: List[Tuple[object, Tuple[object, ...]]]) -> List[object]:
    """Run independent tasks and return results in submission order.

    ``tasks`` is a sequence of ``(function, arguments)`` pairs. Each function
    returns the rows it produced. Serial execution is used when ``executor`` is
    ``None`` or only one task exists, which keeps row order byte-for-byte
    identical to the single-process implementation.
    """
    if executor is None or len(tasks) <= 1:
        return [function(*arguments) for function, arguments in tasks]
    futures = [executor.submit(function, *arguments) for function, arguments in tasks]
    return [future.result() for future in futures]


def _frontier_subset_task(
    cell: AuditCell, adapter: str, q: int, subset: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], List[Dict[str, object]], Dict[int, Optional[Dict[str, object]]]]:
    rows: List[Dict[str, object]] = []
    points = enumerate_operating_points(cell, adapter, subset, False)
    selected_by_cap: Dict[int, Optional[Dict[str, object]]] = {}
    for cap in BIT_CAPS:
        point = select_under_budget(points, cap)
        if point is None:
            selected_by_cap[cap] = None
            _persist(
                rows,
                cell,
                "frontier_subset",
                adapter,
                None,
                float("nan"),
                q=q,
                peer_indices="|".join(str(value) for value in subset),
                budget_cap=cap,
                lambda_index=-1,
            )
            continue
        selected_by_cap[cap] = {
            "validation_error": point.validation_error,
            "complete_bits": point.encoded.complete_bits,
            "rank": point.decoded.rank,
            "lambda_index": point.state.lambda_index,
            "encoded": point.encoded,
        }
        _persist(
            rows,
            cell,
            "frontier_subset",
            adapter,
            point.encoded,
            point.validation_error,
            q=q,
            peer_indices="|".join(str(value) for value in subset),
            budget_cap=cap,
            lambda_index=point.state.lambda_index,
        )
    return subset, rows, selected_by_cap


def _freeze_frontier(
    rows: List[Dict[str, object]], cell: AuditCell, executor
) -> None:
    task_specs: List[Tuple[int, str, Tuple[int, ...]]] = []
    for q in (1, 2, 3):
        for adapter in ADAPTERS:
            for subset in peer_subsets(q):
                task_specs.append((q, adapter, subset))
    tasks = [
        (_frontier_subset_task, (cell, adapter, q, subset))
        for q, adapter, subset in task_specs
    ]
    selected_by_cap_by_group: Dict[
        Tuple[int, str],
        Dict[Tuple[int, ...], Dict[int, Optional[Dict[str, object]]]],
    ] = {}
    for (q, adapter, subset), (_, subset_rows, selected_by_cap) in zip(
        task_specs, _map_ordered(executor, tasks)
    ):
        rows.extend(subset_rows)
        selected_by_cap_by_group.setdefault((q, adapter), {})[subset] = selected_by_cap
    for (q, adapter), selected_by_cap_by_subset in selected_by_cap_by_group.items():
        for cap in BIT_CAPS:
            candidates = []
            for subset, selected_by_cap in selected_by_cap_by_subset.items():
                summary = selected_by_cap[cap]
                if summary is not None:
                    candidates.append((summary, subset))
            selected = (
                min(
                    candidates,
                    key=lambda value: (
                        value[0]["validation_error"],
                        value[0]["complete_bits"],
                        value[0]["rank"],
                        value[1],
                    ),
                )
                if candidates
                else None
            )
            summary, subset = selected if selected is not None else (None, ())
            _persist(
                rows,
                cell,
                "frontier_selected",
                adapter,
                summary["encoded"] if summary is not None else None,
                summary["validation_error"] if summary is not None else float("nan"),
                q=q,
                peer_indices="|".join(str(value) for value in subset),
                budget_cap=cap,
                lambda_index=summary["lambda_index"] if summary is not None else -1,
            )


def _native_adapter_task(
    cell: AuditCell, adapter: str
) -> Tuple[str, AdapterState, List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    context = cell.context((0, 1, 2))
    state, validation_tv = select_native_configuration(
        adapter,
        context,
        cell.peer_coordinates["fit"],
        cell.target_coordinates["fit"],
        cell.peer_coordinates["validation"],
        cell.target_coordinates["validation"],
    )
    native = encode_state(
        state, context, "native_fp64", _codec_rank(state, context)
    )
    _persist(
        rows,
        cell,
        "native",
        adapter,
        native,
        validation_tv,
        q=3,
        peer_indices="0|1|2",
        budget_cap="native",
        lambda_index=state.lambda_index,
    )
    for codec in registered_codecs(False):
        for rank in retained_ranks(state, context, False):
            try:
                encoded = encode_state(state, context, codec, rank)
            except ValueError:
                _persist(
                    rows,
                    cell,
                    "disclosure",
                    adapter,
                    None,
                    validation_tv,
                    q=3,
                    peer_indices="0|1|2",
                    retained_rank=rank,
                    lambda_index=state.lambda_index,
                )
                continue
            _persist(
                rows,
                cell,
                "disclosure",
                adapter,
                encoded,
                validation_tv,
                q=3,
                peer_indices="0|1|2",
                retained_rank=encoded.rank,
                lambda_index=state.lambda_index,
            )
    return adapter, state, rows


def _swap_adapter_task(
    cell: AuditCell, adapter: str, native_state: AdapterState
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    context = cell.context((0, 1, 2))
    state = native_state
    neutral = neutral_state(state)
    encoded = encode_state(
        neutral, context, "native_fp64", _codec_rank(neutral, context)
    )
    _persist(
        rows,
        cell,
        "swap_control",
        adapter,
        encoded,
        float("nan"),
        condition="neutral",
        q=3,
        peer_indices="0|1|2",
        lambda_index=state.lambda_index,
    )
    permutation = np.random.RandomState(GLOBAL_SEED).permutation(
        len(cell.target_coordinates["fit"])
    )
    shuffled = fit_adapter(
        adapter,
        context,
        cell.peer_coordinates["fit"],
        cell.target_coordinates["fit"][permutation],
        state.lambda_index,
        state.rank,
    )
    encoded = encode_state(
        shuffled, context, "native_fp64", _codec_rank(shuffled, context)
    )
    _persist(
        rows,
        cell,
        "swap_control",
        adapter,
        encoded,
        float("nan"),
        condition="row_shuffled_refit",
        q=3,
        peer_indices="0|1|2",
        lambda_index=state.lambda_index,
    )
    return rows


def _freeze_native_and_swaps(
    rows: List[Dict[str, object]], cell: AuditCell, executor
) -> None:
    native_tasks = [
        (_native_adapter_task, (cell, adapter)) for adapter in ADAPTERS
    ]
    native_states: Dict[str, AdapterState] = {}
    for adapter, state, adapter_rows in _map_ordered(executor, native_tasks):
        native_states[adapter] = state
        rows.extend(adapter_rows)
    swap_tasks = [
        (_swap_adapter_task, (cell, adapter, native_states[adapter]))
        for adapter in SWAP_ADAPTERS
    ]
    for adapter_rows in _map_ordered(executor, swap_tasks):
        rows.extend(adapter_rows)


def _persist_control(
    rows: List[Dict[str, object]],
    cell: AuditCell,
    adapter: str,
    condition: str,
    q: int,
    point: Optional[OperatingPoint],
    feature_state_path: str = "",
    feature_state_sha256: str = "",
) -> None:
    _persist(
        rows,
        cell,
        "peer_source_control",
        adapter,
        point.encoded if point is not None else None,
        point.validation_error if point is not None else float("nan"),
        condition=condition,
        feature_state_path=feature_state_path,
        feature_state_sha256=feature_state_sha256,
        q=q,
        peer_indices="0|1|2" if q == 3 else "0",
        budget_cap=STATE_CAP,
        lambda_index=point.state.lambda_index if point is not None else -1,
    )


def _control_task(
    cell: AuditCell,
    adapter: str,
    condition: str,
    tfidf_relative: str,
    tfidf_sha256: str,
    tfidf_state: Dict[str, object],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if condition in PEER_CONDITIONS:
        controlled = (
            cell if condition == "actual" else shuffled_peer_cell(cell, condition)
        )
        point = select_budget_point(controlled, adapter, (0, 1, 2), STATE_CAP, False)
        _persist_control(rows, cell, adapter, condition, 3, point)
    else:
        feature_state = tfidf_state if condition == "input_tfidf" else None
        controlled = alternative_feature_cell(cell, condition, feature_state)
        point = select_budget_point(controlled, adapter, (0,), STATE_CAP, False)
        _persist_control(
            rows,
            cell,
            adapter,
            condition,
            0,
            point,
            feature_state_path=tfidf_relative if feature_state is not None else "",
            feature_state_sha256=tfidf_sha256 if feature_state is not None else "",
        )
    return rows


def _freeze_controls(
    rows: List[Dict[str, object]], cell: AuditCell, executor
) -> None:
    text = cell.input_text if cell.input_text is not None else cell.item_ids
    tfidf_state = fit_tfidf_state(text["fit"])
    tfidf_path = STATES_DIR / "frozen/public_features" / (cell.cell_id + "_tfidf.json")
    write_json(tfidf_path, tfidf_state)
    tfidf_relative = str(tfidf_path.relative_to(PROJECT_ROOT))
    tfidf_sha256 = sha256_file(tfidf_path)
    tasks = []
    for adapter in CONTROL_ADAPTERS:
        for condition in PEER_CONDITIONS + ALTERNATIVE_CONDITIONS:
            tasks.append(
                (
                    _control_task,
                    (
                        cell,
                        adapter,
                        condition,
                        tfidf_relative,
                        tfidf_sha256,
                        tfidf_state,
                    ),
                )
            )
    for adapter_rows in _map_ordered(executor, tasks):
        rows.extend(adapter_rows)


def _fit_size_task(
    cell: AuditCell, label: str, size: int, adapter: str
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    prefix = fit_prefix_cell(cell, size)
    context = prefix.context((0, 1, 2))
    state, validation_tv = select_native_configuration(
        adapter,
        context,
        prefix.peer_coordinates["fit"],
        prefix.target_coordinates["fit"],
        prefix.peer_coordinates["validation"],
        prefix.target_coordinates["validation"],
    )
    encoded = encode_state(
        state, context, "native_fp64", _codec_rank(state, context)
    )
    _persist(
        rows,
        cell,
        "fit_size",
        adapter,
        encoded,
        validation_tv,
        fit_size_label=label,
        fit_size=size,
        q=3,
        peer_indices="0|1|2",
        lambda_index=state.lambda_index,
    )
    return rows


def _freeze_fit_sizes(
    rows: List[Dict[str, object]], cell: AuditCell, executor
) -> None:
    sizes = [(str(size), min(size, len(cell.item_ids["fit"]))) for size in FIT_SIZES]
    sizes.append(("m_max", len(cell.item_ids["fit"])))
    tasks = []
    for label, size in sizes:
        for adapter in ADAPTERS:
            tasks.append((_fit_size_task, (cell, label, size, adapter)))
    for adapter_rows in _map_ordered(executor, tasks):
        rows.extend(adapter_rows)


def _roster_state_task(
    cell: AuditCell,
    adapter: str,
    roster_id: str,
    native_family: str,
    frontier_family: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    subset = (0, 1, 2)
    context = cell.context(subset)
    state, validation_tv = select_native_configuration(
        adapter,
        context,
        cell.peer_coordinates["fit"],
        cell.target_coordinates["fit"],
        cell.peer_coordinates["validation"],
        cell.target_coordinates["validation"],
    )
    native = encode_state(state, context, "native_fp64", _codec_rank(state, context))
    _persist(
        rows,
        cell,
        native_family,
        adapter,
        native,
        validation_tv,
        roster_id=roster_id,
        q=3,
        peer_indices="0|1|2",
        budget_cap="native",
        lambda_index=state.lambda_index,
    )
    points = enumerate_operating_points(cell, adapter, subset, False)
    for budget in MATCHED_BUDGETS:
        point = select_under_budget(points, budget)
        _persist(
            rows,
            cell,
            frontier_family,
            adapter,
            point.encoded if point is not None else None,
            point.validation_error if point is not None else float("nan"),
            roster_id=roster_id,
            q=3,
            peer_indices="0|1|2",
            budget_cap=budget,
            lambda_index=(point.state.lambda_index if point is not None else -1),
        )
    return rows


def _freeze_roster_states(
    rows: List[Dict[str, object]],
    cells: List[AuditCell],
    executor,
    roster_id: str,
    native_family: str,
    frontier_family: str,
) -> None:
    tasks = [
        (
            _roster_state_task,
            (cell, adapter, roster_id, native_family, frontier_family),
        )
        for cell in cells
        for adapter in ROSTER_REPLICATION_ADAPTERS
    ]
    for state_rows in _map_ordered(executor, tasks):
        rows.extend(state_rows)


STAGE_FUNCTIONS = (
    ("frontier", _freeze_frontier),
    ("native_and_swaps", _freeze_native_and_swaps),
    ("controls", _freeze_controls),
    ("fit_sizes", _freeze_fit_sizes),
)


def _freeze_cell(
    cell: AuditCell, contract: Dict[str, str], executor=None
) -> List[Dict[str, object]]:
    checkpoint = _read_cell_checkpoint(cell.cell_id)
    if checkpoint is not None and checkpoint.get("contract") == contract:
        rows = list(checkpoint.get("rows", []))
        completed_stages = [
            str(stage)
            for stage in checkpoint.get("completed_stages", [])
            if str(stage) in {name for name, _ in STAGE_FUNCTIONS}
        ]
        if checkpoint.get("done") and len(completed_stages) == len(STAGE_FUNCTIONS):
            print("[freeze:%s] checkpoint complete; skipping" % cell.cell_id, flush=True)
            return rows
    else:
        if checkpoint is not None:
            print(
                "[freeze:%s] ignoring checkpoint from another source/data revision"
                % cell.cell_id,
                flush=True,
            )
        rows = []
        completed_stages = []

    completed = set(completed_stages)
    for stage_name, stage_function in STAGE_FUNCTIONS:
        if stage_name in completed:
            continue
        print("[freeze:%s] stage %s" % (cell.cell_id, stage_name), flush=True)
        stage_function(rows, cell, executor)
        completed.add(stage_name)
        ordered_completed = [
            name for name, _ in STAGE_FUNCTIONS if name in completed
        ]
        _write_cell_checkpoint(
            cell.cell_id, rows, ordered_completed, False, contract
        )
    _write_cell_checkpoint(
        cell.cell_id,
        rows,
        [name for name, _ in STAGE_FUNCTIONS],
        True,
        contract,
    )
    return rows


def _freeze_cell_worker(payload: object) -> List[Dict[str, object]]:
    _, cell, contract = payload
    return _freeze_cell(cell, contract)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze every target-derived state before the one-time test unlock"
    )
    parser.add_argument(
        "--responses", type=Path, default=CANONICAL_DIR / "matched_responses.parquet"
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--task-workers", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least one")
    if args.task_workers < 1:
        parser.error("--task-workers must be at least one")
    if args.jobs > 1 and args.task_workers > 1:
        parser.error(
            "use either --jobs (across cells) or --task-workers (within a cell), "
            "not both"
        )
    freeze_path = CANONICAL_DIR / "state_freeze.json"
    if freeze_path.is_file():
        raise RuntimeError(
            "states are already frozen; change the protocol version before replacing them"
        )
    observed_splits = set(
        pd.read_parquet(args.responses, columns=["split"])["split"].astype(str)
    )
    if observed_splits != {"fit", "validation"}:
        raise RuntimeError("state freezing accepts fit and validation responses only")
    cells = load_audit_cells(args.responses, required_splits=("fit", "validation"))
    replication_cells = load_audit_cells(
        args.responses,
        required_splits=("fit", "validation"),
        peer_contracts=REPLICATION_PEERS,
    )
    role_reference_cells = load_audit_cells(
        args.responses,
        required_splits=("fit", "validation"),
        peer_contracts=PEERS,
        target_contracts=ROLE_ROTATION_TARGETS,
    )
    role_cross_cells = load_audit_cells(
        args.responses,
        required_splits=("fit", "validation"),
        peer_contracts=CROSS_ROLE_PEERS,
        target_contracts=ROLE_ROTATION_TARGETS,
    )
    split_manifest = pd.read_parquet(CANONICAL_DIR / "split_manifest.parquet")
    build_expected_run_keys(split_manifest)
    source_hash = source_tree_hash()
    responses_hash = fit_validation_response_hash(args.responses)
    checkpoint_contract = {
        "source_tree_hash": source_hash,
        "fit_validation_responses_sha256": responses_hash,
    }
    rows: List[Dict[str, object]] = []
    task_executor = None
    if args.task_workers > 1:
        task_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.task_workers
        )
    try:
        if args.jobs > 1:
            print(
                "[freeze] processing %d cells with %d workers"
                % (len(cells), args.jobs),
                flush=True,
            )
            tasks = [
                (index, cell, checkpoint_contract)
                for index, cell in enumerate(cells, 1)
            ]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.jobs
            ) as executor:
                for _, cell_rows in executor.map(_freeze_cell_worker, tasks):
                    rows.extend(cell_rows)
        else:
            for index, cell in enumerate(cells, 1):
                print(
                    "[freeze] cell %d/%d %s" % (index, len(cells), cell.cell_id),
                    flush=True,
                )
                rows.extend(
                    _freeze_cell(cell, checkpoint_contract, task_executor)
                )
        print(
            "[freeze] independent peer-roster replication states",
            flush=True,
        )
        _freeze_roster_states(
            rows,
            replication_cells,
            task_executor,
            REPLICATION_ROSTER_ID,
            "roster_native",
            "roster_frontier",
        )
        print("[freeze] target-role replication states", flush=True)
        for roster_id, role_cells in (
            (REFERENCE_ROSTER_ID, role_reference_cells),
            (CROSS_ROLE_ROSTER_ID, role_cross_cells),
        ):
            _freeze_roster_states(
                rows,
                role_cells,
                task_executor,
                roster_id,
                "role_rotation_native",
                "role_rotation_frontier",
            )
    finally:
        if task_executor is not None:
            task_executor.shutdown()
    planted_results_path = CANONICAL_DIR / "frozen_planted_results.parquet"
    planted_oracle_path = (
        CANONICAL_DIR.parent / "oracle_truth/frozen_planted_openings.parquet"
    )
    planted_manifest_path = CANONICAL_DIR / "frozen_planted_manifest.json"
    planted_contract = {
        "protocol": PROTOCOL_VERSION,
        "world_law_scope": PLANTED_SCOPE_ID,
        "planted_law": PLANTED_LAW,
        "minimum_response_gain": MIN_RESPONSE_GAIN,
        "adapters": list(PLANTED_ADAPTERS),
        "tv_strengths": list(PLANTED_TV_STRENGTHS),
        "budgets": list(PLANTED_BUDGETS),
        "primary_tv_strength": PRIMARY_TV_STRENGTH,
        "primary_budget": PRIMARY_BUDGET,
        "primary_world_count": PRIMARY_WORLD_COUNT,
        "descriptive_world_count": DESCRIPTIVE_WORLD_COUNT,
        "primary_family_size": 27,
        "calibration_scope": (
            "8 deterministic nonconfirmatory probes per "
            "cell-adapter-budget-strength"
        ),
        "confirmatory_scope": (
            "9 cells x 3 adapters at semantic-TV .25 and 256 KiBit"
        ),
        "world_draws": (
            "fresh scope-specific openings per "
            "cell-adapter-strength-budget-draw"
        ),
        "simultaneous_alpha": SIMULTANEOUS_ALPHA,
        "inference_rule": INFERENCE_RULE,
        "calibration_draws": DELTA_CALIBRATION_DRAWS,
        "calibration_steps": DELTA_CALIBRATION_STEPS,
        "calibration_max_delta": DELTA_CALIBRATION_MAX,
        "configuration_rule": CONFIG_SELECTION_RULE,
        "recovery_rule": RECOVERY_RULE,
        "success_standard": {
            "minimum_point": (
                "bit_error_upper_bound < 0.5 and "
                "information_lower_bound_per_bit > 0"
            ),
            "persuasive_family": "%d or more of 27 primary points pass minimum"
            % PERSUASIVE_PRIMARY_POINTS,
            "strong_point_bit_error_at_most": STRONG_BIT_ERROR_THRESHOLD,
        },
        "source_tree_hash": source_hash,
        "fit_validation_responses_sha256": responses_hash,
        "expected_run_keys_sha256": sha256_file(
            CANONICAL_DIR / "expected_run_keys.parquet"
        ),
    }
    existing_planted = (
        planted_results_path.is_file(),
        planted_oracle_path.is_file(),
        planted_manifest_path.is_file(),
    )
    if any(existing_planted) and not all(existing_planted):
        raise RuntimeError(
            "partial planted freeze exists; archive it before starting the new protocol"
        )
    if all(existing_planted):
        planted_manifest = json.loads(
            planted_manifest_path.read_text(encoding="utf-8")
        )
        if any(
            planted_manifest.get(key) != value
            for key, value in planted_contract.items()
        ):
            raise RuntimeError(
                "existing planted freeze belongs to another protocol; archive it "
                "instead of reusing or overwriting it"
            )
        if (
            planted_manifest.get("results_sha256")
            != sha256_file(planted_results_path)
            or planted_manifest.get("oracle_sha256")
            != sha256_file(planted_oracle_path)
        ):
            raise RuntimeError("existing planted freeze failed its digest contract")
        planted_results = pd.read_parquet(planted_results_path)
        planted_oracle = pd.read_parquet(planted_oracle_path)
    else:
        planted_results, planted_oracle = freeze_planted_geometry(
            cells, workers=max(args.jobs, args.task_workers)
        )
        planted_results_path = write_table(planted_results, planted_results_path)
        planted_oracle_path = write_table(planted_oracle, planted_oracle_path)
        write_json(
            planted_manifest_path,
            {
                **planted_contract,
                "results_sha256": sha256_file(planted_results_path),
                "oracle_sha256": sha256_file(planted_oracle_path),
            },
        )
    manifest = pd.DataFrame(rows)
    manifest["budget_cap"] = manifest["budget_cap"].astype(str)
    manifest["state_key"] = manifest.apply(
        lambda row: stable_hash(
            {
                key: row[key]
                for key in sorted(manifest.columns)
                if key != "validation_tv"
            }
        ),
        axis=1,
    )
    if manifest.duplicated("state_key").any():
        raise RuntimeError("frozen-state manifest contains duplicate semantic keys")
    manifest_path = write_table(manifest, CANONICAL_DIR / "serialized_states.parquet")
    expected_keys = build_expected_run_keys(split_manifest, manifest)
    requirements = PROJECT_ROOT / "requirements.txt"
    write_json(
        freeze_path,
        {
            "protocol": PROTOCOL_VERSION,
            "state_count": len(manifest),
            "state_families": manifest.groupby("state_family")
            .size()
            .astype(int)
            .to_dict(),
            "serialized_states_sha256": sha256_file(manifest_path),
            "expected_run_key_count": len(expected_keys),
            "expected_run_keys_sha256": sha256_file(
                CANONICAL_DIR / "expected_run_keys.parquet"
            ),
            "protocol_lock_sha256": sha256_file(CANONICAL_DIR / "protocol_lock.json"),
            "split_manifest_sha256": sha256_file(
                CANONICAL_DIR / "split_manifest.parquet"
            ),
            "split_contract_sha256": sha256_file(
                CANONICAL_DIR / "split_contract.parquet"
            ),
            "prompt_manifest_sha256": sha256_file(
                CANONICAL_DIR / "prompt_manifest.parquet"
            ),
            "tokenizer_length_audit_sha256": sha256_file(
                CANONICAL_DIR / "tokenizer_length_audit.parquet"
            ),
            "fit_validation_responses_sha256": responses_hash,
            "requirements_sha256": sha256_file(requirements),
            "python_version_file_sha256": sha256_file(PROJECT_ROOT / ".python-version"),
            "pyproject_sha256": sha256_file(PROJECT_ROOT / "pyproject.toml"),
            "source_tree_hash": source_hash,
            "frozen_planted_worlds": int(
                (planted_results["row_kind"] == "world").sum()
            ),
            "frozen_planted_results_sha256": sha256_file(planted_results_path),
            "frozen_planted_oracle_sha256": sha256_file(planted_oracle_path),
            "frozen_planted_manifest_sha256": sha256_file(
                planted_manifest_path
            ),
            "state_freeze_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "test_unlocks": 0,
        },
    )
    print(manifest.groupby(["state_family", "adapter"]).size().rename("states"))
    print("frozen:", freeze_path)


if __name__ == "__main__":
    main()
