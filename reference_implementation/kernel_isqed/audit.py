from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .adapters import AdapterContext, make_context
from .contracts import GLOBAL_SEED, ModelContract, PEERS, TARGETS, TASKS
from .response import coordinates_to_probabilities, total_variation


SPLITS = ("fit", "validation", "iid_test", "ood_test")


@dataclass
class AuditCell:
    target_id: str
    task_id: str
    item_ids: Dict[str, np.ndarray]
    peer_coordinates: Dict[str, np.ndarray]
    target_coordinates: Dict[str, np.ndarray]
    input_text: Optional[Dict[str, np.ndarray]] = None
    item_clusters: Optional[Dict[str, np.ndarray]] = None
    domains: Optional[Dict[str, np.ndarray]] = None
    gold: Optional[Dict[str, np.ndarray]] = None

    @property
    def cell_id(self) -> str:
        return "%s__%s" % (self.target_id.replace("/", "--"), self.task_id)

    @property
    def output_dimension(self) -> int:
        return int(self.target_coordinates["fit"].shape[1])

    def context(self, peer_indices: Sequence[int]) -> AdapterContext:
        return make_context(self.peer_coordinates["fit"], peer_indices)


def _coordinate_columns(frame: pd.DataFrame) -> List[str]:
    columns = [column for column in frame.columns if column.startswith("coord_")]
    return sorted(columns, key=lambda value: int(value.split("_")[1]))


def load_audit_cells(
    path: Path,
    required_splits: Sequence[str] = SPLITS,
    peer_contracts: Sequence[ModelContract] = PEERS,
    target_contracts: Sequence[ModelContract] = TARGETS,
) -> List[AuditCell]:
    frame = pd.read_parquet(path)
    required = {"model_id", "role", "task_id", "item_id", "split", "item_cluster", "domain", "gold"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("matched response table is missing %s" % sorted(missing))
    coordinate_columns = _coordinate_columns(frame)
    if not coordinate_columns:
        raise ValueError("matched response table has no coord_N columns")
    peer_ids = [contract.model_id for contract in peer_contracts]
    if len(peer_ids) != 3 or len(set(peer_ids)) != 3:
        raise ValueError("an audit roster must contain three distinct peers")
    target_ids = [contract.model_id for contract in target_contracts]
    if len(target_ids) != 3 or len(set(target_ids)) != 3:
        raise ValueError("an audit target population must contain three distinct models")
    overlap = set(peer_ids).intersection(target_ids)
    if overlap:
        raise ValueError("target and peer roles must be disjoint: %s" % sorted(overlap))
    cells = []
    for target_id in target_ids:
        for task in TASKS:
            item_ids: Dict[str, np.ndarray] = {}
            peer_coordinates: Dict[str, np.ndarray] = {}
            target_coordinates: Dict[str, np.ndarray] = {}
            input_text: Dict[str, np.ndarray] = {}
            item_clusters: Dict[str, np.ndarray] = {}
            domains: Dict[str, np.ndarray] = {}
            gold: Dict[str, np.ndarray] = {}
            for split in required_splits:
                selected = frame[(frame["task_id"] == task.task_id) & (frame["split"] == split)]
                target = selected[selected["model_id"] == target_id].sort_values("item_id")
                if target.empty:
                    raise ValueError("no %s rows for %s/%s" % (split, target_id, task.task_id))
                ids = target["item_id"].astype(str).to_numpy()
                peers = []
                for peer_id in peer_ids:
                    peer = selected[selected["model_id"] == peer_id].sort_values("item_id")
                    if peer["item_id"].astype(str).tolist() != ids.tolist():
                        raise ValueError("peer and target rows are not aligned")
                    peers.append(peer[coordinate_columns[: task.option_count - 1]].to_numpy(dtype=np.float64))
                item_ids[split] = ids
                peer_coordinates[split] = np.stack(peers, axis=1)
                target_coordinates[split] = target[coordinate_columns[: task.option_count - 1]].to_numpy(
                    dtype=np.float64
                )
                item_clusters[split] = target["item_cluster"].astype(str).to_numpy()
                domains[split] = target["domain"].astype(str).to_numpy()
                gold[split] = target["gold"].to_numpy(dtype=int)
                if "prompt_text" in target.columns:
                    input_text[split] = target["prompt_text"].astype(str).to_numpy()
                else:
                    input_text[split] = ids.copy()
            cells.append(
                AuditCell(
                    target_id,
                    task.task_id,
                    item_ids,
                    peer_coordinates,
                    target_coordinates,
                    input_text,
                    item_clusters,
                    domains,
                    gold,
                )
            )
    return cells


def get_cells(path: Path) -> List[AuditCell]:
    if not path.is_file():
        raise FileNotFoundError("matched response cache not found: %s" % path)
    return load_audit_cells(path)


def split_tv(cell: AuditCell, split: str, predicted_coordinates: np.ndarray) -> np.ndarray:
    actual = coordinates_to_probabilities(cell.target_coordinates[split])
    predicted = coordinates_to_probabilities(predicted_coordinates)
    return total_variation(actual, predicted)


def paired_bootstrap_interval(
    left: np.ndarray,
    right: Optional[np.ndarray] = None,
    draws: int = 2000,
    seed: int = GLOBAL_SEED,
    clusters: Optional[np.ndarray] = None,
    blocks: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    left_values = np.asarray(left, dtype=np.float64)
    values = left_values if right is None else left_values - np.asarray(right, dtype=np.float64)
    if clusters is not None:
        frame = pd.DataFrame(
            {
                "value": values,
                "cluster": np.asarray(clusters).astype(str),
            }
        )
        if blocks is not None:
            frame["block"] = np.asarray(blocks).astype(str)
            values = (
                frame.groupby("cluster", as_index=False)
                .agg(value=("value", "mean"), block=("block", "first"))
                .groupby("block", as_index=False)["value"]
                .mean()["value"]
                .to_numpy(dtype=np.float64)
            )
        else:
            values = (
                frame.groupby("cluster", as_index=False)["value"]
                .mean()["value"]
                .to_numpy(dtype=np.float64)
            )
    elif blocks is not None:
        values = (
            pd.DataFrame({"value": values, "block": np.asarray(blocks).astype(str)})
            .groupby("block", as_index=False)["value"]
            .mean()["value"]
            .to_numpy(dtype=np.float64)
        )
    if not len(values):
        raise ValueError("paired bootstrap received no sampling units")
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(values), size=(draws, len(values)))
    estimates = values[indices].mean(axis=1)
    return float(values.mean()), float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))
