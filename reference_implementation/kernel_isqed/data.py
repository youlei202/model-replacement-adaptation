from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

from .contracts import (
    GLOBAL_SEED,
    MIN_DOMAIN_STRATUM,
    MIN_GOLD_STRATUM,
    PREFLIGHT_ITEMS,
    TASKS,
    TaskContract,
)
from .io import CANONICAL_DIR, stable_hash, write_table
from .response import SemanticItem, canonical_permutation, render_prompt


class OptionCountMismatch(ValueError):
    def __init__(self, task_id: str, observed: int, expected: int):
        super().__init__("%s item has %d options" % (task_id, observed))
        self.task_id = task_id
        self.observed = observed
        self.expected = expected


def normalize_text(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value).strip().lower())
    return re.sub(r"[^\w\s]", "", value)


def _conversation_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for message in value:
            if isinstance(message, dict):
                role = str(message.get("role", "user")).upper()
                content = str(message.get("content", ""))
                parts.append("%s: %s" % (role, content))
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return str(value)


def _candidate_response(value: object) -> str:
    if isinstance(value, list):
        for message in reversed(value):
            if (
                isinstance(message, dict)
                and str(message.get("role", "")).lower() == "assistant"
            ):
                return str(message.get("content", ""))
    return _conversation_text(value)


def _answer_index(value: object, option_count: int) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    if len(text) == 1 and text.upper().isalpha():
        return ord(text.upper()) - ord("A")
    raise ValueError("cannot parse answer index %r" % value)


def _semantic_item(
    task: TaskContract, row: Dict[str, object], row_index: int
) -> SemanticItem:
    if task.task_id == "mmlu_pro":
        context = str(row["question"])
        options = tuple(str(value) for value in row["options"])
        gold = _answer_index(row.get("answer", row.get("answer_index")), len(options))
    elif task.task_id == "hellaswag":
        context = str(
            row.get("ctx", "%s %s" % (row.get("ctx_a", ""), row.get("ctx_b", "")))
        )
        options = tuple(str(value) for value in row["endings"])
        gold = _answer_index(row["label"], len(options))
    else:
        context = _conversation_text(row.get("prompt", row.get("conversation", "")))
        chosen = _candidate_response(row["chosen"])
        rejected = _candidate_response(row["rejected"])
        raw_id = str(row.get("id", row_index))
        swap = (
            int(
                hashlib.sha256((raw_id + "|rewardbench").encode("utf-8")).hexdigest(),
                16,
            )
            % 2
        )
        options = (chosen, rejected) if swap == 0 else (rejected, chosen)
        gold = 0 if swap == 0 else 1
    if len(options) != task.option_count:
        raise OptionCountMismatch(task.task_id, len(options), task.option_count)
    if gold < 0 or gold >= len(options):
        raise ValueError("%s item has an out-of-range gold answer" % task.task_id)
    domain = str(row.get(task.domain_field, "unknown"))
    source_id = row.get("id", row.get("question_id", row_index))
    item_id = "%s:%s:%d" % (task.task_id, source_id, row_index)
    return SemanticItem(task.task_id, item_id, context, options, gold, domain)


def load_semantic_items(
    task: TaskContract, revision: Optional[str] = None
) -> Tuple[List[SemanticItem], pd.DataFrame]:
    config = None if task.dataset_config == "default" else task.dataset_config
    dataset = load_dataset(
        task.dataset_id, config, split=task.source_split, revision=revision
    )
    items = []
    rejected = []
    for index, row in enumerate(dataset):
        record = dict(row)
        try:
            items.append(_semantic_item(task, record, index))
        except OptionCountMismatch as error:
            if task.task_id != "mmlu_pro":
                raise
            rejected.append(
                {
                    "task_id": task.task_id,
                    "source_id": str(
                        record.get("id", record.get("question_id", index))
                    ),
                    "reason": "option_count_mismatch",
                    "observed_option_count": error.observed,
                    "required_option_count": error.expected,
                }
            )
    if not items:
        raise RuntimeError(
            "no contract-eligible items were loaded for %s" % task.task_id
        )
    return (
        items,
        pd.DataFrame(
            rejected,
            columns=(
                "task_id",
                "source_id",
                "reason",
                "observed_option_count",
                "required_option_count",
            ),
        ),
    )


def _shingles(text: str, width: int = 5) -> Set[str]:
    words = normalize_text(text).split()
    if len(words) < width:
        return {" ".join(words)}
    return {
        " ".join(words[index : index + width])
        for index in range(len(words) - width + 1)
    }


def _signature(shingles: Set[str], count: int = 64) -> Tuple[int, ...]:
    if not shingles:
        return tuple(0 for _ in range(count))
    values = []
    for seed in range(count):
        minimum = min(
            int.from_bytes(
                hashlib.blake2b(
                    (str(seed) + token).encode("utf-8"), digest_size=8
                ).digest(),
                "big",
            )
            for token in shingles
        )
        values.append(minimum)
    return tuple(values)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def duplicate_clusters(
    items: Sequence[SemanticItem], threshold: float = 0.85
) -> List[str]:
    union = _UnionFind(len(items))
    exact: Dict[str, int] = {}
    shingles = []
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
    for index, item in enumerate(items):
        text = item.context + "\n" + "\n".join(item.options)
        normalized = normalize_text(text)
        if normalized in exact:
            union.union(index, exact[normalized])
        else:
            exact[normalized] = index
        item_shingles = _shingles(text)
        shingles.append(item_shingles)
        signature = _signature(item_shingles)
        for band in range(8):
            key = (band, signature[band * 8 : (band + 1) * 8])
            for other in buckets[key]:
                intersection = len(item_shingles & shingles[other])
                total = len(item_shingles | shingles[other])
                if total and intersection / total >= threshold:
                    union.union(index, other)
            buckets[key].append(index)
    roots = [union.find(index) for index in range(len(items))]
    return [
        "cluster_%s" % stable_hash([items[root].task_id, roots[root]])[:16]
        for root in roots
    ]


def _domain_order(
    task: TaskContract, domains: Iterable[str], dataset_revision: str
) -> List[str]:
    values = []
    for domain in set(domains):
        key = "%s|%d|%s" % (dataset_revision, GLOBAL_SEED, domain)
        values.append((hashlib.sha256(key.encode("utf-8")).hexdigest(), domain))
    return [domain for _, domain in sorted(values)]


def candidate_frame(
    task: TaskContract, items: Sequence[SemanticItem], dataset_revision: str
) -> pd.DataFrame:
    clusters = duplicate_clusters(items)
    return pd.DataFrame(
        [
            {
                **asdict(item),
                "options": json.dumps(item.options, ensure_ascii=False),
                "item_cluster": cluster,
                "dataset_revision": dataset_revision,
            }
            for item, cluster in zip(items, clusters)
        ]
    )


def _validate_split(task: TaskContract, frame: pd.DataFrame) -> pd.DataFrame:
    active = frame[frame["split"].isin(("fit", "validation", "iid_test", "ood_test"))]
    cluster_splits = active.groupby("item_cluster")["split"].nunique()
    if (cluster_splits > 1).any():
        raise RuntimeError("item clusters cross split boundaries for %s" % task.task_id)
    in_domain = set(active[active["split"] != "ood_test"]["domain"].astype(str))
    ood_domain = set(active[active["split"] == "ood_test"]["domain"].astype(str))
    if in_domain.intersection(ood_domain):
        raise RuntimeError(
            "OOD domains leak into fit/validation/IID for %s" % task.task_id
        )
    in_domain_support = [
        set(active[active["split"] == split]["domain"].astype(str))
        for split in ("fit", "validation", "iid_test")
    ]
    if any(support != in_domain_support[0] for support in in_domain_support[1:]):
        raise RuntimeError(
            "fit/validation/IID do not share the same domain support for %s"
            % task.task_id
        )
    for split in ("fit", "validation", "iid_test", "ood_test"):
        selected = active[active["split"] == split]
        if selected.empty:
            raise RuntimeError("empty %s split for %s" % (split, task.task_id))
        observed_gold = set(selected["gold"].astype(int))
        if observed_gold != set(range(task.option_count)):
            raise RuntimeError("gold strata collapsed in %s/%s" % (task.task_id, split))
        if int(selected["gold"].value_counts().min()) < MIN_GOLD_STRATUM:
            raise RuntimeError(
                "gold stratum below the registered minimum in %s/%s"
                % (task.task_id, split)
            )
        if int(selected["domain"].value_counts().min()) < MIN_DOMAIN_STRATUM:
            raise RuntimeError(
                "domain stratum below the registered minimum in %s/%s"
                % (task.task_id, split)
            )
    return (
        active.groupby("split", as_index=False)
        .agg(
            actual_items=("item_id", "nunique"),
            actual_clusters=("item_cluster", "nunique"),
            actual_domains=("domain", "nunique"),
            minimum_gold_stratum=(
                "gold",
                lambda values: int(values.value_counts().min()),
            ),
        )
        .assign(task_id=task.task_id)
    )


def split_candidate_frame(
    task: TaskContract, candidates: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = candidates[candidates["within_tokenizer_intersection"].astype(bool)].copy()
    if base.empty:
        raise RuntimeError("tokenizer intersection is empty for %s" % task.task_id)
    revisions = set(base["dataset_revision"].astype(str))
    if len(revisions) != 1:
        raise RuntimeError("candidate pool mixes dataset revisions")
    dataset_revision = next(iter(revisions))
    cluster_consistency = base.groupby("item_cluster", as_index=False).agg(
        candidate_items=("item_id", "nunique"),
        domain_count=("domain", "nunique"),
        gold_count=("gold", "nunique"),
    )
    ambiguous_clusters = cluster_consistency[
        (cluster_consistency["domain_count"] > 1)
        | (cluster_consistency["gold_count"] > 1)
    ].copy()
    exclusion_audit = ambiguous_clusters.assign(
        task_id=task.task_id,
        reason="cluster_crosses_domain_or_gold",
        dataset_revision=dataset_revision,
    )[
        [
            "task_id",
            "item_cluster",
            "reason",
            "candidate_items",
            "domain_count",
            "gold_count",
            "dataset_revision",
        ]
    ]
    if not ambiguous_clusters.empty:
        base = base[
            ~base["item_cluster"].isin(ambiguous_clusters["item_cluster"])
        ].copy()
    if base.empty:
        raise RuntimeError(
            "no unambiguous tokenizer-eligible clusters remain for %s" % task.task_id
        )
    domain_counts = base.groupby("domain").size().to_dict()
    domain_cluster_counts = base.groupby("domain")["item_cluster"].nunique()
    in_domain_domains = set(
        domain_cluster_counts[
            domain_cluster_counts >= MIN_DOMAIN_STRATUM * 3
        ].index.astype(str)
    )
    required_in_domain = task.fit + task.validation + task.iid_test
    selected_domains = []
    ood_count = 0
    ood_candidates = [
        domain
        for domain in base["domain"].astype(str).unique()
        if int(domain_counts[domain]) >= MIN_DOMAIN_STRATUM
    ]
    remaining_domains = _domain_order(task, ood_candidates, dataset_revision)
    for domain in remaining_domains:
        proposed = selected_domains + [domain]
        remaining = base[
            base["domain"].astype(str).isin(in_domain_domains)
            & ~base["domain"].isin(proposed)
        ]
        gold_capacity = remaining.groupby("gold").size()
        feasible = (
            len(remaining) >= required_in_domain
            and set(gold_capacity.index.astype(int)) == set(range(task.option_count))
            and int(gold_capacity.min()) >= MIN_GOLD_STRATUM * 3
        )
        if not feasible:
            continue
        selected_domains.append(domain)
        ood_count += int(domain_counts[domain])
        ood = base[base["domain"].isin(selected_domains)]
        ood_gold = ood.groupby("gold").size()
        if (
            ood_count >= task.ood_test
            and set(ood_gold.index.astype(int)) == set(range(task.option_count))
            and int(ood_gold.min()) >= MIN_GOLD_STRATUM
        ):
            break
    if not selected_domains:
        raise RuntimeError(
            "no whole-domain OOD split is feasible for %s" % task.task_id
        )
    base = base[
        base["domain"].isin(selected_domains)
        | base["domain"].astype(str).isin(in_domain_domains)
    ].copy()
    base["split"] = "unassigned"
    base.loc[base["domain"].isin(selected_domains), "split"] = "ood_test"
    quotas = {"fit": task.fit, "validation": task.validation, "iid_test": task.iid_test}
    remaining = base[base["split"] == "unassigned"].copy()
    cluster_sizes = remaining.groupby("item_cluster").size().rename("cluster_size")
    cluster_rows = (
        remaining.groupby("item_cluster", as_index=False)
        .first()
        .merge(cluster_sizes, on="item_cluster", how="left")
    )
    cluster_rows["stratum"] = (
        cluster_rows["domain"].astype(str) + "|" + cluster_rows["gold"].astype(str)
    )
    cluster_rows["order"] = cluster_rows["item_cluster"].map(
        lambda value: stable_hash([value, GLOBAL_SEED, "in_domain"])
    )
    cluster_rows = cluster_rows.sort_values(["stratum", "order"])
    assignments: Dict[str, str] = {}
    counts = {key: 0 for key in quotas}
    domain_counts_by_split = defaultdict(int)
    gold_counts_by_split = defaultdict(int)
    split_order = ("validation", "iid_test", "fit")
    records = cluster_rows.to_dict("records")

    def assign(row: Dict[str, object], split: str) -> None:
        cluster = str(row["item_cluster"])
        size = int(row["cluster_size"])
        assignments[cluster] = split
        counts[split] += size
        domain_counts_by_split[(split, str(row["domain"]))] += size
        gold_counts_by_split[(split, int(row["gold"]))] += size

    for domain in sorted(cluster_rows["domain"].astype(str).unique()):
        for split in split_order:
            while domain_counts_by_split[(split, domain)] < MIN_DOMAIN_STRATUM:
                candidates = [
                    row
                    for row in records
                    if str(row["item_cluster"]) not in assignments
                    and str(row["domain"]) == domain
                    and counts[split] + int(row["cluster_size"]) <= quotas[split]
                ]
                if not candidates:
                    raise RuntimeError(
                        "cannot preserve shared in-domain support for %s/%s"
                        % (task.task_id, domain)
                    )
                row = min(
                    candidates,
                    key=lambda value: (
                        gold_counts_by_split[(split, int(value["gold"]))]
                        >= MIN_GOLD_STRATUM,
                        str(value["order"]),
                    ),
                )
                assign(row, split)

    for gold in range(task.option_count):
        for split in split_order:
            while gold_counts_by_split[(split, gold)] < MIN_GOLD_STRATUM:
                candidates = [
                    row
                    for row in records
                    if str(row["item_cluster"]) not in assignments
                    and int(row["gold"]) == gold
                    and counts[split] + int(row["cluster_size"]) <= quotas[split]
                ]
                if not candidates:
                    raise RuntimeError(
                        "cannot preserve gold strata for %s/label-%d"
                        % (task.task_id, gold)
                    )
                assign(min(candidates, key=lambda value: str(value["order"])), split)

    for row in records:
        if str(row["item_cluster"]) in assignments:
            continue
        size = int(row["cluster_size"])
        available = [
            name for name in split_order if counts[name] + size <= quotas[name]
        ]
        if not available:
            continue
        name = min(
            available,
            key=lambda value: (
                counts[value] / max(1, quotas[value]),
                split_order.index(value),
            ),
        )
        assign(row, name)
    base.loc[base["split"] == "unassigned", "split"] = (
        base.loc[base["split"] == "unassigned", "item_cluster"]
        .map(assignments)
        .fillna("unused")
    )
    base["split_protocol"] = "tokenizer_intersection_then_whole_domain_ood"
    audit = _validate_split(task, base)
    requested = {
        "fit": task.fit,
        "validation": task.validation,
        "iid_test": task.iid_test,
        "ood_test": task.ood_test,
    }
    audit["requested_items"] = audit["split"].map(requested).astype(int)
    ood_at_or_above = (audit["split"] == "ood_test") & (
        audit["actual_items"] >= audit["requested_items"]
    )
    audit["count_rule"] = "cluster_disjoint_capacity_limited_below_target"
    audit.loc[
        (audit["split"] != "ood_test")
        & (audit["actual_items"] == audit["requested_items"]),
        "count_rule",
    ] = "cluster_disjoint_exact_target"
    audit.loc[
        ood_at_or_above, "count_rule"
    ] = "whole_domain_nearest_feasible_at_or_above_target"
    audit.loc[
        (audit["split"] == "ood_test") & ~ood_at_or_above, "count_rule"
    ] = "whole_domain_capacity_limited_below_target"
    iid_count = int(audit.loc[audit["split"] == "iid_test", "actual_items"].iloc[0])
    if iid_count != task.iid_test:
        raise RuntimeError(
            "registered IID count cannot be met without splitting a cluster for %s"
            % task.task_id
        )
    return base, audit, exclusion_audit


def build_candidate_manifest(
    output_dir: Path = CANONICAL_DIR,
    dataset_revisions: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    frames = []
    prompts = []
    filter_audits = []
    for task in TASKS:
        revision = dataset_revisions.get(task.dataset_id) if dataset_revisions else None
        items, filter_audit = load_semantic_items(task, revision)
        if revision is None:
            raise ValueError("dataset revision is required for the frozen manifest")
        frames.append(candidate_frame(task, items, revision))
        filter_audit["dataset_revision"] = revision
        filter_audits.append(filter_audit)
        for item in items:
            permutation = canonical_permutation(item.item_id, len(item.options))
            system, user = render_prompt(item, permutation)
            prompts.append(
                {
                    "task_id": item.task_id,
                    "item_id": item.item_id,
                    "permutation": json.dumps(permutation),
                    "system_prompt": system,
                    "user_prompt": user,
                    "prompt_hash": stable_hash([system, user]),
                }
            )
    manifest = pd.concat(frames, ignore_index=True)
    prompt_manifest = pd.DataFrame(prompts)
    filter_audit = pd.concat(filter_audits, ignore_index=True)
    key_columns = ["task_id", "item_id"]
    if (
        manifest.duplicated(key_columns).any()
        or prompt_manifest.duplicated(key_columns).any()
    ):
        raise RuntimeError("candidate manifest contains non-unique task/item keys")
    write_table(manifest, output_dir / "candidate_manifest.parquet")
    write_table(prompt_manifest, output_dir / "prompt_manifest.parquet")
    write_table(filter_audit, output_dir / "candidate_filter_audit.parquet")
    return manifest


def build_split_manifest(
    candidates: pd.DataFrame, output_dir: Path = CANONICAL_DIR
) -> pd.DataFrame:
    frames = []
    audits = []
    exclusion_audits = []
    for task in TASKS:
        selected = candidates[candidates["task_id"] == task.task_id].copy()
        split, audit, exclusion_audit = split_candidate_frame(task, selected)
        frames.append(split)
        audits.append(audit)
        exclusion_audits.append(exclusion_audit)
    manifest = pd.concat(frames, ignore_index=True)
    write_table(manifest, output_dir / "split_manifest.parquet")
    write_table(
        pd.concat(audits, ignore_index=True), output_dir / "split_contract.parquet"
    )
    write_table(
        pd.concat(exclusion_audits, ignore_index=True),
        output_dir / "split_cluster_exclusion_audit.parquet",
    )
    return manifest


def select_preflight_items(
    split_manifest: pd.DataFrame,
    tokenizer_audit: pd.DataFrame,
    item_count: int = PREFLIGHT_ITEMS,
) -> pd.DataFrame:
    active = split_manifest[
        split_manifest["split"].isin(("fit", "validation", "iid_test", "ood_test"))
    ][["task_id", "item_id"]].drop_duplicates()
    lengths = tokenizer_audit.merge(active, on=["task_id", "item_id"], how="inner")
    if lengths.empty:
        raise RuntimeError("preflight selection has no tokenizer-length audit")
    model_ids = set(lengths["model_id"].astype(str))
    if not model_ids:
        raise RuntimeError("preflight selection has no registered models")

    maximum = (
        lengths.groupby(["task_id", "item_id"], as_index=False)["rendered_tokens"]
        .max()
        .rename(columns={"rendered_tokens": "maximum_rendered_tokens"})
    )
    base, remainder = divmod(item_count, len(TASKS))
    selected = []
    for task_index, task in enumerate(TASKS):
        quota = base + (1 if task_index < remainder else 0)
        part = maximum[maximum["task_id"] == task.task_id].copy()
        if len(part) < quota:
            raise RuntimeError(
                "preflight quota exceeds the eligible pool for %s" % task.task_id
            )
        part["selection_order"] = part["item_id"].map(
            lambda value: stable_hash([task.task_id, value, "preflight_length_bin"])
        )
        part = part.sort_values(
            ["maximum_rendered_tokens", "selection_order"]
        ).reset_index(drop=True)
        part["length_bin"] = np.minimum(
            np.arange(len(part), dtype=int) * 4 // len(part), 3
        )

        anchors: Set[str] = set()
        task_lengths = lengths[lengths["task_id"] == task.task_id].copy()
        for model_id in sorted(model_ids):
            model_lengths = task_lengths[
                task_lengths["model_id"].astype(str) == model_id
            ].copy()
            if model_lengths.empty:
                raise RuntimeError(
                    "tokenizer audit omits %s/%s" % (model_id, task.task_id)
                )
            model_lengths["selection_order"] = model_lengths["item_id"].map(
                lambda value: stable_hash(
                    [task.task_id, model_id, value, "preflight_tail"]
                )
            )
            for quantile in (0.95, 0.99):
                tail = float(
                    model_lengths["rendered_tokens"].quantile(
                        quantile, interpolation="nearest"
                    )
                )
                row = (
                    model_lengths.assign(
                        distance=(model_lengths["rendered_tokens"] - tail).abs()
                    )
                    .sort_values(["distance", "selection_order"])
                    .iloc[0]
                )
                anchors.add(str(row["item_id"]))
        if len(anchors) > quota:
            raise RuntimeError("preflight tail anchors exceed the task quota")

        part["tail_anchor"] = part["item_id"].astype(str).isin(anchors)
        chosen = part[part["tail_anchor"]].copy()
        bin_counts = chosen["length_bin"].value_counts().to_dict()
        queues = {
            bin_id: part[(part["length_bin"] == bin_id) & ~part["tail_anchor"]]
            .sort_values("selection_order")
            .to_dict("records")
            for bin_id in range(4)
        }
        while len(chosen) < quota:
            available = [bin_id for bin_id in range(4) if queues[bin_id]]
            if not available:
                raise RuntimeError("preflight length-bin queues were exhausted")
            bin_id = min(available, key=lambda value: (bin_counts.get(value, 0), value))
            chosen = pd.concat(
                [chosen, pd.DataFrame([queues[bin_id].pop(0)])], ignore_index=True
            )
            bin_counts[bin_id] = bin_counts.get(bin_id, 0) + 1
        selected.append(chosen)

    output = pd.concat(selected, ignore_index=True)
    if len(output) != item_count or output.duplicated(["task_id", "item_id"]).any():
        raise RuntimeError("preflight selector did not produce unique registered items")
    return output[
        ["task_id", "item_id", "length_bin", "tail_anchor", "maximum_rendered_tokens"]
    ]
