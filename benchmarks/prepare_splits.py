#!/usr/bin/env python3
"""Freeze corrected, target-independent splits for the replacement benchmarks.

This creates a NEW protocol and never modifies the released experiment bundle.
Only item metadata and cache coverage keys are read, never model-response values
or test losses. The query distribution is uniform over canonical representatives
of all tokenizer-eligible, non-ambiguous candidate clusters outside the original
fixed OOD domains, including domains excluded by the old minimum-coverage policy.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (20261109, 20261110, 20261111, 20261112, 20261113)
SIZES = {
    "mmlu_pro": {"fit": 2048, "validation": 512, "iid_test": 1024},
    "hellaswag": {"fit": 2048, "validation": 512, "iid_test": 1024},
    "rewardbench": {"fit": 1024, "validation": 256, "iid_test": 512},
}
CORE_MODELS = (
    "Qwen/Qwen3-8B",
    "allenai/Olmo-3-7B-Instruct",
    "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "Qwen/Qwen3.5-9B",
    "google/gemma-4-12B-it",
    "ibm-granite/granite-3.3-8b-instruct",
)
VERSION = "replacement_corrected_uniform_cluster_representative_v1"
KEY = ["task_id", "item_id"]
CLUSTER_KEY = ["task_id", "item_cluster"]
METADATA = [
    "task_id", "item_id", "item_cluster", "domain", "gold",
    "dataset_revision", "split", "within_tokenizer_intersection",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*values: object) -> str:
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def freeze_protocol(path: Path, protocol: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError(
                f"Refusing to overwrite a different frozen protocol: {path}. "
                "Use a new output directory for a revised protocol."
            )
        return
    write_json(path, protocol)


def construct_population(original: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if original.duplicated(KEY).any():
        raise ValueError("Released manifest contains duplicate item keys")
    if not original["within_tokenizer_intersection"].all():
        raise ValueError("Released manifest includes tokenizer-ineligible items")
    rows = []
    summary = {}
    for task in SIZES:
        old = original[original.task_id == task].copy()
        source = candidates[(candidates.task_id == task) & candidates.within_tokenizer_intersection].copy()
        consistency = source.groupby("item_cluster").agg(
            domains=("domain", "nunique"), labels=("gold", "nunique")
        )
        ambiguous = set(consistency.loc[(consistency.domains > 1) | (consistency.labels > 1)].index)
        ood_clusters = set(old.loc[old.split == "ood_test", "item_cluster"])
        ood_domains = sorted(old.loc[old.split == "ood_test", "domain"].unique())
        pool = source[
            ~source.domain.isin(ood_domains) & ~source.item_cluster.isin(ood_clusters | ambiguous)
        ].copy()
        support = sorted(pool.domain.unique().tolist())
        old_support = set(old.loc[old.split.isin(SIZES[task]), "domain"])
        pool["split"] = pool.item_id.map(old.set_index("item_id").split).fillna("excluded_by_original_manifest")
        if set(support) & set(ood_domains):
            raise ValueError(f"Released in-domain/OOD support overlap for {task}")
        if not set(pool.split).issubset({"fit", "validation", "iid_test", "unused", "excluded_by_original_manifest"}):
            raise ValueError(f"Unexpected source split in {task} population")
        consistency = pool.groupby("item_cluster").agg(
            domains=("domain", "nunique"), labels=("gold", "nunique")
        )
        if ((consistency.domains != 1) | (consistency.labels != 1)).any():
            raise ValueError(f"Ambiguous duplicate cluster in {task} population")
        pool["representative_hash"] = [
            stable_hash(VERSION, "representative", task, cluster, item)
            for cluster, item in pool[["item_cluster", "item_id"]].itertuples(
                index=False, name=None
            )
        ]
        reps = pool.sort_values(["representative_hash", "item_id"]).drop_duplicates(
            "item_cluster", keep="first"
        )
        reps = reps.sort_values("item_id").rename(columns={"split": "original_split"})
        reps["population_weight"] = 1.0 / len(reps)
        reps["source_cluster_size"] = reps.item_cluster.map(
            pool.groupby("item_cluster").size()
        )
        if len(reps) < sum(SIZES[task].values()):
            raise ValueError(f"Not enough cluster representatives for {task}")
        summary[task] = {
            "in_domain_domains": support,
            "original_ood_domains_excluded": ood_domains,
            "original_ood_clusters_excluded": len(ood_clusters),
            "ambiguous_clusters_excluded": len(ambiguous),
            "ambiguous_items_excluded": int(source.item_cluster.isin(ambiguous).sum()),
            "restored_previously_excluded_domains": sorted(set(support) - old_support),
            "eligible_items_in_support": len(pool),
            "canonical_cluster_representatives": len(reps),
            "duplicate_items_removed": len(pool) - len(reps),
            "population_domain_weights": {
                str(domain): float(count / len(reps))
                for domain, count in reps.domain.value_counts().sort_index().items()
            },
            "population_label_weights": {
                str(label): float(count / len(reps))
                for label, count in reps.gold.value_counts().sort_index().items()
            },
        }
        rows.append(reps)
    return pd.concat(rows, ignore_index=True), summary


def make_manifest(population: pd.DataFrame, seed: int) -> pd.DataFrame:
    parts = []
    for task, sizes in SIZES.items():
        part = population[population.task_id == task].sort_values("item_id").copy()
        part["split"] = "unused"
        rng_seed = int(stable_hash(VERSION, "partition", seed, task), 16)
        rng = np.random.Generator(np.random.PCG64(rng_seed))
        order = rng.permutation(len(part))
        offset = 0
        split_column = part.columns.get_loc("split")
        for split, count in sizes.items():
            part.iloc[order[offset:offset + count], split_column] = split
            offset += count
        part["seed"] = seed
        part["split_protocol"] = VERSION
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def audit_manifest(manifest: pd.DataFrame, original: pd.DataFrame) -> list[dict]:
    audits = []
    for task, sizes in SIZES.items():
        part = manifest[manifest.task_id == task]
        old = original[original.task_id == task]
        ood = set(old.loc[old.split == "ood_test", "item_cluster"])
        cluster_sets = {}
        for split, expected in sizes.items():
            selected = part[part.split == split]
            clusters = set(selected.item_cluster)
            assert len(selected) == expected
            assert len(clusters) == expected
            assert not (clusters & ood)
            cluster_sets[split] = clusters
            audits.append({
                "seed": int(part.seed.iloc[0]), "task_id": task,
                "check": "partition_size_and_ood_exclusion", "split_a": split,
                "split_b": "original_ood", "expected_items": expected,
                "actual_items": len(selected), "actual_clusters": len(clusters),
                "overlap_clusters": len(clusters & ood), "passed": True,
            })
        for left, right in itertools.combinations(sizes, 2):
            overlap = cluster_sets[left] & cluster_sets[right]
            assert not overlap
            audits.append({
                "seed": int(part.seed.iloc[0]), "task_id": task,
                "check": "within_seed_cluster_disjoint", "split_a": left,
                "split_b": right, "expected_items": None, "actual_items": None,
                "actual_clusters": None, "overlap_clusters": len(overlap),
                "passed": True,
            })
    return audits


def count_tables(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    domain_rows, label_rows, joint_rows = [], [], []
    seed = int(manifest.seed.iloc[0])
    for task, sizes in SIZES.items():
        population = manifest[manifest.task_id == task]
        domains = sorted(population.domain.unique())
        labels = sorted(int(v) for v in population.gold.unique())
        pop_domains = population.domain.value_counts()
        pop_labels = population.gold.value_counts()
        for split in [*sizes, "unused"]:
            selected = population[population.split == split]
            total = len(selected)
            base = {"seed": seed, "task_id": task, "split": split, "partition_items": total}
            ds = selected.domain.value_counts()
            ls = selected.gold.value_counts()
            js = selected.groupby(["domain", "gold"]).size()
            for domain in domains:
                n = int(ds.get(domain, 0))
                domain_rows.append({**base, "domain": domain, "count": n,
                    "fraction": n / total if total else 0.0,
                    "population_fraction": float(pop_domains[domain] / len(population))})
            for label in labels:
                n = int(ls.get(label, 0))
                label_rows.append({**base, "gold": label, "count": n,
                    "fraction": n / total if total else 0.0,
                    "population_fraction": float(pop_labels[label] / len(population))})
            for domain, label in itertools.product(domains, labels):
                n = int(js.get((domain, label), 0))
                joint_rows.append({**base, "domain": domain, "gold": label,
                    "count": n, "fraction": n / total if total else 0.0})
    return pd.DataFrame(domain_rows), pd.DataFrame(label_rows), pd.DataFrame(joint_rows)


def cache_coverage(bundle: Path, all_splits: pd.DataFrame, output: Path) -> dict:
    required = all_splits[all_splits.split != "unused"].copy()
    union = required.drop_duplicates(KEY).sort_values(KEY)
    prompts = pd.read_parquet(bundle / "data/canonical/prompt_manifest.parquet")
    union_metadata = union[KEY + ["item_cluster", "domain", "gold", "dataset_revision"]]
    required_prompts = union_metadata.merge(prompts, on=KEY, how="left", validate="one_to_one")
    if required_prompts[["prompt_hash", "system_prompt", "user_prompt", "permutation"]].isna().any().any():
        raise ValueError("Selected items lack original prompt records")
    required_prompts.to_parquet(output / "required_items.parquet", index=False)
    by_partition, by_union, missing = [], [], []
    seen = set()
    for path in sorted((bundle / "data/canonical/response_cache").glob("*.parquet")):
        # Deliberately restrict reading to cache identity/coverage metadata.
        cached = pd.read_parquet(path, columns=["model_id", "task_id", "item_id", "revision"])
        model_ids = cached.model_id.unique()
        if len(model_ids) != 1 or cached.duplicated(KEY).any():
            raise ValueError(f"Invalid response-cache keys: {path}")
        model_id = str(model_ids[0])
        seen.add(model_id)
        keys = set(cached[KEY].itertuples(index=False, name=None))
        availability = required[KEY].apply(tuple, axis=1).isin(keys)
        marked = required.assign(response_cached=availability)
        for (seed, task, split), group in marked.groupby(["seed", "task_id", "split"]):
            by_partition.append({"model_id": model_id, "core_model": model_id in CORE_MODELS,
                "seed": int(seed), "task_id": task, "split": split,
                "required": len(group), "cached": int(group.response_cached.sum()),
                "missing": int((~group.response_cached).sum())})
        unique = union.assign(response_cached=union[KEY].apply(tuple, axis=1).isin(keys))
        for task, group in unique.groupby("task_id"):
            by_union.append({"model_id": model_id, "core_model": model_id in CORE_MODELS,
                "task_id": task, "required": len(group),
                "cached": int(group.response_cached.sum()),
                "missing": int((~group.response_cached).sum())})
        needed = unique[~unique.response_cached][KEY + ["item_cluster", "domain", "gold", "dataset_revision", "prompt_hash"]].copy()
        needed["model_id"] = model_id
        needed["core_model"] = model_id in CORE_MODELS
        missing.append(needed)
    if not set(CORE_MODELS).issubset(seen):
        raise ValueError("Missing complete response-cache file for a core model")
    partition = pd.DataFrame(by_partition)
    unique = pd.DataFrame(by_union)
    partition.to_csv(output / "cache_coverage_by_partition.csv", index=False)
    unique.to_csv(output / "cache_coverage_union.csv", index=False)
    pd.concat(missing, ignore_index=True).to_parquet(output / "missing_responses.parquet", index=False)
    return {
        "unique_selected_items_across_five_seeds": len(union),
        "missing_model_item_responses_core_six": int(unique.loc[unique.core_model, "missing"].sum()),
        "missing_model_item_responses_all_nine": int(unique["missing"].sum()),
        "cache_values_or_losses_read": False,
    }


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=default_root / "reference_implementation")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "splits")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = bundle / "data/canonical/split_manifest.parquet"
    original = pd.read_parquet(source, columns=METADATA)
    candidate_path = bundle / "data/canonical/candidate_manifest.parquet"
    candidates = pd.read_parquet(candidate_path, columns=KEY + ["item_cluster", "domain", "gold", "dataset_revision"])
    tokenizer = pd.read_parquet(bundle / "data/canonical/tokenizer_length_audit.parquet", columns=KEY + ["model_id", "within_limit"])
    if tokenizer.duplicated(["model_id"] + KEY).any():
        raise ValueError("Duplicate model/item tokenizer audit keys")
    model_count = tokenizer.model_id.nunique()
    eligibility = tokenizer.groupby(KEY, as_index=False).agg(
        within_tokenizer_intersection=("within_limit", "all"), audited_models=("model_id", "nunique")
    )
    if (eligibility.audited_models != model_count).any():
        raise ValueError("Incomplete model coverage in tokenizer intersection audit")
    candidates = candidates.merge(eligibility, on=KEY, how="left", validate="one_to_one")
    if candidates.within_tokenizer_intersection.isna().any():
        raise ValueError("Candidates lack tokenizer intersection audit")
    prompts = pd.read_parquet(bundle / "data/canonical/prompt_manifest.parquet", columns=KEY + ["prompt_hash"])
    candidates = candidates.merge(prompts, on=KEY, how="left", validate="one_to_one")
    if candidates.prompt_hash.isna().any():
        raise ValueError("Candidates lack prompt hashes")
    population, population_summary = construct_population(original, candidates)
    protocol = {
        "protocol_version": VERSION,
        "status": "new_benchmarks_replication_protocol_frozen_before_response_analysis",
        "seeds": list(SEEDS), "partition_sizes": SIZES,
        "query_distribution": (
            "For each task, uniform over one target-independent canonical representative per "
            "tokenizer-eligible, non-ambiguous duplicate cluster outside the original fixed OOD domains "
            "and OOD clusters. All original unused items and all domains previously excluded solely "
            "by minimum-domain-coverage requirements are eligible."
        ),
        "sampling": (
            "Unstratified uniform permutation without replacement, using explicit NumPy PCG64 "
            "seeded with SHA256(version, partition, seed, task), then sequential fit/validation/IID slices. "
            "No alphabetical priority, minimum-domain quotas, response-based filtering, or label balancing. "
            "All three partitions target the same probability law; realized domain/label fractions "
            "may differ by finite-sample randomness and are reported in full."
        ),
        "representative_selection": (
            "Minimum SHA256(version, representative, task_id, item_cluster, item_id), "
            "with item_id as tie-breaker, independent of target and model responses."
        ),
        "loss_weighting": "Uniform per representative within each task and partition; no post-hoc domain reweighting.",
        "aggregation_scope": "Three named target models x three named tasks; any across-cell aggregate must weight cells explicitly.",
        "repeated_seed_caveat": (
            "Seeds reuse the same finite candidate population and may overlap across seeds. "
            "Do not treat item-seed observations as independent or claim five independent datasets. "
            "Published response arrays were already available, so these are newly frozen corrected "
            "resplits, not a claim of an untouched historical test set."
        ),
        "scope_caveat": (
            "The population remains conditional on the original fixed OOD exclusions, option-count "
            "eligibility, tokenizer-length eligibility, and unambiguous duplicate clusters. Previously "
            "excluded sparse HellaSwag domains are restored. Uniform cluster representatives intentionally "
            "give every duplicate cluster equal mass, unlike uniform raw items."
        ),
        "population": population_summary,
        "core_models": list(CORE_MODELS),
        "source_files_sha256": {
            str(source.relative_to(bundle)): sha256_file(source),
            "data/canonical/candidate_manifest.parquet": sha256_file(bundle / "data/canonical/candidate_manifest.parquet"),
            "data/canonical/prompt_manifest.parquet": sha256_file(bundle / "data/canonical/prompt_manifest.parquet"),
            "data/canonical/tokenizer_length_audit.parquet": sha256_file(bundle / "data/canonical/tokenizer_length_audit.parquet"),
            "configs/llm_revisions.json": sha256_file(bundle / "configs/llm_revisions.json"),
        },
        "split_preparation_script_sha256": sha256_file(Path(__file__)),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
    }
    # Freeze distribution, seeds, sizes and source identity before creating splits
    # or inspecting model cache coverage. Outcome values are never accessed.
    freeze_protocol(output / "protocol.json", protocol)
    population.to_parquet(output / "population.parquet", index=False)
    manifests, audits, domain_tables, label_tables, joint_tables = [], [], [], [], []
    for seed in SEEDS:
        manifest = make_manifest(population, seed)
        # Input row order cannot influence assignments or representative identity.
        again = make_manifest(population.iloc[::-1], seed)
        pd.testing.assert_frame_equal(manifest, again)
        audits.extend(audit_manifest(manifest, original))
        domain, label, joint = count_tables(manifest)
        domain_tables.append(domain)
        label_tables.append(label)
        joint_tables.append(joint)
        manifest.to_parquet(output / f"split_seed_{seed}.parquet", index=False)
        manifests.append(manifest)
    all_splits = pd.concat(manifests, ignore_index=True)
    pd.concat(domain_tables, ignore_index=True).to_csv(output / "domain_counts.csv", index=False)
    pd.concat(label_tables, ignore_index=True).to_csv(output / "label_counts.csv", index=False)
    pd.concat(joint_tables, ignore_index=True).to_csv(output / "domain_label_counts.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "overlap_audit.csv", index=False)
    cross_seed = []
    for task in SIZES:
        for left, right in itertools.combinations(SEEDS, 2):
            a = set(all_splits.loc[(all_splits.task_id == task) & (all_splits.seed == left) & (all_splits.split == "iid_test"), "item_cluster"])
            b = set(all_splits.loc[(all_splits.task_id == task) & (all_splits.seed == right) & (all_splits.split == "iid_test"), "item_cluster"])
            assert a != b, "Different seeds generated identical test sets"
            cross_seed.append({"task_id": task, "seed_a": left, "seed_b": right,
                "test_clusters_a": len(a), "test_clusters_b": len(b),
                "overlap_test_clusters": len(a & b)})
    pd.DataFrame(cross_seed).to_csv(output / "cross_seed_overlap.csv", index=False)
    coverage = cache_coverage(bundle, all_splits, output)
    summary = {"protocol_sha256": sha256_file(output / "protocol.json"),
        "seeds": list(SEEDS), "population_sizes": {
            task: record["canonical_cluster_representatives"] for task, record in population_summary.items()
        }, "within_seed_audits_passed": len(audits),
        "input_order_invariance_verified": True, **coverage}
    write_json(output / "summary.json", summary)
    artifacts = {path.name: sha256_file(path) for path in sorted(output.iterdir())
        if path.is_file() and path.name not in {"artifact_hashes.json", "freeze_record.json"}}
    write_json(output / "artifact_hashes.json", artifacts)
    if not (output / "freeze_record.json").exists():
        write_json(output / "freeze_record.json", {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": summary["protocol_sha256"],
            "note": "New corrective experiment; not retrospective preregistration of the published results.",
        })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
